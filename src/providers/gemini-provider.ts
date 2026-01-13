import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import { LLMProvider } from './base-provider.js';
import fs from 'fs/promises';

export class GeminiProvider implements LLMProvider {
  name = 'Gemini';
  private apiKey: string;
  private modelName: string;
  private genAI: GoogleGenerativeAI;
  private model: GenerativeModel;

  private readonly MAX_FREE_TIER_CHARS = 400000;
  private readonly MAX_RETRIES = 3;

  constructor(
    apiKey: string,
    model: string = 'gemini-3.0-pro'
  ) {
    this.apiKey = apiKey;
    this.modelName = model;
    this.genAI = new GoogleGenerativeAI(this.apiKey);
    this.model = this.genAI.getGenerativeModel({
      model: this.modelName,
      generationConfig: { responseMimeType: 'application/json' },
      systemInstruction: 'Return a flat JSON object. Do not nest the result under any root key like "analysis" or "json".'
    });
  }

  private findDataObject(obj: any): any {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
      return obj;
    }

    const keys = Object.keys(obj);

    // If it has expected keys, return it
    const hasExpectedKeys = keys.some(k =>
      ['file_title', 'file_category', 'file_tags', 'file_summary'].includes(k)
    );
    if (hasExpectedKeys) return obj;

    // If single key pointing to object, go deeper
    if (keys.length === 1 && typeof obj[keys[0]] === 'object' && !Array.isArray(obj[keys[0]])) {
      return this.findDataObject(obj[keys[0]]);
    }

    // If multiple keys but one looks like a wrapper (e.g. "response", "data", "analysis")
    // and its value is an object, prefer that.
    const wrapperKeys = ['analysis', 'result', 'response', 'data', 'content', 'output', 'json'];
    for (const key of wrapperKeys) {
        if (obj[key] && typeof obj[key] === 'object' && !Array.isArray(obj[key])) {
            return this.findDataObject(obj[key]);
        }
    }

    return obj;
  }

  private normalizeResponse(jsonStr: string): string {
    try {
      let data = JSON.parse(jsonStr);

      // 1. Recursive search for the data object
      data = this.findDataObject(data);

      // 2. Remap keys to application schema (title, summary, category, etc.)
      const normalized: any = {};

      // Mapping table: Input Key (lower) -> Target Key
      const map: Record<string, string> = {
        'file_title': 'title', 'title': 'title', 'filename': 'title', 'name': 'title',
        'file_category': 'category', 'category': 'category', 'classification': 'category', 'type': 'category',
        'file_summary': 'summary', 'summary': 'summary', 'description': 'summary', 'abstract': 'summary',
        'file_tags': 'tags', 'tags': 'tags', 'topics': 'tags',
        'file_keywords': 'keywords', 'keywords': 'keywords',
        'suggestedpath': 'suggestedPath', 'path': 'suggestedPath',
        'suggestedfilename': 'suggestedFilename',
        'confidence': 'confidence'
      };

      for (const [key, value] of Object.entries(data)) {
         const lowerKey = key.toLowerCase();
         if (map[lowerKey]) {
             // If target already set, don't overwrite unless current value is truthy and previous was falsy
             if (!normalized[map[lowerKey]]) {
                 normalized[map[lowerKey]] = value;
             }
         } else {
             // Preserve other keys (mainTopic, contentType, subcategories, etc.)
             normalized[key] = value;
         }
      }

      // 3. Enforce defaults and types
      if (!normalized.title) normalized.title = 'Untitled';
      if (!normalized.category) normalized.category = 'General';
      if (!normalized.summary) normalized.summary = '';
      if (!normalized.tags) normalized.tags = [];
      if (!normalized.keywords) normalized.keywords = [];

      // Ensure arrays
      if (!Array.isArray(normalized.tags)) normalized.tags = [String(normalized.tags)];
      if (!Array.isArray(normalized.keywords)) normalized.keywords = [String(normalized.keywords)];

      // 4. Synthesize suggestedPath/Filename if missing
      if (!normalized.suggestedPath) {
        const safeTitle = String(normalized.title).replace(/[^a-zA-Z0-9_-]/g, '_');
        const safeCategory = String(normalized.category).replace(/[^a-zA-Z0-9_-]/g, '_');
        normalized.suggestedPath = `${safeCategory}/${safeTitle}`;
      }

      if (!normalized.suggestedFilename) {
         const safeTitle = String(normalized.title).replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
         normalized.suggestedFilename = `${safeTitle}.txt`;
      }

      // 5. Confidence
      if (typeof normalized.confidence !== 'number') {
          normalized.confidence = 0.8;
      }

      const finalJson = JSON.stringify(normalized);
      console.log('Final JSON Payload Length:', finalJson.length);
      return finalJson;
    } catch (e) {
      // If parsing fails, we return the cleaned string and let the consumer try to handle it
      return jsonStr;
    }
  }

  private cleanJsonOutput(text: string): string {
    let cleanText = text.trim();
    // Remove markdown code fences if present
    if (cleanText.startsWith('```json')) {
      cleanText = cleanText.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (cleanText.startsWith('```')) {
      cleanText = cleanText.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    // Find the first '{' and last '}' to extract JSON object
    const firstBrace = cleanText.indexOf('{');
    const lastBrace = cleanText.lastIndexOf('}');

    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      return cleanText.substring(firstBrace, lastBrace + 1);
    }

    return cleanText;
  }

  private handleGeminiError(error: any, context: string): never {
    const errorMessage = error.message || '';
    if (errorMessage.includes('404') || errorMessage.toLowerCase().includes('not found')) {
      throw new Error(
        `Gemini API error (${context}): The model '${this.modelName}' was not found. ` +
        `Please verify the model name or run 'npm run list-models' to see available models.`
      );
    }
    throw new Error(`Gemini API error (${context}): ${errorMessage}`);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async generateContentWithRetry(params: any, context: string): Promise<string> {
    let attempts = 0;
    while (attempts <= this.MAX_RETRIES) {
      try {
        const result = await this.model.generateContent(params);
        const response = await result.response;
        const text = response.text();
        const cleaned = this.cleanJsonOutput(text);
        return this.normalizeResponse(cleaned);
      } catch (error: any) {
        attempts++;
        const errorMessage = error.message || '';

        // Check for 429 or rate limit message
        // The error might contain "429" or text like "Please retry in 54s"
        const isRateLimit = errorMessage.includes('429') ||
                            errorMessage.toLowerCase().includes('too many requests') ||
                            errorMessage.toLowerCase().includes('quota') ||
                            errorMessage.toLowerCase().includes('retry in');

        if (isRateLimit && attempts <= this.MAX_RETRIES) {
            // Try to parse wait time
            // Example: "Please retry in 54s" or "Retry after 60 seconds"
            const match = errorMessage.match(/retry (?:in|after) ([0-9.]+)s/i);
            let waitSeconds = 60; // Default wait

            if (match && match[1]) {
                waitSeconds = parseFloat(match[1]);
            }

            console.warn(`⚠️ Quota exceeded (Attempt ${attempts}/${this.MAX_RETRIES}). Waiting ${waitSeconds}s as requested by API...`);
            await this.sleep((waitSeconds * 1000) + 1000); // Add 1s buffer
            continue;
        }

        // If not rate limit or max retries reached, throw
        if (attempts > this.MAX_RETRIES) {
           this.handleGeminiError(error, context);
        } else {
            // If it's another kind of error (like 404), throw immediately without retry
             this.handleGeminiError(error, context);
        }
      }
    }
    throw new Error(`Gemini API error (${context}): Max retries exceeded`);
  }

  async sendMessage(prompt: string): Promise<string> {
    let content = prompt;
    if (content.length > this.MAX_FREE_TIER_CHARS) {
      console.warn(`File content (${content.length} chars) exceeds free tier limit. Truncated to ${this.MAX_FREE_TIER_CHARS} chars to prevent API errors.`);
      content = content.substring(0, this.MAX_FREE_TIER_CHARS);
    }
    // Append strict format instruction
    content += '\n\nIMPORTANT: Return a FLAT JSON object. Keys MUST be exactly: file_title, file_category, file_tags, file_summary.';

    return this.generateContentWithRetry(content, 'sendMessage');
  }

  async analyzeImage(imagePath: string, prompt: string = 'Describe this image concisely'): Promise<string> {
    try {
      const imageBuffer = await fs.readFile(imagePath);
      const base64Image = imageBuffer.toString('base64');
      const ext = imagePath.split('.').pop()?.toLowerCase() || 'jpg';
      const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';

      const imagePart = {
        inlineData: {
          data: base64Image,
          mimeType: mimeType,
        },
      };

      const promptWithFormat = prompt + '\n\nIMPORTANT: Return a FLAT JSON object. Keys MUST be exactly: file_title, file_category, file_tags, file_summary.';

      return this.generateContentWithRetry([promptWithFormat, imagePart], 'vision');
    } catch (error: any) {
        // readFile errors might be caught here if they happen before generateContentWithRetry
        throw new Error(`Gemini vision API error: ${error.message}`);
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      // Basic check: Ensure API key is present
      return !!this.apiKey;
    } catch {
      return false;
    }
  }
}
