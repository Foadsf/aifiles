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
      generationConfig: { responseMimeType: 'application/json' }
    });
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
        return this.cleanJsonOutput(text);
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

      return this.generateContentWithRetry([prompt, imagePart], 'vision');
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
