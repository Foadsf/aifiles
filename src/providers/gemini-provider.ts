import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import { LLMProvider } from './base-provider.js';
import fs from 'fs/promises';

export class GeminiProvider implements LLMProvider {
  name = 'Gemini';
  private apiKey: string;
  private modelName: string;
  private genAI: GoogleGenerativeAI;
  private model: GenerativeModel;

  constructor(
    apiKey: string,
    model: string = 'gemini-3.0-pro'
  ) {
    this.apiKey = apiKey;
    this.modelName = model;
    this.genAI = new GoogleGenerativeAI(this.apiKey);
    this.model = this.genAI.getGenerativeModel({ model: this.modelName });
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

  async sendMessage(prompt: string): Promise<string> {
    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error: any) {
      this.handleGeminiError(error, 'sendMessage');
    }
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

      const result = await this.model.generateContent([prompt, imagePart]);
      const response = await result.response;
      return response.text();
    } catch (error: any) {
      this.handleGeminiError(error, 'vision');
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
