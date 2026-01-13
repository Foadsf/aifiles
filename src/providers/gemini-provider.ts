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

  async sendMessage(prompt: string): Promise<string> {
    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error: any) {
      throw new Error(`Gemini API error: ${error.message}`);
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
