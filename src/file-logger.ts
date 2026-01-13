import fs from 'fs/promises';
import path from 'path';

export class FileLogger {
  private static instance: FileLogger;

  private constructor() {}

  public static getInstance(): FileLogger {
    if (!FileLogger.instance) {
      FileLogger.instance = new FileLogger();
    }
    return FileLogger.instance;
  }

  /**
   * Appends a log entry to a file specific to the processed file.
   * @param filePath The path of the file being processed (e.g., 'image.png')
   * @param stage The stage of processing (e.g., 'Prompt', 'Response', 'Error')
   * @param data The data to log (string or object)
   */
  async log(filePath: string, stage: string, data: any): Promise<void> {
    try {
      const logFile = `${filePath}.aifiles.log`;
      const timestamp = new Date().toISOString();

      let content = '';
      if (typeof data === 'string') {
        content = data;
      } else if (data instanceof Error) {
        content = data.stack || data.message;
      } else {
        try {
          content = JSON.stringify(data, null, 2);
        } catch (e) {
          content = String(data);
        }
      }

      const entry = `\n[${timestamp}] [${stage}]\n${content}\n${'-'.repeat(80)}\n`;

      await fs.appendFile(logFile, entry, 'utf-8');
    } catch (error) {
      // Silently fail to avoid crashing the main process
      console.warn(`Failed to write to log file for ${filePath}`, error);
    }
  }
}
