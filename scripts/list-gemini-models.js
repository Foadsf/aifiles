
import fs from 'fs';
import path from 'path';
import os from 'os';

// Helper to read config file
function readConfig() {
  const configPath = path.join(os.homedir(), '.aifiles', 'config');
  try {
    if (fs.existsSync(configPath)) {
      const content = fs.readFileSync(configPath, 'utf-8');
      const lines = content.split('\n');
      const config = {};
      for (const line of lines) {
        if (line.trim() && !line.startsWith('#')) {
          const [key, value] = line.split('=');
          if (key && value) {
            config[key.trim()] = value.trim();
          }
        }
      }
      return config;
    }
  } catch (e) {
    // Ignore error
  }
  return {};
}

export async function getGeminiModels(apiKey) {
  try {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const models = [];

    if (data.models && Array.isArray(data.models)) {
        data.models.forEach((model) => {
            // Filter for generateContent supported models usually used for chat/text
            if (model.supportedGenerationMethods && model.supportedGenerationMethods.includes('generateContent')) {
                 models.push({
                    name: model.name.replace('models/', ''),
                    displayName: model.displayName,
                    version: model.version,
                    description: model.description
                 });
            }
        });
    }
    return models;

  } catch (error) {
    throw new Error(`Failed to list models: ${error.message}`);
  }
}

async function main() {
  const config = readConfig();
  const apiKey = process.env.AIFILES_GEMINI_API_KEY ||
                 process.env.GEMINI_API_KEY ||
                 config.GEMINI_API_KEY;

  if (!apiKey) {
    console.error('Error: Gemini API key not found.');
    console.error('Please set AIFILES_GEMINI_API_KEY environment variable or ensure it is in ~/.aifiles/config');
    process.exit(1);
  }

  try {
    console.log('Fetching available models...');
    const models = await getGeminiModels(apiKey);

    if (models.length > 0) {
        console.log('\nAvailable Gemini Models:');
        console.log('------------------------');
        models.forEach(model => {
             console.log(`Name: ${model.name}`);
             console.log(`Display Name: ${model.displayName}`);
             console.log(`Version: ${model.version}`);
             console.log(`Description: ${model.description}`);
             console.log('------------------------');
        });
    } else {
        console.log('No models found or unexpected response format.');
    }
  } catch (error) {
      console.error(error.message);
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
