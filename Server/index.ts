import express from 'express';
import type { Request, Response, Express } from 'express';
import cors from 'cors';
import axios from 'axios';
import { CONFIG } from 'config';

// Explicitly define the Axios interface
interface AxiosInstance {
  post<T = any>(url: string, data?: any, config?: any): Promise<{ data: T }>;
}

// const axiosInstance = axios as unknown as AxiosInstance;
// Create axios instance with defaults
const axiosInstance = axios.create({
  timeout: CONFIG.TIMEOUT,
  maxContentLength: 5 * 1024 * 1024, // 5MB limit
  maxBodyLength: 5 * 1024 * 1024,    // 5MB limit
  headers: { 'Content-Type': 'application/json' }
});
// Configuration


const app: Express = express();
app.use(cors());
app.use(express.json());

// Define request body type
interface GenerateRequest {
  goal: string;
}

// Define response type from AI service
interface Step {
  description: string;
  level: string;
  resources: { title: string; link: string }[];
}

interface AIResponse {
  steps: Step[];
}



// Add retry utility
async function withRetry<T>(operation: () => Promise<T>): Promise<T> {
  let lastError: Error;
  
  for (let attempt = 1; attempt <= CONFIG.MAX_RETRIES; attempt++) {
      try {
          return await operation();
      } catch (error) {
          lastError = error as Error;
          if (attempt < CONFIG.MAX_RETRIES) {
              console.log(`Attempt ${attempt} failed. Retrying...`);
              await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
          }
      }
  }
  
  throw lastError!;
}

// Explicitly type the handler
app.post('/generate', async (req: Request, res: Response): Promise<void> => {
  try {
    const { goal } = req.body as GenerateRequest;
    
    if (!goal?.trim()) {
      res.status(400).json({ error: 'Goal is required' });
      return;
    }
    
    const aiResponse = await withRetry(() => 
      axiosInstance.post<AIResponse>(
        CONFIG.AI_SERVICE_URL,
        { text: goal },
      )
    );
    
    console.log('AI Response:', aiResponse.data); // Debugging
    res.json(aiResponse.data); // Forward steps to frontend
  } catch (error: unknown) {
    console.error('Error in /generate:', error instanceof Error ? error.message : 'Unknown error');
  // Improved error responses
    if (error && typeof error === 'object' && 'code' in error) {
      if (error.code === 'ECONNABORTED') {
        res.status(504).json({ error: 'Request timed out after multiple retries' });
      } else if (error.code === 'ECONNREFUSED') {
        res.status(503).json({ error: 'AI service is unavailable' });
      } else {
        res.status(500).json({ error: 'Failed to generate roadmap' });
      }
    } else {
      res.status(500).json({ error: 'Failed to generate roadmap' });
    }
  }
});

// Error handling middleware
app.use((err: Error, req: Request, res: Response, _next: any) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server with memory handling
const server = app.listen(CONFIG.PORT, () => {
  console.log(`Server running on port ${CONFIG.PORT}`);
});

// Handle server shutdown gracefully
process.on('SIGTERM', () => {
  server.close(() => {
    console.log('Server shutdown complete');
    process.exit(0);
  });
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
  server.close(() => {
    process.exit(1);
  });
});