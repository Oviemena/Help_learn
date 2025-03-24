import express, { type Express, type Request, type Response } from 'express';
import cors from 'cors';
import axios from 'axios';

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

// Explicitly type the handler
app.post('/generate', async (req: Request<{}, {}, GenerateRequest>, res: Response<AIResponse | { error: string }>): Promise<void> => {
  try {
    const { goal } = req.body;
    if (!goal) {
      res.status(400).json({ error: 'Goal is required' });
      return;
    }

    const aiResponse = await axios.post<AIResponse>(
      'http://localhost:5001/analyze',
      { text: goal },
      { timeout: 10000 } // 10s timeout
    );

    console.log('AI Response:', aiResponse.data); // Debugging
    res.json(aiResponse.data); // Forward steps to frontend
  } catch (error) {
    console.error('Error in /generate:', error instanceof Error ? error.message : 'Unknown error');
    res.status(500).json({ error: 'Failed to generate roadmap' });
  }
});

const PORT: number | string = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});