import React, { useState, type FormEvent } from 'react';
import './index.css';
import { API_URL } from 'Config/config';

// Define types for API response (shared with backend)
interface Resource {
  title: string;
  link: string;
}

interface Step {
  description: string;
  level: string;
  resources: Resource[];
  isOffline?: boolean;
}

interface Roadmap {
  steps: Step[];
}

const FALLBACK_RESOURCES = {
  beginner: [
    'https://www.freecodecamp.org',
    'https://www.w3schools.com',
    'https://developer.mozilla.org'
  ],
  intermediate: [
    'https://medium.com/topic/programming',
    'https://dev.to',
    'https://lab.github.com'
  ],
  advanced: [
    'https://github.com/topics',
    'https://stackoverflow.com',
    'https://arxiv.org/list/cs/recent'
  ]
};

const isAllFallback = (steps: Step[]) => {
  return steps.every(step => {
    const levelUrls = FALLBACK_RESOURCES[step.level as keyof typeof FALLBACK_RESOURCES];
    return step.resources.every(res => levelUrls.includes(res.link));
  });
};

const App: React.FC = () => {
  const [goal, setGoal] = useState<string>('');
  const [roadmap, setRoadmap] = useState<Roadmap | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setRoadmap(null);

    try {

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 35000); // 35s timeout

      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ goal }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data: Roadmap = await response.json();
      setRoadmap(data);
      console.log(data)
    } catch (err) {
      setError('Something went wrong. Please try again.');
      console.error('Frontend error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[900px] mx-auto p-8 font-sans bg-[#f4f7fa] min-h-screen">
      <h1 className="text-center text-[#2c3e50] mb-8 text-4xl">
        Personalized Learning Path Generator
      </h1>

      <form onSubmit={handleSubmit} className="flex justify-center gap-4">
        <input
          type="text"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="E.g., Learn Python for data analysis"
          className="w-3/5 p-3 border-2 border-[#dfe6e9] rounded-md text-base 
                     transition-colors duration-300 focus:border-[#0984e3] focus:outline-none"
          disabled={loading}
        />
        <button
          type="submit"
          className="px-6 py-3 bg-[#0984e3] text-white border-none rounded-md text-base 
                     cursor-pointer transition-colors duration-300 
                     hover:enabled:bg-[#0652dd] disabled:bg-[#b2bec3] disabled:cursor-not-allowed"
          disabled={loading}
        >
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </form>

      {error && (
        <p className="text-[#e74c3c] text-center mt-5 text-lg">{error}</p>
      )}

      {roadmap && (
        <div className="mt-10 p-6 bg-white rounded-lg shadow-md">
          {roadmap && (
            <div className="mt-10 p-6 bg-white rounded-lg shadow-md">
              {isAllFallback(roadmap.steps) && (
                <div className="mb-6 p-4 bg-yellow-50 border-l-4 border-yellow-400">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-yellow-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <p className="text-sm text-yellow-700">
                      No Internet Connection - Using Default Resources
                    </p>
                  </div>
                </div>
              )}
              <h2 className="text-[#2c3e50] mb-5 text-2xl">Your Learning Roadmap</h2>
              <ol className="pl-5">
                {roadmap.steps.map((step, index) => (
                  <li key={index} className="mb-6 leading-relaxed">
                    <strong>{step.description}</strong>
                    <span className="text-[#636e72] text-sm ml-1.5">({step.level})</span>
                    <ul className="mt-2.5 pl-5 list-disc">
                      {step.resources.map((res, i) => (
                        <li key={i} className="mb-2">
                          <a
                            href={res.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[#0984e3] no-underline transition-colors duration-300 
                                 hover:text-[#0652dd] hover:underline"
                          >
                            {res.title}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )
      }
    </div>
  );
};

export default App;