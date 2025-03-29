export const CONFIG = {
  PORT: process.env.PORT || 5000,
  AI_SERVICE_URL: 'http://localhost:5001/analyze',
  TIMEOUT: 30000,
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000,
  MEMORY_LIMIT: '512mb'
};