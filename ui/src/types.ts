export type Source = { 
  title: string; 
  url: string; 
};

export type AskResponse = { 
  answer: string; 
  sources: string[];  // Backend sends array of strings
  latency_ms?: number; // Optional latency field from backend
};