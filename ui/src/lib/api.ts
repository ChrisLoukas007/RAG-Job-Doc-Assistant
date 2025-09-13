const API_BASE = import.meta.env.VITE_API_BASE as string;

// Send question to backend and get answer with sources
export async function ask(question: string) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as { answer: string; sources: string[]; latency_ms?: number };
}

// Streaming answer from backend with sources
export function stream(question: string, handlers: {
  onToken?: (t: string) => void;
  onMeta?: (m: any) => void;
  onDone?: () => void;
  onError?: (e: any) => void;
}) {
  const url = `${API_BASE}/stream?q=${encodeURIComponent(question)}`;
  const ev = new EventSource(url);
  
  ev.addEventListener("token", (e) => {
    const data = JSON.parse((e as MessageEvent).data);
    handlers.onToken?.(data.token);
  });
  
  ev.addEventListener("meta", (e) => {
    const data = JSON.parse((e as MessageEvent).data);
    handlers.onMeta?.(data);
  });
  
  ev.addEventListener("done", () => {
    handlers.onDone?.();
    ev.close();
  });
  
  ev.onerror = (err) => {
    handlers.onError?.(err);
    ev.close();
  };
  
  return () => ev.close(); // return unsubscribe
}