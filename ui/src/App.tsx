import { useRef, useState } from "react";
import { ask, stream } from "./lib/api";

export default function App() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState<"idle" | "ask" | "stream">("idle");
  const stopRef = useRef<(() => void) | null>(null);

  async function onAsk() {
    setLoading("ask");
    setAnswer("");
    setSources([]);
    setLatency(null);
    
    try {
      const res = await ask(q);
      setAnswer(res.answer);
      setSources(res.sources || []);
      setLatency(res.latency_ms || null);
    } catch (e: any) {
      setAnswer("Error: " + (e.message || String(e)));
    } finally {
      setLoading("idle");
    }
  }

  function onStream() {
    setLoading("stream");
    setAnswer("");
    setSources([]);
    setLatency(null);
    
    stopRef.current = stream(q, {
      onToken: (t) => setAnswer((prev) => prev + t),
      onMeta: (m) => {
        // Handle citations from streaming (they come as citation objects)
        if (m.citations) {
          const sourceNames = m.citations.map((c: any) => c.title || c.url);
          setSources(sourceNames);
        }
        // Handle other meta information
        if (m.status) {
          console.log("Stream status:", m.status);
        }
      },
      onDone: () => setLoading("idle"),
      onError: () => {
        setAnswer((prev) => prev + "\n[stream error]");
        setLoading("idle");
      }
    });
  }

  function onStop() {
    stopRef.current?.();
    setLoading("idle");
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", fontFamily: "system-ui" }}>
      <h1>RAG Job Doc Assistant</h1>
      
      <textarea
        rows={4}
        style={{ width: "100%", padding: 12 }}
        placeholder="Ask something about your documents…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button onClick={onAsk} disabled={!q || loading !== "idle"}>
          Ask
        </button>
        {loading !== "stream" ? (
          <button onClick={onStream} disabled={!q}>
            Stream (SSE)
          </button>
        ) : (
          <button onClick={onStop}>Stop</button>
        )}
      </div>

      <h3 style={{ marginTop: 20 }}>Answer</h3>
      <pre style={{ 
        whiteSpace: "pre-wrap", 
        background: "#f6f6f6", 
        padding: 12, 
        borderRadius: 8 
      }}>
        {answer || "—"}
      </pre>

      {latency && (
        <p style={{ color: "#666", fontSize: "0.9em", marginTop: 8 }}>
          Response time: {latency.toFixed(0)}ms
        </p>
      )}

      {sources.length > 0 && (
        <>
          <h4>Sources</h4>
          <ul>
            {sources.map((source, i) => (
              <li key={i}>
                <span style={{ fontFamily: "monospace", background: "#f0f0f0", padding: "2px 6px", borderRadius: 3 }}>
                  {source}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}