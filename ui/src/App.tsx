import { useEffect, useRef, useState } from "react";
import { ask, stream } from "./lib/api";
import "./App.css";

export default function App() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState<"idle" | "ask" | "stream">("idle");
  const stopRef = useRef<(() => void) | null>(null);

  // Keep the answer panel scrolled to bottom while tokens stream in.
  const answerRef = useRef<HTMLPreElement | null>(null);
  useEffect(() => {
    if (loading === "stream" && answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer, loading]);

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

    // NOTE: stream() is your existing client that talks to POST /stream
    stopRef.current = stream(q, {
      onToken: (t) => setAnswer((prev) => prev + t),
      onMeta: (m) => {
        // If the server emits a one-off "meta" frame with citations
        if (m?.citations) {
          const sourceNames = m.citations.map((c: any) => c.title || c.url);
          setSources(sourceNames);
        }
      },
      onDone: () => setLoading("idle"),
      onError: () => {
        setAnswer((prev) => prev + "\n[stream error]");
        setLoading("idle");
      },
    });
  }

  function onStop() {
    stopRef.current?.();
    setLoading("idle");
  }

  function onCopy() {
    if (!answer) return;
    navigator.clipboard?.writeText(answer);
  }

  return (
    <div id="root">
      <div className="app-card">
        <header>
          <h1 className="app-title">RAG Job Doc Assistant</h1>
          <p className="app-subtitle">
            Ask questions about your documents. Choose <b>Ask</b> or <b>Stream</b>.
          </p>
        </header>

        <textarea
          className="app-textarea"
          rows={5}
          placeholder="Ask something…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />

        <div className="actions-row">
          <button
            onClick={onAsk}
            disabled={!q || loading !== "idle"}
            className="btn btn--primary"
          >
            {loading === "ask" ? <span className="spinner" /> : "Ask"}
          </button>

          {loading !== "stream" ? (
            <button
              onClick={onStream}
              disabled={!q}
              className="btn btn--secondary"
            >
              Stream
            </button>
          ) : (
            <button onClick={onStop} className="btn btn--stop">
              Stop
            </button>
          )}

          {latency && <span className="chip">Response: {latency.toFixed(0)}ms</span>}
        </div>

        <div className="section-header">
          <h3 className="section-title">Answer</h3>
          <button
            onClick={onCopy}
            disabled={!answer}
            className={`icon-btn ${!answer ? "icon-btn--disabled" : ""}`}
          >
            ⧉ Copy
          </button>
        </div>
        <pre className="answer-box">{answer || "—"}</pre>

        <h4 className="section-title" style={{ marginTop: "12px" }}>Sources</h4>
        {sources.length === 0 ? (
          <p className="muted">No sources yet.</p>
        ) : (
          <div className="badge-wrap">
            {sources.map((s, i) => (
              <span key={i} className="badge">{s}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}