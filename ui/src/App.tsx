import { useRef, useState } from "react";
import { ask, stream } from "./lib/api";
import type { Citation } from "./types";

export default function App() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [loading, setLoading] = useState<"idle"|"ask"|"stream">("idle");
const stopRef = useRef<(() => void) | null>(null);

  async function onAsk() {
    setLoading("ask");
    setAnswer("");
    setCitations([]);
    try {
      const res = await ask(q);
      setAnswer(res.answer);
      setCitations(res.citations || []);
    } catch (e:any) {
      setAnswer("Error: " + (e.message || String(e)));
    } finally {
      setLoading("idle");
    }
  }

  function onStream() {
    setLoading("stream");
    setAnswer("");
    setCitations([]);

    stopRef.current = stream(q, {
      onToken: (t) => setAnswer((prev) => prev + t),
      onMeta: (m) => {
        if (m.citations) setCitations(m.citations);
      },
      onDone: () => setLoading("idle"),
      onError: (e) => {
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
        <button onClick={onAsk} disabled={!q || loading !== "idle"}>Ask</button>
        {loading !== "stream" ? (
          <button onClick={onStream} disabled={!q}>Stream (SSE)</button>
        ) : (
          <button onClick={onStop}>Stop</button>
        )}
      </div>

      <h3 style={{ marginTop: 20 }}>Answer</h3>
      <pre style={{ whiteSpace: "pre-wrap", background: "#f6f6f6", padding: 12, borderRadius: 8 }}>
        {answer || "—"}
      </pre>

      {citations.length > 0 && (
        <>
          <h4>Citations</h4>
          <ul>
            {citations.map((c, i) => (
              <li key={i}><a href={c.url} target="_blank">{c.title || c.url}</a></li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
