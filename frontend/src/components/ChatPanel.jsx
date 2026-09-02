import { useState, useRef, useEffect } from "react";
import { useSSE } from "../hooks/useSSE";
import SourceCard from "./SourceCard";

export default function ChatPanel({ categories, selectedCategory }) {
  const [queryCategory, setQueryCategory] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const { query, streaming, abort } = useSSE();
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");

    const userMsg = { role: "user", text: q };
    const assistantMsg = { role: "assistant", text: "", sources: [] };
    setMessages((v) => [...v, userMsg, assistantMsg]);
    const idx = messages.length + 1;

    await query({
      question: q,
      categoryId: queryCategory,
      onSources: (sources) => {
        setMessages((v) =>
          v.map((m, i) => (i === idx ? { ...m, sources } : m)),
        );
      },
      onToken: (token) => {
        setMessages((v) =>
          v.map((m, i) => (i === idx ? { ...m, text: m.text + token } : m)),
        );
      },
      onDone: () => {},
    });
  }

  return (
    <div
      className="flex flex-col flex-1 h-full"
      style={{ background: "#0a0e1a" }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-800">
        <span className="text-xs text-slate-400">查詢範圍：</span>
        <select
          value={queryCategory ?? ""}
          onChange={(e) =>
            setQueryCategory(
              e.target.value === "" ? null : Number(e.target.value),
            )
          }
          className="text-sm bg-slate-800 border border-slate-700 rounded px-2 py-1 outline-none focus:border-cyan-500 text-slate-200"
        >
          <option value="">全部類別</option>
          {categories.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-600 space-y-2">
            <p className="text-2xl">⚡</p>
            <p className="text-sm">上傳 PDF 後輸入問題開始查詢</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] space-y-3 ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col`}
            >
              <div
                className={`rounded-xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-cyan-500/20 text-cyan-100 border border-cyan-500/30"
                    : "bg-slate-800/60 text-slate-200"
                }`}
              >
                {msg.text ||
                  (msg.role === "assistant" &&
                  streaming &&
                  i === messages.length - 1 ? (
                    <span className="animate-pulse text-slate-500">
                      思考中...
                    </span>
                  ) : null)}
              </div>
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <div className="w-full space-y-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">
                    參考來源
                  </p>
                  {msg.sources.map((s, j) => (
                    <SourceCard key={j} source={s} />
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-800">
        <div className="flex gap-2">
          <input
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-sm outline-none focus:border-cyan-500 text-slate-200 placeholder-slate-500"
            placeholder="輸入問題..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            disabled={streaming}
          />
          <button
            onClick={streaming ? abort : handleSend}
            className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
              streaming
                ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
                : "bg-cyan-500 text-slate-900 hover:bg-cyan-400"
            }`}
          >
            {streaming ? "停止" : "送出"}
          </button>
        </div>
      </div>
    </div>
  );
}
