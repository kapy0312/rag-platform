import { useState } from "react";
import { createCategory, deleteCategory } from "../api";

const PRESET_COLORS = [
  "#00D4FF",
  "#7C3AED",
  "#059669",
  "#DC2626",
  "#D97706",
  "#DB2777",
];

export default function CategoryPanel({
  categories,
  selected,
  onSelect,
  onRefresh,
}) {
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    name: "",
    description: "",
    color: "#00D4FF",
  });
  const [loading, setLoading] = useState(false);

  async function handleCreate(e) {
    e.preventDefault();
    if (!form.name.trim()) return;
    setLoading(true);
    try {
      await createCategory(form);
      setForm({ name: "", description: "", color: "#00D4FF" });
      setShowForm(false);
      onRefresh();
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(e, id) {
    e.stopPropagation();
    if (!confirm("刪除此類別將同時刪除所有文件與向量，確定？")) return;
    try {
      await deleteCategory(id);
      onRefresh();
    } catch (err) {
      alert(err.message);
    }
  }

  return (
    <div
      className="flex flex-col h-full"
      style={{
        width: 220,
        background: "#0f1320",
        borderRight: "1px solid #1e2a3a",
      }}
    >
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-xs font-semibold tracking-widest text-slate-400 uppercase mb-3">
          類別
        </h2>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="w-full text-sm py-1.5 rounded border border-cyan-500 text-cyan-400 hover:bg-cyan-500/10 transition-colors"
        >
          {showForm ? "取消" : "＋ 新增類別"}
        </button>
      </div>

      {showForm && (
        <form
          onSubmit={handleCreate}
          className="p-4 border-b border-slate-800 space-y-2"
        >
          <input
            className="w-full bg-slate-800 text-sm rounded px-2 py-1.5 outline-none border border-slate-700 focus:border-cyan-500"
            placeholder="類別名稱"
            value={form.name}
            onChange={(e) => setForm((v) => ({ ...v, name: e.target.value }))}
          />
          <input
            className="w-full bg-slate-800 text-sm rounded px-2 py-1.5 outline-none border border-slate-700 focus:border-cyan-500"
            placeholder="描述（選填）"
            value={form.description}
            onChange={(e) =>
              setForm((v) => ({ ...v, description: e.target.value }))
            }
          />
          <div className="flex gap-1.5 flex-wrap">
            {PRESET_COLORS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setForm((v) => ({ ...v, color: c }))}
                className="w-5 h-5 rounded-full transition-transform hover:scale-110"
                style={{
                  background: c,
                  outline: form.color === c ? `2px solid ${c}` : "none",
                  outlineOffset: 2,
                }}
              />
            ))}
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full text-sm py-1.5 rounded bg-cyan-500 text-slate-900 font-semibold hover:bg-cyan-400 disabled:opacity-50"
          >
            {loading ? "建立中..." : "建立"}
          </button>
        </form>
      )}

      <div className="flex-1 overflow-y-auto py-2">
        {categories.map((cat) => (
          <div
            key={cat.id}
            onClick={() => onSelect(selected?.id === cat.id ? null : cat)}
            className={`flex items-center gap-2 px-3 py-2 mx-2 rounded cursor-pointer group transition-colors ${
              selected?.id === cat.id
                ? "bg-slate-700/60"
                : "hover:bg-slate-800/60"
            }`}
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: cat.color }}
            />
            <span className="text-sm flex-1 truncate">{cat.name}</span>
            <button
              onClick={(e) => handleDelete(e, cat.id)}
              className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 text-xs transition-opacity"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
