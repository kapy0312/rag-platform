import { useState, useRef, useEffect, useCallback } from "react";
import {
  getDocuments,
  uploadDocument,
  deleteDocument,
  getDocumentStatus,
} from "../api";

const STATUS_STYLE = {
  pending: { label: "待處理", color: "#94a3b8" },
  processing: { label: "處理中", color: "#f59e0b" },
  done: { label: "完成", color: "#10b981" },
  error: { label: "錯誤", color: "#ef4444" },
};

export default function DocumentPanel({ category, onDocumentsChange }) {
  const [docs, setDocs] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef();
  const pollingRef = useRef({});

  const loadDocs = useCallback(async () => {
    if (!category) return;
    const data = await getDocuments(category.id);
    setDocs(data);
    onDocumentsChange?.(data);
    data
      .filter((d) => d.status === "pending" || d.status === "processing")
      .forEach(startPolling);
  }, [category]);

  useEffect(() => {
    setDocs([]);
    Object.values(pollingRef.current).forEach(clearInterval);
    pollingRef.current = {};
    loadDocs();
  }, [category]);

  function startPolling(doc) {
    if (pollingRef.current[doc.id]) return;
    pollingRef.current[doc.id] = setInterval(async () => {
      try {
        const s = await getDocumentStatus(doc.id);
        if (s.status === "done" || s.status === "error") {
          clearInterval(pollingRef.current[doc.id]);
          delete pollingRef.current[doc.id];
          loadDocs();
        }
      } catch {
        clearInterval(pollingRef.current[doc.id]);
        delete pollingRef.current[doc.id];
      }
    }, 2000);
  }

  async function handleUpload(file) {
    if (!file || !file.name.endsWith(".pdf")) return alert("只接受 PDF");
    setUploading(true);
    try {
      const doc = await uploadDocument(file, category.id);
      setDocs((v) => [doc, ...v]);
      startPolling(doc);
    } catch (err) {
      alert(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id) {
    if (!confirm("確定刪除此文件？")) return;
    try {
      await deleteDocument(id);
      clearInterval(pollingRef.current[id]);
      delete pollingRef.current[id];
      setDocs((v) => v.filter((d) => d.id !== id));
    } catch (err) {
      alert(err.message);
    }
  }

  if (!category)
    return (
      <div
        className="flex items-center justify-center h-full text-slate-600 text-sm"
        style={{ width: 300 }}
      >
        請先選擇類別
      </div>
    );

  return (
    <div
      className="flex flex-col h-full"
      style={{
        width: 300,
        background: "#0d1526",
        borderRight: "1px solid #1e2a3a",
      }}
    >
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center gap-2 mb-3">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: category.color }}
          />
          <h2 className="text-sm font-semibold truncate">{category.name}</h2>
        </div>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleUpload(e.dataTransfer.files[0]);
          }}
          onClick={() => fileRef.current.click()}
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            dragging
              ? "border-cyan-400 bg-cyan-500/10"
              : "border-slate-700 hover:border-slate-500"
          }`}
        >
          <p className="text-xs text-slate-400">
            {uploading ? "上傳中..." : "拖曳或點擊上傳 PDF"}
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => handleUpload(e.target.files[0])}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-2 space-y-1 px-2">
        {docs.length === 0 && (
          <p className="text-center text-slate-600 text-xs mt-8">尚無文件</p>
        )}
        {docs.map((doc) => {
          const s = STATUS_STYLE[doc.status] || STATUS_STYLE.pending;
          return (
            <div
              key={doc.id}
              className="flex items-start gap-2 p-2 rounded bg-slate-800/40 group"
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs truncate text-slate-200">
                  {doc.original_name}
                </p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px]" style={{ color: s.color }}>
                    {s.label}
                  </span>
                  {doc.status === "done" && (
                    <span className="text-[10px] text-slate-500">
                      {doc.chunk_count} chunks
                    </span>
                  )}
                  {doc.status === "error" && (
                    <span
                      className="text-[10px] text-red-400 truncate"
                      title={doc.error_msg}
                    >
                      ⚠ {doc.error_msg}
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => handleDelete(doc.id)}
                className="opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 text-xs flex-shrink-0 transition-opacity"
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
