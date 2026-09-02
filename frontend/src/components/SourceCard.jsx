export default function SourceCard({ source }) {
  return (
    <div className="rounded border border-slate-700 bg-slate-800/50 p-3 text-xs space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-cyan-400 font-medium truncate">
          {source.original_name}
        </span>
        <span className="text-slate-500 flex-shrink-0">
          第 {source.page_number} 頁
        </span>
      </div>
      <p className="text-slate-400 leading-relaxed line-clamp-3">
        {source.snippet}
      </p>
    </div>
  );
}
