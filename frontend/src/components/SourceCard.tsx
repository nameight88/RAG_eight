import type { SourceDoc } from '../types';

export function SourceCard({ source }: { source: SourceDoc }) {
  return (
    <div className="bg-[var(--bg-elevated)] border-l-4 border-[var(--accent)] p-3 rounded-lg text-sm mb-2">
      <div className="font-semibold text-[var(--text-primary)]">{source.institution || '알 수 없음'}</div>
      <div className="text-[var(--text-secondary)] text-xs mb-1">
        {source.date} · {source.doc_type} · 유사도 {(source.score * 100).toFixed(0)}%
      </div>
      <div className="text-[var(--text-primary)] line-clamp-2">{source.content}</div>
    </div>
  );
}
