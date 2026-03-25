import type { SourceDoc } from '../types';

export function SourceCard({ source }: { source: SourceDoc }) {
  return (
    <div className="bg-gray-50 border-l-4 border-blue-400 p-3 rounded text-sm mb-2">
      <div className="font-semibold text-blue-800">{source.institution || '알 수 없음'}</div>
      <div className="text-gray-500 text-xs mb-1">
        {source.date} · {source.doc_type} · 유사도 {(source.score * 100).toFixed(0)}%
      </div>
      <div className="text-gray-700 line-clamp-2">{source.content}</div>
    </div>
  );
}
