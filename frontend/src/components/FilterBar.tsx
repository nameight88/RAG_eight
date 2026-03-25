import type { FilterParams } from '../types';

interface Props {
  filters: FilterParams;
  onChange: (f: FilterParams) => void;
}

export function FilterBar({ filters, onChange }: Props) {
  return (
    <div className="flex gap-3 p-3 bg-gray-50 border-b items-center flex-wrap">
      <select
        value={filters.doc_type ?? ''}
        onChange={e => onChange({ ...filters, doc_type: e.target.value || null })}
        className="text-sm border rounded px-2 py-1 bg-white"
      >
        <option value="">전체 문서</option>
        <option value="sanctions">제재결과</option>
        <option value="management">경영유의사항</option>
      </select>
      <input
        type="date"
        value={filters.date_from ?? ''}
        onChange={e => onChange({ ...filters, date_from: e.target.value || null })}
        className="text-sm border rounded px-2 py-1"
      />
      <span className="text-gray-400 text-sm">~</span>
      <input
        type="date"
        value={filters.date_to ?? ''}
        onChange={e => onChange({ ...filters, date_to: e.target.value || null })}
        className="text-sm border rounded px-2 py-1"
      />
      <input
        type="text"
        value={filters.institution ?? ''}
        onChange={e => onChange({ ...filters, institution: e.target.value || null })}
        placeholder="기관명"
        className="text-sm border rounded px-2 py-1 w-32"
      />
    </div>
  );
}
