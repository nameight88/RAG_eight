import type { FilterParams } from '../types';

interface Props {
  filters: FilterParams;
  onChange: (f: FilterParams) => void;
}

const EMPTY_FILTERS: FilterParams = {
  doc_type: null,
  date_from: null,
  date_to: null,
  institution: null,
};

const inputClass = "text-sm border border-[var(--border)] rounded-lg px-2.5 py-1.5 bg-[var(--bg-elevated)] text-[var(--text-primary)] w-full focus:outline-none focus:ring-1 focus:ring-[var(--accent)] placeholder-[var(--text-secondary)]";
const labelClass = "text-xs text-[var(--text-secondary)] mb-1 block";

export function FilterBar({ filters, onChange }: Props) {
  return (
    <div className="p-3 border-t border-[var(--border)] flex flex-col gap-2.5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wide">필터</p>
        <button
          onClick={() => onChange(EMPTY_FILTERS)}
          className="text-xs bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded-full px-2.5 py-0.5 hover:text-[var(--text-primary)] transition-colors"
        >
          초기화
        </button>
      </div>

      <select
        value={filters.doc_type ?? ''}
        onChange={e => onChange({ ...filters, doc_type: e.target.value || null })}
        className={inputClass}
      >
        <option value="">전체 문서</option>
        <option value="sanctions">제재결과</option>
        <option value="management">경영유의사항</option>
      </select>

      <div className="flex flex-col gap-1.5">
        <div>
          <label className={labelClass}>시작일</label>
          <input
            type="date"
            value={filters.date_from ?? ''}
            onChange={e => onChange({ ...filters, date_from: e.target.value || null })}
            className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>종료일</label>
          <input
            type="date"
            value={filters.date_to ?? ''}
            onChange={e => onChange({ ...filters, date_to: e.target.value || null })}
            className={inputClass}
          />
        </div>
      </div>

      <input
        type="text"
        value={filters.institution ?? ''}
        onChange={e => onChange({ ...filters, institution: e.target.value || null })}
        placeholder="기관명"
        className={inputClass}
      />
    </div>
  );
}
