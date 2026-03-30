import type { Conversation, FilterParams } from '../types';
import { FilterBar } from './FilterBar';
import { ThemeToggle } from './ThemeToggle';

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  filters: FilterParams;
  onFilterChange: (f: FilterParams) => void;
}

export function Sidebar({ conversations, activeId, onSelect, onNew, filters, onFilterChange }: Props) {
  return (
    <aside className="w-64 flex flex-col h-full flex-shrink-0 bg-[var(--bg-surface)] border-r border-[var(--border)]">
      {/* 헤더: 브랜드 + 테마 토글 */}
      <div className="px-4 py-4 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-[var(--accent)]">
            <path d="M11.584 2.376a.75.75 0 01.832 0l9 6a.75.75 0 11-.832 1.248L12 3.901 3.416 9.624a.75.75 0 01-.832-1.248l9-6z" />
            <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 010 1.5H3a.75.75 0 010-1.5h.75v-9.918a.75.75 0 01.634-.74A49.109 49.109 0 0112 9c2.59 0 5.134.202 7.616.592a.75.75 0 01.634.74zm-7.5 2.418a.75.75 0 00-1.5 0v6.75a.75.75 0 001.5 0v-6.75zm3-.75a.75.75 0 01.75.75v6.75a.75.75 0 01-1.5 0v-6.75a.75.75 0 01.75-.75zM9 12.75a.75.75 0 00-1.5 0v6.75a.75.75 0 001.5 0v-6.75z" clipRule="evenodd" />
          </svg>
          <span className="text-sm font-semibold text-[var(--text-primary)]">FSS 챗봇</span>
        </div>
        <ThemeToggle />
      </div>

      {/* 새 대화 버튼 */}
      <div className="px-3 pb-3 flex-shrink-0">
        <button
          onClick={onNew}
          className="w-full py-2 px-3 bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
          </svg>
          새 대화
        </button>
      </div>

      {/* 대화 목록 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2">
        {conversations.length === 0 && (
          <p className="text-[var(--text-secondary)] text-xs text-center mt-4">대화 이력이 없습니다</p>
        )}
        {conversations.map(conv => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`w-full text-left px-3 py-2 rounded-lg mb-0.5 text-sm truncate transition-colors text-[var(--text-primary)] ${
              activeId === conv.id
                ? 'bg-[var(--bg-elevated)] border-l-[3px] border-[var(--accent)]'
                : 'hover:bg-[var(--bg-elevated)] border-l-[3px] border-transparent'
            }`}
            title={conv.title}
          >
            {conv.title}
          </button>
        ))}
      </div>

      {/* 필터 (하단 고정) */}
      <div className="flex-shrink-0">
        <FilterBar filters={filters} onChange={onFilterChange} />
      </div>
    </aside>
  );
}
