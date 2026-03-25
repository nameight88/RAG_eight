import type { Conversation } from '../types';

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function Sidebar({ conversations, activeId, onSelect, onNew }: Props) {
  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col h-full flex-shrink-0">
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={onNew}
          className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
        >
          + 새 대화
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 && (
          <p className="text-gray-500 text-xs text-center mt-4">대화 이력이 없습니다</p>
        )}
        {conversations.map(conv => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`w-full text-left px-3 py-2 rounded-lg mb-1 text-sm truncate transition-colors ${
              activeId === conv.id ? 'bg-gray-700' : 'hover:bg-gray-800'
            }`}
            title={conv.title}
          >
            {conv.title}
          </button>
        ))}
      </div>
    </aside>
  );
}
