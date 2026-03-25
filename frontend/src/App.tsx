import { useEffect } from 'react';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar';
import { FilterBar } from './components/FilterBar';
import { ChatWindow } from './components/ChatWindow';

export default function App() {
  const {
    conversations,
    messages,
    isLoading,
    filters,
    activeConversationId,
    loadConversations,
    selectConversation,
    sendMessage,
    newConversation,
    setFilters,
  } = useChat();

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar
        conversations={conversations}
        activeId={activeConversationId}
        onSelect={selectConversation}
        onNew={newConversation}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="px-6 py-3 bg-blue-800 text-white font-semibold text-lg flex-shrink-0">
          금융감독원 제재정보 챗봇
        </header>
        <FilterBar filters={filters} onChange={setFilters} />
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onSend={sendMessage}
        />
      </div>
    </div>
  );
}
