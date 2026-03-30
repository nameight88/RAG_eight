import { useEffect } from 'react';
import { useChat } from './hooks/useChat';
import { Sidebar } from './components/Sidebar';
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
    <div className="flex h-screen bg-[var(--bg-base)]">
      <Sidebar
        conversations={conversations}
        activeId={activeConversationId}
        onSelect={selectConversation}
        onNew={newConversation}
        filters={filters}
        onFilterChange={setFilters}
      />
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSend={sendMessage}
      />
    </div>
  );
}
