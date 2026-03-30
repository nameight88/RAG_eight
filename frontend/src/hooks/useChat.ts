import { useState, useCallback } from 'react';
import { api } from '../api/client';
import type { Message, Conversation, FilterParams, ChatResponse } from '../types';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState<FilterParams>({});

  const loadConversations = useCallback(async () => {
    try {
      const data = await api.listConversations();
      setConversations(data);
    } catch {
      // Backend may be sleeping on Render free tier - ignore
    }
  }, []);

  const selectConversation = useCallback(async (id: string) => {
    setActiveConversationId(id);
    try {
      const msgs = await api.getMessages(id);
      setMessages(msgs);
    } catch {
      setMessages([]);
    }
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    const tempId = `temp-${Date.now()}`;
    const tempUserMsg: Message = {
      id: tempId,
      role: 'user',
      content: text,
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);
    setIsLoading(true);

    try {
      const response: ChatResponse = await api.chat({
        conversation_id: activeConversationId,
        message: text,
        filters,
      });

      if (!activeConversationId) {
        setActiveConversationId(response.conversation_id);
        await loadConversations();
      }

      const botMsg: Message = {
        id: response.message_id,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [
        ...prev.filter(m => m.id !== tempId),
        { ...tempUserMsg, id: `user-${response.message_id}` },
        botMsg,
      ]);
    } catch (err) {
      const errMsg: Message = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: '⚠️ 답변을 가져오는 데 실패했습니다. 잠시 후 다시 시도해주세요.',
        sources: [],
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [
        ...prev.filter(m => m.id !== tempId),
        { ...tempUserMsg, id: `user-err-${Date.now()}` },
        errMsg,
      ]);
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [activeConversationId, filters, isLoading, loadConversations]);

  const newConversation = useCallback(() => {
    setActiveConversationId(null);
    setMessages([]);
  }, []);

  return {
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
  };
}
