import type { ChatRequest, ChatResponse, Conversation, Message } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export const api = {
  chat: (body: ChatRequest): Promise<ChatResponse> =>
    request('/api/chat', { method: 'POST', body: JSON.stringify(body) }),

  listConversations: (): Promise<Conversation[]> =>
    request('/api/conversations'),

  getMessages: (conversationId: string): Promise<Message[]> =>
    request(`/api/conversations/${conversationId}/messages`),
};
