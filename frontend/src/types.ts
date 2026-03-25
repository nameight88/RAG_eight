export interface FilterParams {
  doc_type?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  institution?: string | null;
}

export interface SourceDoc {
  institution: string;
  date: string;
  doc_type: string;
  content: string;
  score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: SourceDoc[];
  filters?: FilterParams | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatRequest {
  conversation_id?: string | null;
  message: string;
  filters: FilterParams;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  sources: SourceDoc[];
}
