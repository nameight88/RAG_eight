# Phase 2B: React 프론트엔드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI 백엔드와 연동하는 챗봇 웹 UI를 React로 구현한다. 사이드바에 대화 목록, 우측에 채팅 인터페이스를 배치하고 기관/날짜 필터를 제공한다.

**Architecture:** Vite + React + TypeScript 단일 페이지 앱. TanStack Query로 서버 상태 관리. shadcn/ui + Tailwind로 UI 구성. API 호출 함수는 `src/api/` 에 격리한다.

**Tech Stack:** Node.js 20+, Vite 5, React 18, TypeScript 5, Tailwind CSS 3, shadcn/ui, TanStack Query v5, Axios

**전제 조건:** Phase 2A 백엔드 완료 (FastAPI 서버 로컬에서 기동 가능한 상태)

---

## 파일 맵

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── .env.local.example              # VITE_API_URL=http://localhost:8000
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── api/
    │   └── client.ts               # FastAPI 호출 함수 + 타입 정의
    ├── components/
    │   ├── ConversationList.tsx    # 사이드바 대화 목록
    │   ├── ChatWindow.tsx          # 메시지 표시 영역
    │   ├── ChatInput.tsx           # 입력창 + 필터 드롭다운
    │   └── SourceCard.tsx          # 참고문서 카드
    └── pages/
        └── ChatPage.tsx            # 메인 레이아웃 (사이드바 + 채팅)
```

---

## Task 1: React 프로젝트 초기화

- [ ] **Step 1: Vite + React + TypeScript 프로젝트 생성**

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: 의존성 설치**

```bash
npm install axios @tanstack/react-query lucide-react
npm install -D tailwindcss postcss autoprefixer tailwindcss-animate
npx tailwindcss init -p
```

- [ ] **Step 3: shadcn/ui 초기화**

```bash
npx shadcn@latest init
# 프롬프트: TypeScript=yes, style=Default, baseColor=Slate, CSS variables=yes
npx shadcn@latest add button input textarea card badge scroll-area separator
```

- [ ] **Step 4: tailwind.config.ts 설정**

`frontend/tailwind.config.ts`:
```ts
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [require("tailwindcss-animate")],
}
```

- [ ] **Step 5: .env.local.example 작성**

```
VITE_API_URL=http://localhost:8000
```

실제 사용:
```bash
cp .env.local.example .env.local
```

- [ ] **Step 6: 개발 서버 기동 확인**

```bash
npm run dev
# http://localhost:5173 → Vite 기본 페이지 확인
```

- [ ] **Step 7: 커밋**

```bash
cd ..
git add frontend/
git commit -m "feat: initialize React frontend with Vite + shadcn/ui + TanStack Query"
```

---

## Task 2: API 클라이언트 타입 정의

**Files:**
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: client.ts 작성**

`frontend/src/api/client.ts`:
```typescript
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

// ---------- 타입 ----------

export interface ChatFilters {
  doc_type?: string;      // "제재결과" | "경영유의사항"
  date_from?: string;     // "YYYY-MM-DD"
  date_to?: string;
  institution?: string;
}

export interface SourceDocument {
  institution: string;
  date: string;
  doc_type: string;
  excerpt: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceDocument[];
  conversation_id: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDocument[];
  created_at: string;
}

// ---------- API 함수 ----------

export async function sendChat(
  query: string,
  filters: ChatFilters,
  conversation_id?: string
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/api/chat", {
    query,
    filters,
    conversation_id,
  });
  return data;
}

export async function fetchConversations(): Promise<Conversation[]> {
  const { data } = await api.get<Conversation[]>("/api/conversations");
  return data;
}

export async function fetchMessages(conversationId: string): Promise<Message[]> {
  const { data } = await api.get<Message[]>(
    `/api/conversations/${conversationId}/messages`
  );
  return data;
}
```

- [ ] **Step 2: TypeScript 컴파일 오류 없음 확인**

```bash
cd frontend
npx tsc --noEmit
```

Expected: 오류 없음

- [ ] **Step 3: 커밋**

```bash
cd ..
git add frontend/src/api/client.ts
git commit -m "feat: add typed API client functions"
```

---

## Task 3: SourceCard 컴포넌트

**Files:**
- Create: `frontend/src/components/SourceCard.tsx`

- [ ] **Step 1: SourceCard 작성**

`frontend/src/components/SourceCard.tsx`:
```tsx
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { SourceDocument } from "@/api/client";

interface Props {
  source: SourceDocument;
}

export function SourceCard({ source }: Props) {
  return (
    <Card className="text-sm border border-slate-200">
      <CardContent className="p-3 space-y-1">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {source.doc_type}
          </Badge>
          <span className="font-medium text-slate-700">{source.institution}</span>
          <span className="text-slate-400 ml-auto text-xs">{source.date}</span>
        </div>
        <p className="text-slate-600 line-clamp-2">{source.excerpt}</p>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: 커밋**

```bash
git add frontend/src/components/SourceCard.tsx
git commit -m "feat: add SourceCard component"
```

---

## Task 4: ConversationList 컴포넌트 (사이드바)

**Files:**
- Create: `frontend/src/components/ConversationList.tsx`

- [ ] **Step 1: ConversationList 작성**

`frontend/src/components/ConversationList.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { fetchConversations, type Conversation } from "@/api/client";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { MessageSquare, Plus } from "lucide-react";

interface Props {
  selectedId?: string;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function ConversationList({ selectedId, onSelect, onNew }: Props) {
  const { data: conversations = [] } = useQuery<Conversation[]>({
    queryKey: ["conversations"],
    queryFn: fetchConversations,
    refetchInterval: 10_000,
  });

  return (
    <div className="flex flex-col h-full border-r border-slate-200 bg-slate-50 w-64">
      <div className="p-3">
        <Button onClick={onNew} className="w-full" variant="outline" size="sm">
          <Plus className="w-4 h-4 mr-2" />
          새 대화
        </Button>
      </div>
      <Separator />
      <ScrollArea className="flex-1 p-2">
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`w-full text-left px-3 py-2 rounded-md text-sm mb-1 flex items-center gap-2 transition-colors ${
              selectedId === conv.id
                ? "bg-slate-200 text-slate-900"
                : "hover:bg-slate-100 text-slate-600"
            }`}
          >
            <MessageSquare className="w-4 h-4 shrink-0 text-slate-400" />
            <span className="truncate">{conv.title}</span>
          </button>
        ))}
      </ScrollArea>
    </div>
  );
}
```

- [ ] **Step 2: 커밋**

```bash
git add frontend/src/components/ConversationList.tsx
git commit -m "feat: add ConversationList sidebar component"
```

---

## Task 5: ChatInput 컴포넌트 (입력창 + 필터)

**Files:**
- Create: `frontend/src/components/ChatInput.tsx`

- [ ] **Step 1: ChatInput 작성**

`frontend/src/components/ChatInput.tsx`:
```tsx
import { useState, type KeyboardEvent } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import type { ChatFilters } from "@/api/client";

interface Props {
  onSend: (query: string, filters: ChatFilters) => void;
  isLoading: boolean;
}

const DOC_TYPES = ["전체", "제재결과", "경영유의사항"];

export function ChatInput({ onSend, isLoading }: Props) {
  const [query, setQuery] = useState("");
  const [docType, setDocType] = useState("전체");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const handleSend = () => {
    if (!query.trim() || isLoading) return;
    const filters: ChatFilters = {
      doc_type: docType === "전체" ? undefined : docType,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
    };
    onSend(query.trim(), filters);
    setQuery("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-slate-200 p-4 space-y-2 bg-white">
      {/* 필터 행 */}
      <div className="flex gap-2 text-sm">
        <select
          value={docType}
          onChange={(e) => setDocType(e.target.value)}
          className="border border-slate-200 rounded-md px-2 py-1 text-slate-600 text-xs"
        >
          {DOC_TYPES.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <input
          type="date"
          value={dateFrom}
          onChange={(e) => setDateFrom(e.target.value)}
          className="border border-slate-200 rounded-md px-2 py-1 text-xs text-slate-600"
          placeholder="시작일"
        />
        <span className="text-slate-400 self-center">~</span>
        <input
          type="date"
          value={dateTo}
          onChange={(e) => setDateTo(e.target.value)}
          className="border border-slate-200 rounded-md px-2 py-1 text-xs text-slate-600"
          placeholder="종료일"
        />
      </div>
      {/* 입력 행 */}
      <div className="flex gap-2">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="금융 제재 또는 경영유의사항에 대해 질문하세요... (Enter로 전송)"
          className="resize-none min-h-[60px] text-sm"
          rows={2}
        />
        <Button
          onClick={handleSend}
          disabled={!query.trim() || isLoading}
          size="icon"
          className="h-auto"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 커밋**

```bash
git add frontend/src/components/ChatInput.tsx
git commit -m "feat: add ChatInput with filter dropdowns"
```

---

## Task 6: ChatWindow 컴포넌트 (메시지 영역)

**Files:**
- Create: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: ChatWindow 작성**

`frontend/src/components/ChatWindow.tsx`:
```tsx
import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SourceCard } from "@/components/SourceCard";
import type { Message } from "@/api/client";

interface Props {
  messages: Message[];
  isLoading: boolean;
}

export function ChatWindow({ messages, isLoading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="max-w-3xl mx-auto space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-400 mt-20 text-sm">
            금융감독원 제재결과 및 경영유의사항에 대해 질문해보세요.
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] space-y-2 ${
                msg.role === "user" ? "items-end" : "items-start"
              } flex flex-col`}
            >
              <div
                className={`rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-slate-800 text-white"
                    : "bg-white border border-slate-200 text-slate-800"
                }`}
              >
                {msg.content}
              </div>
              {/* 참고문서 카드 */}
              {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                <div className="w-full space-y-1">
                  <p className="text-xs text-slate-400 px-1">참고 문서</p>
                  {msg.sources.map((src, i) => (
                    <SourceCard key={i} source={src} />
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-200 rounded-2xl px-4 py-2 text-sm text-slate-400 animate-pulse">
              답변 생성 중...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
```

- [ ] **Step 2: 커밋**

```bash
git add frontend/src/components/ChatWindow.tsx
git commit -m "feat: add ChatWindow component with source cards"
```

---

## Task 7: ChatPage 메인 페이지 조립

**Files:**
- Create: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: ChatPage 작성**

`frontend/src/pages/ChatPage.tsx`:
```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ConversationList } from "@/components/ConversationList";
import { ChatWindow } from "@/components/ChatWindow";
import { ChatInput } from "@/components/ChatInput";
import {
  sendChat,
  fetchMessages,
  type ChatFilters,
  type Message,
} from "@/api/client";

export function ChatPage() {
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [localMessages, setLocalMessages] = useState<Message[]>([]);
  const queryClient = useQueryClient();

  // 선택된 대화의 메시지 로드
  const { data: savedMessages = [] } = useQuery<Message[]>({
    queryKey: ["messages", conversationId],
    queryFn: () => fetchMessages(conversationId!),
    enabled: !!conversationId,
  });

  // MVP 허용 동작: 기존 대화를 선택한 후 새 메시지를 보내면
  // localMessages에만 누적되어 이전 이력이 보이지 않음.
  // 포트폴리오 MVP 범위 내 허용. 개선 시 localMessages 초기값에 savedMessages 포함.
  const messages = localMessages.length > 0 ? localMessages : savedMessages;

  const { mutate: chat, isPending } = useMutation({
    mutationFn: ({
      query,
      filters,
    }: {
      query: string;
      filters: ChatFilters;
    }) => sendChat(query, filters, conversationId),

    onMutate: ({ query }) => {
      // 낙관적 업데이트: 사용자 메시지 즉시 표시
      const userMsg: Message = {
        id: `tmp-${Date.now()}`,
        role: "user",
        content: query,
        created_at: new Date().toISOString(),
      };
      setLocalMessages((prev) => [...prev, userMsg]);
    },

    onSuccess: (data) => {
      setConversationId(data.conversation_id);
      const assistantMsg: Message = {
        id: `tmp-${Date.now()}`,
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        created_at: new Date().toISOString(),
      };
      setLocalMessages((prev) => [...prev, assistantMsg]);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });

  const handleSend = (query: string, filters: ChatFilters) => {
    chat({ query, filters });
  };

  const handleSelectConversation = (id: string) => {
    setConversationId(id);
    setLocalMessages([]);
  };

  const handleNewConversation = () => {
    setConversationId(undefined);
    setLocalMessages([]);
  };

  return (
    <div className="flex h-screen bg-slate-50">
      <ConversationList
        selectedId={conversationId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <header className="px-6 py-3 border-b border-slate-200 bg-white">
          <h1 className="text-base font-semibold text-slate-800">
            금융감독원 제재·경영유의사항 검색
          </h1>
        </header>
        <ChatWindow messages={messages} isLoading={isPending} />
        <ChatInput onSend={handleSend} isLoading={isPending} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: App.tsx 수정**

`frontend/src/App.tsx`:
```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatPage } from "@/pages/ChatPage";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChatPage />
    </QueryClientProvider>
  );
}
```

- [ ] **Step 3: TypeScript 컴파일 확인**

```bash
cd frontend
npx tsc --noEmit
```

Expected: 오류 없음

- [ ] **Step 4: 백엔드 기동 후 통합 확인**

```bash
# 터미널 1: 백엔드
cd backend && uvicorn main:app --reload --port 8000

# 터미널 2: 프론트엔드
cd frontend && npm run dev
```

브라우저에서 `http://localhost:5173` 접속 → 챗봇 UI 표시 확인 → 질문 전송 → 답변 수신 확인

- [ ] **Step 5: 커밋**

```bash
cd ..
git add frontend/src/pages/ChatPage.tsx frontend/src/App.tsx
git commit -m "feat: assemble ChatPage - chatbot UI complete"
```

---

## Task 8: 빌드 확인 및 Vercel 배포 준비

- [ ] **Step 1: 프로덕션 빌드 확인**

```bash
cd frontend
npm run build
```

Expected: `dist/` 생성, 오류 없음

- [ ] **Step 2: vercel.json 작성 (SPA 라우팅)**

`frontend/vercel.json`:
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

- [ ] **Step 3: Render용 백엔드 배포 파일 작성**

`backend/render.yaml`:
```yaml
services:
  - type: web
    name: fss-rag-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: LLM_API_KEY
        sync: false
      - key: LLM_PROVIDER
        value: openai
      - key: LLM_MODEL
        value: gpt-4o-mini
      - key: ALLOWED_ORIGINS
        sync: false
```

- [ ] **Step 4: 최종 커밋**

```bash
git add frontend/vercel.json backend/render.yaml
git commit -m "chore: add Vercel and Render deployment config, phase2b complete"
```
