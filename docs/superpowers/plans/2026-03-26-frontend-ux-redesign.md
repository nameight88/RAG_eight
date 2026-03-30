# Frontend UX Redesign (ChatGPT Style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ChatGPT 스타일 챗봇 UI로 개선 — 마크다운 렌더링, 예시 질문 카드, bouncing 로딩 애니메이션, 필터 사이드바 이동, 에러 메시지 표시.

**Architecture:** FilterBar를 사이드바 하단으로 이동하고 채팅 영역 헤더를 제거한다. react-markdown으로 AI 답변을 렌더링하고, useChat의 에러 핸들러에서 에러 메시지를 messages 배열에 push한다.

**Tech Stack:** React 18, TypeScript, Vite, TailwindCSS v4, react-markdown

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/App.tsx` | Modify | FilterBar 제거, 헤더 제거, Sidebar에 filter props 전달 |
| `frontend/src/components/Sidebar.tsx` | Modify | FilterBar 통합, 서비스명, Sidebar props 확장 |
| `frontend/src/components/FilterBar.tsx` | Modify | 세로 레이아웃, 필터 초기화 버튼 |
| `frontend/src/components/ChatWindow.tsx` | Modify | 예시 카드, bouncing dots 로딩, 전송 아이콘 |
| `frontend/src/components/MessageBubble.tsx` | Modify | react-markdown 렌더링 |
| `frontend/src/hooks/useChat.ts` | Modify | 에러 시 assistant 메시지 push |
| `frontend/package.json` | Modify | react-markdown 의존성 추가 |

---

## Task 1: react-markdown 설치

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: 패키지 설치**

```bash
cd frontend && npm install react-markdown
```

Expected: `package.json`의 dependencies에 `"react-markdown": "^9.x.x"` 추가됨

- [ ] **Step 2: 빌드 확인**

```bash
npm run build
```

Expected: 오류 없이 빌드 성공

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore: add react-markdown dependency"
```

---

## Task 2: useChat 에러 핸들러 개선

**Files:**
- Modify: `frontend/src/hooks/useChat.ts:69-71`

현재 catch 블록에서 사용자 메시지를 제거하고 `console.error`만 호출함. 에러 메시지를 AI 버블로 표시하도록 변경.

- [ ] **Step 1: catch 블록 수정**

`frontend/src/hooks/useChat.ts`의 catch 블록(line 69-71)을 다음으로 교체:

```ts
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
}
```

- [ ] **Step 2: 개발 서버 실행 및 타입 오류 없음 확인**

```bash
npm run dev
```

Expected: TypeScript 오류 없음

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useChat.ts
git commit -m "fix: show error message bubble on chat failure"
```

---

## Task 3: FilterBar 세로 레이아웃 + 초기화 버튼

**Files:**
- Modify: `frontend/src/components/FilterBar.tsx`

- [ ] **Step 1: FilterBar 전체 교체**

`frontend/src/components/FilterBar.tsx` 파일을 다음으로 교체:

```tsx
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

export function FilterBar({ filters, onChange }: Props) {
  return (
    <div className="p-3 border-t border-gray-700 flex flex-col gap-2">
      <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">필터</p>
      <select
        value={filters.doc_type ?? ''}
        onChange={e => onChange({ ...filters, doc_type: e.target.value || null })}
        className="text-sm border border-gray-600 rounded px-2 py-1 bg-gray-800 text-gray-200 w-full"
      >
        <option value="">전체 문서</option>
        <option value="sanctions">제재결과</option>
        <option value="management">경영유의사항</option>
      </select>
      <div className="flex flex-col gap-1">
        <input
          type="date"
          value={filters.date_from ?? ''}
          onChange={e => onChange({ ...filters, date_from: e.target.value || null })}
          className="text-sm border border-gray-600 rounded px-2 py-1 bg-gray-800 text-gray-200 w-full"
        />
        <input
          type="date"
          value={filters.date_to ?? ''}
          onChange={e => onChange({ ...filters, date_to: e.target.value || null })}
          className="text-sm border border-gray-600 rounded px-2 py-1 bg-gray-800 text-gray-200 w-full"
        />
      </div>
      <input
        type="text"
        value={filters.institution ?? ''}
        onChange={e => onChange({ ...filters, institution: e.target.value || null })}
        placeholder="기관명"
        className="text-sm border border-gray-600 rounded px-2 py-1 bg-gray-800 text-gray-200 w-full placeholder-gray-500"
      />
      <button
        onClick={() => onChange(EMPTY_FILTERS)}
        className="text-xs text-gray-400 hover:text-gray-200 text-left transition-colors"
      >
        필터 초기화
      </button>
    </div>
  );
}
```

- [ ] **Step 2: 브라우저에서 필터 렌더링 확인** (개발 서버 실행 중이면 바로 확인)

Expected: 세로 배치 4개 입력 + 초기화 버튼 보임

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/FilterBar.tsx
git commit -m "feat: vertical FilterBar layout with reset button"
```

---

## Task 4: Sidebar에 FilterBar 통합 + 서비스명

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Sidebar 전체 교체**

`frontend/src/components/Sidebar.tsx` 파일을 다음으로 교체:

```tsx
import type { Conversation, FilterParams } from '../types';
import { FilterBar } from './FilterBar';

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
    <aside className="w-64 bg-gray-900 text-white flex flex-col h-full flex-shrink-0">
      {/* 상단: 서비스명 + 새 대화 */}
      <div className="p-4 border-b border-gray-700 flex-shrink-0">
        <p className="text-xs text-gray-500 mb-2 font-medium">FSS 챗봇</p>
        <button
          onClick={onNew}
          className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
        >
          + 새 대화
        </button>
      </div>

      {/* 중단: 대화 목록 (남은 공간 채움, 넘치면 스크롤) */}
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
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

      {/* 하단: 필터 (항상 고정) */}
      <div className="flex-shrink-0">
        <FilterBar filters={filters} onChange={onFilterChange} />
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: 타입 오류 없음 확인**

```bash
npm run build 2>&1 | head -20
```

Expected: 오류 없음

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: integrate FilterBar into Sidebar with service name"
```

---

## Task 5: App.tsx에서 헤더·FilterBar 제거 및 props 전달

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: App.tsx 전체 교체**

```tsx
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
    <div className="flex h-screen bg-gray-100">
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
```

- [ ] **Step 2: 브라우저에서 레이아웃 확인**

- 사이드바에 "FSS 챗봇" + 새 대화 버튼 + 필터 보임
- 채팅 영역에 헤더 없음
- FilterBar가 채팅 영역 상단에 없음

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: remove header and move FilterBar to Sidebar"
```

---

## Task 6: ChatWindow — 예시 카드 + bouncing 로딩 + 전송 아이콘

**Files:**
- Modify: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: ChatWindow 전체 교체**

```tsx
import { useEffect, useRef, useState } from 'react';
import type { Message } from '../types';
import { MessageBubble } from './MessageBubble';

interface Props {
  messages: Message[];
  isLoading: boolean;
  onSend: (text: string) => void;
}

const EXAMPLE_QUESTIONS = [
  '2024년 은행권 제재 현황을 요약해줘',
  '전자금융 관련 최근 제재 사례를 알려줘',
  '최근 과태료 부과 사례는?',
];

function BouncingDots() {
  return (
    <div className="flex gap-1 items-center px-1">
      {[0, 150, 300].map(delay => (
        <span
          key={delay}
          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce inline-block"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

export function ChatWindow({ messages, isLoading, onSend }: Props) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center mt-20 gap-6">
            <p className="text-gray-500 text-lg font-medium">무엇이든 물어보세요</p>
            <div className="flex flex-wrap gap-3 justify-center max-w-2xl">
              {EXAMPLE_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => onSend(q)}
                  className="px-4 py-3 bg-white border border-gray-200 rounded-xl text-sm text-gray-700 hover:border-blue-400 hover:text-blue-700 transition-colors shadow-sm text-left max-w-xs"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white border rounded-2xl px-4 py-3 shadow-sm">
              <BouncingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
        <div className="flex gap-2 items-end">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="flex-1 border rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className={`p-2.5 rounded-xl transition-colors ${
              input.trim() && !isLoading
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
            aria-label="전송"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
              <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}
```

- [ ] **Step 2: 브라우저에서 확인**

- 빈 화면에 예시 카드 3개 보임
- 카드 클릭 시 질문이 전송되고 로딩 bouncing dots 나타남
- 입력 없으면 전송 버튼 회색, 입력 있으면 파란색

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatWindow.tsx
git commit -m "feat: example question cards, bouncing loader, send icon"
```

---

## Task 7: MessageBubble — react-markdown 렌더링

**Files:**
- Modify: `frontend/src/components/MessageBubble.tsx`

- [ ] **Step 1: MessageBubble 수정**

`frontend/src/components/MessageBubble.tsx` 파일을 다음으로 교체:

```tsx
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Message } from '../types';
import { SourceCard } from './SourceCard';

export function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 shadow-sm ${
          isUser ? 'bg-blue-100 text-blue-900' : 'bg-white border text-gray-800'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        ) : (
          <div className="[&_h1]:text-base [&_h1]:font-bold [&_h1]:mb-2 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:my-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:my-1 [&_li]:mb-0.5 [&_p]:mb-2 [&_p:last-child]:mb-0 leading-relaxed text-sm">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && message.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="text-xs text-blue-600 hover:underline"
            >
              참고 문서 {message.sources.length}개 {showSources ? '접기 ▲' : '보기 ▼'}
            </button>
            {showSources && (
              <div className="mt-2">
                {message.sources.map((src, i) => (
                  <SourceCard key={i} source={src} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 브라우저에서 마크다운 렌더링 확인**

질문을 전송하고 AI 답변에서:
- 볼드(`**텍스트**`) → `<strong>` 렌더링
- 목록(`- 항목`) → bullet 렌더링
- 줄바꿈 유지

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/MessageBubble.tsx
git commit -m "feat: render AI responses with react-markdown"
```

---

## Task 8: 최종 통합 테스트

- [ ] **Step 1: 개발 서버 + 백엔드 동시 실행**

터미널 1 (백엔드):
```bash
cd d:/work/rag/RAG_eight && python -m uvicorn backend.main:app --reload --port 8000
```

터미널 2 (프론트엔드):
```bash
cd d:/work/rag/RAG_eight/frontend && npm run dev
```

- [ ] **Step 2: 전체 플로우 확인**

브라우저 `http://localhost:5173` 에서:

| 시나리오 | 기대 결과 |
|----------|----------|
| 첫 접속 | 예시 카드 3개 표시, 헤더 없음, 필터 사이드바 하단에 보임 |
| 예시 카드 클릭 | 질문 전송, bouncing dots 표시, 마크다운 답변 렌더링 |
| 필터 설정 후 질문 | 필터 적용된 답변 |
| 필터 초기화 버튼 | 모든 필터 빈 값으로 초기화 |
| 참고 문서 보기 | SourceCard 토글 정상 동작 |

- [ ] **Step 3: 빌드 오류 없음 확인**

```bash
cd frontend && npm run build
```

Expected: 오류 없이 빌드 완료
