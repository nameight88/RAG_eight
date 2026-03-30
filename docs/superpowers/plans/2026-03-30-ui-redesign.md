# UI 리디자인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FSS 챗봇 프론트엔드를 Claude.ai/Notion 스타일의 라이트/다크 테마 지원 UI로 전면 개선한다.

**Architecture:** CSS 변수로 색상 토큰 시스템을 구축하고 Tailwind v4의 `@theme` 블록으로 유틸리티 클래스를 등록한다. `useTheme` 훅이 `localStorage` 기반 테마 상태를 관리하며 `html[data-theme]` 속성으로 전환한다. 모든 컴포넌트는 하드코딩된 Tailwind 색상 클래스 대신 CSS 변수 기반 클래스를 사용한다.

**Tech Stack:** React 19, TypeScript, Tailwind CSS v4 (`@tailwindcss/vite`), Vite 8

> **참고:** 이 프로젝트에는 테스트 인프라가 없다. 각 태스크 후 검증은 TypeScript 빌드(`cd frontend && npm run build`)로 수행한다. `dev` 서버(http://localhost:5173)에서 시각적으로 확인한다.

---

## 파일 구조

```
frontend/
├── index.html                          수정 — FOUC 방지 스크립트
├── src/
│   ├── index.css                       수정 — CSS 변수 + @theme 블록
│   ├── App.css                         삭제 — Vite 보일러플레이트, 미사용
│   ├── App.tsx                         수정 — root bg 토큰 교체
│   ├── hooks/
│   │   ├── useChat.ts                  변경 없음
│   │   └── useTheme.ts                 신규 — 테마 상태 관리
│   └── components/
│       ├── ThemeToggle.tsx             신규 — 해/달 토글 버튼
│       ├── Sidebar.tsx                 수정 — 헤더/목록/스타일
│       ├── FilterBar.tsx               수정 — 레이블/스타일
│       ├── ChatWindow.tsx              수정 — textarea, 빈화면, 입력카드
│       ├── MessageBubble.tsx           수정 — 토큰 기반 색상
│       └── SourceCard.tsx              수정 — 토큰 기반 색상
```

---

## Task 1: CSS 토큰 시스템 + FOUC 방지

**Files:**
- Modify: `frontend/src/index.css`
- Modify: `frontend/index.html`

- [ ] **Step 1: index.css에 CSS 변수 + @theme 블록 추가**

`frontend/src/index.css`를 다음으로 교체:

```css
@import "tailwindcss";

/* ── 라이트 테마 (기본) ── */
:root {
  --bg-base: #f9fafb;
  --bg-surface: #ffffff;
  --bg-elevated: #f3f4f6;
  --border: #e5e7eb;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --user-bubble: #eff6ff;
  --user-text: #1e40af;
}

/* ── 다크 테마 ── */
[data-theme="dark"] {
  --bg-base: #111827;
  --bg-surface: #1f2937;
  --bg-elevated: #374151;
  --border: #4b5563;
  --text-primary: #f9fafb;
  --text-secondary: #9ca3af;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --user-bubble: #1e3a5f;
  --user-text: #bfdbfe;
}

/* ── Tailwind v4 @theme: CSS 변수를 유틸리티 클래스로 등록 ── */
@theme {
  --color-base: var(--bg-base);
  --color-surface: var(--bg-surface);
  --color-elevated: var(--bg-elevated);
  --color-border: var(--border);
  --color-text-primary: var(--text-primary);
  --color-text-secondary: var(--text-secondary);
  --color-accent: var(--accent);
  --color-accent-hover: var(--accent-hover);
}
```

- [ ] **Step 2: index.html에 FOUC 방지 스크립트 추가**

`frontend/index.html`의 `<head>` 닫는 태그 바로 앞에 삽입:

```html
    <script>
      (function() {
        var t = localStorage.getItem('theme');
        if (t === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
      })();
    </script>
  </head>
```

결과:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FSS 챗봇</title>
    <script>
      (function() {
        var t = localStorage.getItem('theme');
        if (t === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
      })();
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 3: 빌드 검증**

```bash
cd frontend && npm run build
```

Expected: 빌드 성공, 오류 없음.

- [ ] **Step 4: 커밋**

```bash
cd frontend && git add src/index.css index.html
git commit -m "feat: add CSS token system and FOUC prevention"
```

---

## Task 2: useTheme 훅 + ThemeToggle 컴포넌트

**Files:**
- Create: `frontend/src/hooks/useTheme.ts`
- Create: `frontend/src/components/ThemeToggle.tsx`

- [ ] **Step 1: useTheme 훅 작성**

`frontend/src/hooks/useTheme.ts` 생성:

```typescript
import { useState, useCallback } from 'react';

type Theme = 'light' | 'dark';

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('theme') as Theme) ?? 'light';
  });

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next: Theme = prev === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      return next;
    });
  }, []);

  return { theme, toggleTheme };
}
```

- [ ] **Step 2: ThemeToggle 컴포넌트 작성**

`frontend/src/components/ThemeToggle.tsx` 생성:

```tsx
import { useTheme } from '../hooks/useTheme';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      aria-label={theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환'}
      aria-pressed={theme === 'dark'}
      className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] transition-colors"
    >
      {theme === 'dark' ? (
        /* 해 아이콘 (라이트로 전환) */
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
          <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.166a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.061l1.591-1.59zM21.75 12a.75.75 0 01-.75.75h-2.25a.75.75 0 010-1.5H21a.75.75 0 01.75.75zM17.834 18.894a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 10-1.061 1.06l1.59 1.591zM12 18a.75.75 0 01.75.75V21a.75.75 0 01-1.5 0v-2.25A.75.75 0 0112 18zM7.758 17.303a.75.75 0 00-1.061-1.06l-1.591 1.59a.75.75 0 001.06 1.061l1.591-1.59zM6 12a.75.75 0 01-.75.75H3a.75.75 0 010-1.5h2.25A.75.75 0 016 12zM6.697 7.757a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 00-1.061 1.06l1.59 1.591z" />
        </svg>
      ) : (
        /* 달 아이콘 (다크로 전환) */
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M9.528 1.718a.75.75 0 01.162.819A8.97 8.97 0 009 6a9 9 0 009 9 8.97 8.97 0 003.463-.69.75.75 0 01.981.98 10.503 10.503 0 01-9.694 6.46c-5.799 0-10.5-4.701-10.5-10.5 0-4.368 2.667-8.112 6.46-9.694a.75.75 0 01.818.162z" clipRule="evenodd" />
        </svg>
      )}
    </button>
  );
}
```

- [ ] **Step 3: 빌드 검증**

```bash
cd frontend && npm run build
```

Expected: 빌드 성공.

- [ ] **Step 4: 커밋**

```bash
cd frontend && git add src/hooks/useTheme.ts src/components/ThemeToggle.tsx
git commit -m "feat: add useTheme hook and ThemeToggle component"
```

---

## Task 3: App.tsx 루트 배경 토큰 교체

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: App.tsx의 bg-gray-100을 토큰으로 교체**

`frontend/src/App.tsx` line 25의 `bg-gray-100`을 `bg-[var(--bg-base)]`로 교체:

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
```

- [ ] **Step 2: 빌드 검증**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: App.css 삭제**

`frontend/src/App.css` 파일을 삭제한다. 현재 `App.tsx`, `main.tsx` 어느 곳에서도 import되지 않으므로 빌드에 영향 없다.

- [ ] **Step 4: 커밋**

```bash
cd frontend && git add src/App.tsx
git rm src/App.css
git commit -m "feat: apply theme token to root layout, remove unused App.css"
```

---

## Task 4: Sidebar 리디자인

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Sidebar 전면 교체**

`frontend/src/components/Sidebar.tsx`를 다음으로 교체:

```tsx
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
```

- [ ] **Step 2: 빌드 검증**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: 커밋**

```bash
cd frontend && git add src/components/Sidebar.tsx
git commit -m "feat: redesign Sidebar with light theme and branding"
```

---

## Task 5: FilterBar 리디자인

**Files:**
- Modify: `frontend/src/components/FilterBar.tsx`

- [ ] **Step 1: FilterBar 전면 교체**

`frontend/src/components/FilterBar.tsx`를 다음으로 교체:

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
```

- [ ] **Step 2: 빌드 검증**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: 커밋**

```bash
cd frontend && git add src/components/FilterBar.tsx
git commit -m "feat: redesign FilterBar with labels and theme tokens"
```

---

## Task 6: ChatWindow 리디자인

**Files:**
- Modify: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: ChatWindow 전면 교체**

`frontend/src/components/ChatWindow.tsx`를 다음으로 교체:

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
          className="w-2 h-2 bg-[var(--text-secondary)] rounded-full animate-bounce inline-block"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

export function ChatWindow({ messages, isLoading, onSend }: Props) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const submit = () => {
    if (input.trim() && !isLoading) {
      onSend(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submit();
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden bg-[var(--bg-base)]">
      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full gap-6 min-h-[300px]">
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent)] flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-6 h-6">
                  <path fillRule="evenodd" d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-[var(--text-primary)] text-xl font-semibold">무엇이든 물어보세요</p>
              <p className="text-[var(--text-secondary)] text-sm">금융감독원 제재·경영유의 문서를 검색합니다</p>
            </div>
            <div className="flex flex-wrap gap-2.5 justify-center max-w-lg">
              {EXAMPLE_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => onSend(q)}
                  className="px-4 py-2.5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-xl text-sm text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors shadow-sm text-left"
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
            <div className="bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl px-4 py-3 shadow-sm">
              <BouncingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* 입력 영역 — 카드 스타일 */}
      <div className="px-4 pb-4 flex-shrink-0">
        <form onSubmit={handleSubmit}>
          <div className="relative bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl shadow-md flex items-end gap-2 px-4 py-3">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="질문을 입력하세요... (Shift+Enter: 줄바꿈)"
              rows={1}
              className="flex-1 resize-none overflow-hidden bg-transparent text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none text-sm leading-relaxed"
              style={{ maxHeight: '120px' }}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className={`flex-shrink-0 p-2 rounded-xl transition-colors ${
                input.trim() && !isLoading
                  ? 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white'
                  : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] cursor-not-allowed'
              }`}
              aria-label="전송"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 빌드 검증**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: 커밋**

```bash
cd frontend && git add src/components/ChatWindow.tsx
git commit -m "feat: redesign ChatWindow with textarea input and improved empty state"
```

---

## Task 7: MessageBubble + SourceCard 리디자인

**Files:**
- Modify: `frontend/src/components/MessageBubble.tsx`
- Modify: `frontend/src/components/SourceCard.tsx`

- [ ] **Step 1: MessageBubble 교체**

`frontend/src/components/MessageBubble.tsx`를 다음으로 교체:

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
          isUser
            ? 'bg-[var(--user-bubble)] text-[var(--user-text)]'
            : 'bg-[var(--bg-surface)] border border-[var(--border)] text-[var(--text-primary)]'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap leading-relaxed text-sm">{message.content}</p>
        ) : (
          <div className="[&_h1]:text-base [&_h1]:font-bold [&_h1]:mb-2 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:my-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:my-1 [&_li]:mb-0.5 [&_p]:mb-2 [&_p:last-child]:mb-0 leading-relaxed text-sm">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && message.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="inline-flex items-center gap-1 text-xs bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded-full px-3 py-1 hover:text-[var(--text-primary)] transition-colors"
            >
              참고 문서 {message.sources.length}개 {showSources ? '▲' : '▼'}
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

- [ ] **Step 2: SourceCard 교체**

`frontend/src/components/SourceCard.tsx`를 다음으로 교체:

```tsx
import type { SourceDoc } from '../types';

export function SourceCard({ source }: { source: SourceDoc }) {
  return (
    <div className="bg-[var(--bg-elevated)] border-l-4 border-[var(--accent)] p-3 rounded-lg text-sm mb-2">
      <div className="font-semibold text-[var(--text-primary)]">{source.institution || '알 수 없음'}</div>
      <div className="text-[var(--text-secondary)] text-xs mb-1">
        {source.date} · {source.doc_type} · 유사도 {(source.score * 100).toFixed(0)}%
      </div>
      <div className="text-[var(--text-primary)] line-clamp-2">{source.content}</div>
    </div>
  );
}
```

- [ ] **Step 3: 최종 빌드 검증**

```bash
cd frontend && npm run build
```

Expected: 빌드 성공, TypeScript 오류 없음.

- [ ] **Step 4: 커밋**

```bash
cd frontend && git add src/components/MessageBubble.tsx src/components/SourceCard.tsx
git commit -m "feat: apply theme tokens to MessageBubble and SourceCard"
```

---

## 완료 기준

모든 태스크 완료 후 http://localhost:5173 에서 확인:
1. 라이트 테마: 전체 흰색/회색 배경, 사이드바 라이트, 필터에 시작일/종료일 레이블 표시
2. 다크 테마: 사이드바 헤더 우측 달 아이콘 클릭 → 전체 다크 전환, 새로고침 후에도 유지
3. 텍스트 입력: Enter 전송, Shift+Enter 줄바꿈, 입력창 자동 높이 조절
4. 예시 질문 카드: hover 시 accent border/텍스트 색상 변경
