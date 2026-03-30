# Frontend UX Redesign — ChatGPT Style

**Date:** 2026-03-26
**Status:** Approved (v2)
**Scope:** `frontend/src/` 전체 컴포넌트 개선 (B안)

---

## 목표

현재 기능적으로만 동작하는 채팅 UI를 일반적인 ChatGPT 스타일 챗봇처럼 개선한다.
외부 라이브러리는 `react-markdown` 하나만 추가한다 (`@tailwindcss/typography` 미사용).

---

## 레이아웃 구조

```
┌──────────────┬──────────────────────────────────────┐
│   Sidebar    │           Chat Area                  │
│  (w-64)      │                                      │
│              │  [빈 화면: 예시 질문 카드 3개]          │
│ "FSS 챗봇"   │                                      │
│ [새 대화]     │  메시지 목록 (스크롤)                  │
│              │                                      │
│ 대화 목록     │                                      │
│ (flex-1,     │  ──────────────────────────────      │
│  min-h-0,    │       [입력창 + 전송 아이콘(↑)]        │
│  scroll)     └──────────────────────────────────────┘
│ ──────────────
│ [필터 영역] (flex-shrink-0)
│ • 문서유형 select
│ • 날짜 from~to
│ • 기관명 input
│ • [필터 초기화] 버튼
└──────────────┘
```

**변경 사항:**
- `FilterBar` 컴포넌트를 `App.tsx` 상단에서 제거하고 `Sidebar` 하단으로 이동
- 채팅 영역 상단 헤더(`<header>`) 제거
- 대화 목록: `flex-1 min-h-0 overflow-y-auto` — 필터 영역이 있어도 압축됨
- 필터 영역: `flex-shrink-0` — 항상 고정 높이 유지

**모바일:** 이번 범위 밖 (데스크탑 전용)

---

## 컴포넌트별 변경

### App.tsx
- `<FilterBar>` 제거
- `<header>` 제거
- `<Sidebar>`에 `filters`, `onChange` 두 props 추가 전달

```tsx
<Sidebar
  conversations={conversations}
  activeId={activeConversationId}
  onSelect={selectConversation}
  onNew={newConversation}
  filters={filters}
  onFilterChange={setFilters}
/>
```

### Sidebar.tsx
Props 인터페이스 추가:
```ts
interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  filters: FilterParams;
  onFilterChange: (f: FilterParams) => void;
}
```
레이아웃: `flex-col h-full` — 상단 헤더 | 중단 대화목록(flex-1 min-h-0) | 하단 필터(flex-shrink-0)

### FilterBar.tsx
- 세로 레이아웃 (`flex-col gap-2 p-3`)
- 라벨 + 입력 쌍 구성
- 하단에 "필터 초기화" 버튼 추가:
  ```tsx
  <button onClick={() => onChange({ doc_type: null, date_from: null, date_to: null, institution: null })}>
    필터 초기화
  </button>
  ```

### ChatWindow.tsx

**빈 화면 예시 카드 (3개):**
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ 2024년 은행권    │  │ 전자금융 관련    │  │ 최근 과태료     │
│ 제재 현황 요약   │  │ 최근 제재 사례   │  │ 부과 사례       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```
카드 클릭 시 `onSend(text)` 직접 호출 (입력창 채우기 없이 바로 전송 — Option A).
입력창은 빈 상태 유지.

**로딩 애니메이션 (bouncing dots):**
```tsx
<span style={{ animationDelay: '0ms' }} className="animate-bounce">●</span>
<span style={{ animationDelay: '150ms' }} className="animate-bounce">●</span>
<span style={{ animationDelay: '300ms' }} className="animate-bounce">●</span>
```
animation-delay는 인라인 스타일로 적용 (Tailwind 유틸리티 미사용).

**에러 상태:**
`sendMessage` 실패 시 AI 버블 자리에 인라인 에러 메시지 표시:
```tsx
// useChat.ts에서 에러 메시지를 assistant 역할로 messages에 push
{ role: 'assistant', content: '⚠️ 답변을 가져오는 데 실패했습니다. 잠시 후 다시 시도해주세요.', sources: [] }
```

**입력창:**
- 전송 버튼 텍스트 → SVG 아이콘 (↑ 화살표)
- 입력 있을 때: `bg-blue-600`, 없을 때: `bg-gray-300`

### MessageBubble.tsx
- AI 답변: `react-markdown` 으로 렌더링
- prose 스타일은 `@tailwindcss/typography` 없이 직접 className으로 적용:
  ```tsx
  <div className="prose-sm [&_h1]:text-base [&_h2]:text-sm [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_p]:mb-2">
    <ReactMarkdown>{message.content}</ReactMarkdown>
  </div>
  ```
- `rehype-raw` 플러그인 미사용 (raw HTML 렌더링 차단 유지)
- 사용자 메시지: 변경 없음

---

## 의존성

```bash
npm install react-markdown
```

`@tailwindcss/typography` 미추가 (v4 호환 불확실, 직접 유틸리티로 대체).
`index.css`의 `@tailwind` 디렉티브는 현재 그대로 유지 (v4 마이그레이션 범위 밖).

---

## 파일 변경 목록

| 파일 | 변경 유형 |
|------|----------|
| `frontend/src/App.tsx` | 수정 — FilterBar 제거, 헤더 제거, Sidebar에 filter props 전달 |
| `frontend/src/components/Sidebar.tsx` | 수정 — FilterBar 통합, 서비스명, props 인터페이스 확장 |
| `frontend/src/components/FilterBar.tsx` | 수정 — 세로 레이아웃, 초기화 버튼 추가 |
| `frontend/src/components/ChatWindow.tsx` | 수정 — 예시 카드, bouncing dots, 아이콘 버튼 |
| `frontend/src/components/MessageBubble.tsx` | 수정 — react-markdown 적용 |
| `frontend/src/hooks/useChat.ts` | 수정 — 에러 시 assistant 에러 메시지 push |
| `frontend/package.json` | 수정 — react-markdown 추가 |

## 비변경 사항

- `types.ts`, `api/client.ts` — 변경 없음
- `SourceCard.tsx` — 변경 없음
- 백엔드 API — 변경 없음
- `tailwind.config.js`, `index.css` — 변경 없음
