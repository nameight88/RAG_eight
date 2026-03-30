# UI 리디자인 설계 문서

**날짜**: 2026-03-30
**프로젝트**: RAG_eight — FSS 챗봇 프론트엔드
**목표**: 전체적인 UI/UX 시각 개선 (Claude.ai / Notion 스타일, 라이트/다크 테마 지원)

---

## 1. 배경 및 목표

현재 UI의 주요 문제점:
- FilterBar 날짜 입력 필드에 레이블 없음 (시작일/종료일 구분 불가)
- 사이드바 하단 필터 영역이 너무 밀집되고 비직관적
- "FSS 챗봇" 브랜딩이 작고 존재감 없음
- 전체 색상이 gray-900 다크 사이드바와 흰색 메인의 단순 조합으로 세련되지 않음
- 다크 모드 미지원

**목표**: Claude.ai / Notion 스타일의 모던·미니멀 라이트 테마, 다크 테마 토글 기능 추가.

---

## 2. 기술 접근 방식

**CSS 변수 기반 테마 시스템** 채택.

- `index.css`에 `:root`(라이트)와 `[data-theme="dark"]` 셀렉터로 CSS 변수 정의
- Tailwind v4를 사용하므로 `tailwind.config.js` 대신 **`index.css` 내 `@theme` 블록**으로 커스텀 색상 등록
  ```css
  @theme {
    --color-surface: var(--bg-surface);
    --color-base: var(--bg-base);
    --color-elevated: var(--bg-elevated);
    --color-border: var(--border);
    --color-text-primary: var(--text-primary);
    --color-text-secondary: var(--text-secondary);
    --color-accent: var(--accent);
    --color-accent-hover: var(--accent-hover);
  }
  ```
  이렇게 하면 `bg-surface`, `bg-base`, `text-primary` 등의 Tailwind 유틸리티 클래스가 동작함.
- `useTheme` 훅으로 테마 상태 관리, `localStorage`에 저장
- `html` 요소의 `data-theme` 속성으로 전환

---

## 3. 색상 토큰

### 라이트 테마 (기본, `:root`)

| 변수 | 값 | 용도 |
|---|---|---|
| `--bg-base` | `#f9fafb` | 전체 배경 |
| `--bg-surface` | `#ffffff` | 사이드바, 카드, 입력창 |
| `--bg-elevated` | `#f3f4f6` | 호버, 활성 항목 |
| `--border` | `#e5e7eb` | 구분선 |
| `--text-primary` | `#111827` | 주요 텍스트 |
| `--text-secondary` | `#6b7280` | 보조 텍스트 |
| `--accent` | `#2563eb` | 버튼, 링크, 강조 |
| `--accent-hover` | `#1d4ed8` | 버튼 호버 |
| `--user-bubble` | `#eff6ff` | 사용자 메시지 배경 |
| `--user-text` | `#1e40af` | 사용자 메시지 텍스트 |

### 다크 테마 (`[data-theme="dark"]`)

| 변수 | 값 | 용도 |
|---|---|---|
| `--bg-base` | `#111827` | 전체 배경 |
| `--bg-surface` | `#1f2937` | 사이드바, 카드, 입력창 |
| `--bg-elevated` | `#374151` | 호버, 활성 항목 |
| `--border` | `#4b5563` | 구분선 (elevated와 구별되는 값) |
| `--text-primary` | `#f9fafb` | 주요 텍스트 |
| `--text-secondary` | `#9ca3af` | 보조 텍스트 |
| `--accent` | `#3b82f6` | 버튼, 링크, 강조 |
| `--accent-hover` | `#2563eb` | 버튼 호버 |
| `--user-bubble` | `#1e3a5f` | 사용자 메시지 배경 |
| `--user-text` | `#bfdbfe` | 사용자 메시지 텍스트 |

---

## 4. 컴포넌트 설계

### 4.1 신규: `useTheme` 훅

**위치**: `frontend/src/hooks/useTheme.ts` (hooks 디렉토리는 이미 존재함)

- `theme: 'light' | 'dark'` 상태 관리
- `toggleTheme()` 함수 제공
- 초기값: `localStorage.getItem('theme') as 'light' | 'dark'` → 없으면 `'light'`
- 변경 시: `document.documentElement.setAttribute('data-theme', newTheme)` 및 `localStorage.setItem('theme', newTheme)` 호출

### 4.2 신규: `ThemeToggle` 컴포넌트

**위치**: `frontend/src/components/ThemeToggle.tsx`

- 라이트 모드: 달 아이콘(🌙 또는 SVG), 다크 모드: 해 아이콘(☀️ 또는 SVG)
- `useTheme` 훅 사용
- 사이드바 헤더 우측에 배치
- 접근성: `aria-label={theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환'}` 및 `aria-pressed={theme === 'dark'}` 추가

### 4.3 `Sidebar` 변경

- 배경: `bg-surface` (`bg-[var(--bg-surface)]`), 우측 `border-r border-[var(--border)]`
- **헤더 영역**: 건물 아이콘(SVG 또는 🏛️) + "FSS 챗봇" 텍스트(굵게, `text-primary`), 우측에 `ThemeToggle`
- **새 대화 버튼**: 플러스 아이콘 + "새 대화", `bg-[var(--accent)] hover:bg-[var(--accent-hover)]` text-white
- **대화 목록 항목**:
  - 비활성: hover 시 `bg-[var(--bg-elevated)]`
  - 활성: 좌측 3px solid `var(--accent)` border + `bg-[var(--bg-elevated)]`, `text-primary`
- **필터 섹션**: `FilterBar` 개선 참조

### 4.4 `FilterBar` 변경

- 배경: `bg-[var(--bg-surface)]`, `border-t border-[var(--border)]`
- "필터" 레이블: `text-[var(--text-secondary)]`, uppercase tracking
- **doc_type select**: border `var(--border)`, `bg-[var(--bg-elevated)]`, `text-[var(--text-primary)]`, rounded-lg
- **날짜 입력**: 각 input 위에 `<label>` 태그로 `시작일` / `종료일` 레이블 추가, 동일 스타일
- **기관명 input**: 동일 스타일, placeholder 유지
- **필터 초기화**: 텍스트 링크 → 회색 pill 버튼 (`bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded-full px-3 py-1 text-xs hover:text-[var(--text-primary)]`)

### 4.5 `ChatWindow` 변경

- 메시지 영역 배경: `bg-[var(--bg-base)]`
- **빈 화면 (Empty State)**:
  - 문서/챗 아이콘 추가
  - "무엇이든 물어보세요" 텍스트 크기 업 (`text-xl`)
  - 예시 질문 카드: `bg-[var(--bg-surface)] border border-[var(--border)]` + shadow-sm, hover 시 `border-[var(--accent)] text-[var(--accent)]`
- **입력 영역**:
  - 하단 `border-t` 제거 → 카드 형태 (`bg-[var(--bg-surface)] shadow-md rounded-2xl mx-4 mb-4`)
  - `input` → `textarea` (rows=1, `overflow-hidden`, `resize-none`)
  - 자동 높이 조절: `onInput` 핸들러에서 `e.target.style.height = 'auto'; e.target.style.height = e.target.scrollHeight + 'px'`; CSS로 `max-height` 를 5행(`~120px`)으로 제한
  - 키보드 동작: `Enter` → 제출, `Shift+Enter` → 줄바꿈 (`onKeyDown` 핸들러 추가)
  - 전송 버튼을 `textarea` 우측 내부에 `absolute` 배치 (wrapper `relative`)
- **로딩 표시기 (`BouncingDots`)**: `bg-gray-400` → `bg-[var(--text-secondary)]`

### 4.6 `MessageBubble` 변경

- **사용자 메시지**: `bg-[var(--user-bubble)]`, `text-[var(--user-text)]`
- **AI 메시지**: `bg-[var(--bg-surface)] shadow-sm border border-[var(--border)]`
- **참고 문서 버튼**: pill 스타일 (`bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded-full px-3 py-1 text-xs hover:text-[var(--text-primary)]`)

### 4.7 `SourceCard` 변경

- `bg-[var(--bg-elevated)]` 배경, `border-l-4 border-[var(--accent)]`
- 기관명: `text-[var(--accent)]` 또는 `text-[var(--text-primary)] font-semibold`
- 메타 정보: `text-[var(--text-secondary)]`
- 본문: `text-[var(--text-primary)]`

---

## 5. 파일 변경 목록

| 파일 | 변경 유형 |
|---|---|
| `frontend/index.html` | 수정 — FOUC 방지 인라인 스크립트 추가 |
| `frontend/src/index.css` | 수정 — CSS 변수 + `@theme` 블록 추가 |
| `frontend/src/App.css` | 삭제 또는 전체 비움 — Vite 보일러플레이트 잔재, 현재 앱에서 미사용 |
| `frontend/src/App.tsx` | 수정 — root div `bg-gray-100` → `bg-[var(--bg-base)]` |
| `frontend/src/hooks/useTheme.ts` | 신규 |
| `frontend/src/components/ThemeToggle.tsx` | 신규 |
| `frontend/src/components/Sidebar.tsx` | 수정 |
| `frontend/src/components/FilterBar.tsx` | 수정 |
| `frontend/src/components/ChatWindow.tsx` | 수정 |
| `frontend/src/components/MessageBubble.tsx` | 수정 |
| `frontend/src/components/SourceCard.tsx` | 수정 |

---

## 6. 비기능 요건

### FOUC 방지

`frontend/index.html`의 `<head>` 안에 다음 인라인 스크립트를 추가해 페이지 로드 시 깜빡임 없이 테마를 즉시 적용:

```html
<script>
  (function() {
    var t = localStorage.getItem('theme');
    if (t === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
  })();
</script>
```

### 기타
- 기존 기능(메시지 전송, 필터, 대화 목록) 동작 변경 없음
- 반응형 고려 없음 (데스크탑 전용 현행 유지)
- `App.css`는 삭제한다 (import 여부와 무관하게). `main.tsx`에 해당 import가 있으면 함께 제거 (현재 없음)
