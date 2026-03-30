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
              className="flex-1 resize-none overflow-y-auto bg-transparent text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none text-sm leading-relaxed"
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
