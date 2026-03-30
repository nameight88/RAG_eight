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
