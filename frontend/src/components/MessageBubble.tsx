import { useState } from 'react';
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
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
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
