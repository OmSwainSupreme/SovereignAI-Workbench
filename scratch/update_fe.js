const fs = require('fs');

// 1. Update C:/lovable/src/services/types.ts
{
  const file = 'C:/lovable/src/services/types.ts';
  let content = fs.readFileSync(file, 'utf8');
  if (!content.includes('thinking?: string;')) {
    content = content.replace(
      'pending?: boolean;',
      'pending?: boolean;\n  thinking?: string;'
    );
  }
  if (!content.includes('thinking-delta')) {
    content = content.replace(
      '| { type: "token-delta"; messageId: string; text: string }',
      '| { type: "token-delta"; messageId: string; text: string }\n  | { type: "thinking-delta"; messageId: string; text: string }'
    );
  }
  fs.writeFileSync(file, content, 'utf8');
  console.log('1. types.ts updated');
}

// 2. Update C:/lovable/src/features/workspace/streamReducer.ts
{
  const file = 'C:/lovable/src/features/workspace/streamReducer.ts';
  let content = fs.readFileSync(file, 'utf8');
  
  if (!content.includes('applyThinkingDelta')) {
    const applyDeltaStr = `function applyDelta(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] {`;
    const applyThinkingDeltaStr = `function applyThinkingDelta(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] {
  const index = messages.findIndex((m) => m.id === messageId);
  if (index === -1) {
    return [
      ...messages,
      {
        id: messageId,
        role: "agent",
        content: "",
        thinking: text,
        createdAt: new Date().toISOString(),
        pending: true,
      },
    ];
  }
  const next = [...messages];
  const current = next[index]!;
  next[index] = { ...current, thinking: (current.thinking || "") + text, pending: true };
  return next;
}\n\n` + applyDeltaStr;

    content = content.replace(applyDeltaStr, applyThinkingDeltaStr);
  }

  if (!content.includes('case "thinking-delta":')) {
    const tokenDeltaCase = `case "token-delta":
          return {
            ...state,
            messages: applyDelta(state.messages, event.messageId, event.text),
          };`;
    const thinkingDeltaCase = `case "thinking-delta":
          return {
            ...state,
            messages: applyThinkingDelta(state.messages, event.messageId, event.text),
          };
        ` + tokenDeltaCase;
    content = content.replace(tokenDeltaCase, thinkingDeltaCase);
  }

  fs.writeFileSync(file, content, 'utf8');
  console.log('2. streamReducer.ts updated');
}

// 3. Update C:/lovable/src/services/stream/pollingAdapter.ts
{
  const file = 'C:/lovable/src/services/stream/pollingAdapter.ts';
  let content = fs.readFileSync(file, 'utf8');

  if (!content.includes('lastThinkingByMsg')) {
    content = content.replace(
      'const lastContentByMsg = new Map<string, string>();',
      'const lastContentByMsg = new Map<string, string>();\n    const lastThinkingByMsg = new Map<string, string>();'
    );
  }

  if (!content.includes('thinking-delta')) {
    const target = `if (message.pending) {
          if (message.content.length > prevText.length) {
            const delta = message.content.slice(prevText.length);
            lastContentByMsg.set(message.id, message.content);
            yield { type: "token-delta", messageId: message.id, text: delta };
          }
        }`;

    const replacement = `if (message.pending) {
          if (message.thinking && message.thinking.length > prevThinking.length) {
            const thinkingDelta = message.thinking.slice(prevThinking.length);
            lastThinkingByMsg.set(message.id, message.thinking);
            yield { type: "thinking-delta", messageId: message.id, text: thinkingDelta };
          }
          if (message.content.length > prevText.length) {
            const delta = message.content.slice(prevText.length);
            lastContentByMsg.set(message.id, message.content);
            yield { type: "token-delta", messageId: message.id, text: delta };
          }
        }`;

    content = content.replace(
      'const prevText = lastContentByMsg.get(message.id) ?? "";',
      'const prevText = lastContentByMsg.get(message.id) ?? "";\n        const prevThinking = lastThinkingByMsg.get(message.id) ?? "";'
    );
    content = content.replace(target, replacement);
  }

  fs.writeFileSync(file, content, 'utf8');
  console.log('3. pollingAdapter.ts updated');
}
