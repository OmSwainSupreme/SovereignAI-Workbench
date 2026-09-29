import { useState, useEffect, useMemo, useCallback } from "react";
import { tasksService } from "@/services/api";
import type { TaskSummary } from "@/services/types";

export interface Conversation {
  id: string;
  title: string;
  preview?: string;
  updatedAt: string;
  status?: string;
}

const STORAGE_KEY = "sovereign_ai_conversations";

function getGroupKey(dateStr: string): "Today" | "Yesterday" | "Previous 7 Days" | "Older" {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0 && date.getDate() === now.getDate()) {
    return "Today";
  }
  if (diffDays <= 1) {
    return "Yesterday";
  }
  if (diffDays <= 7) {
    return "Previous 7 Days";
  }
  return "Older";
}

export function useConversations(activeId?: string) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isInitialized, setIsInitialized] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
          setConversations(JSON.parse(stored));
        }
      } catch (err) {
        console.warn("Failed to load conversations from localStorage:", err);
      }
      setIsInitialized(true);
    }
  }, []);

  // Persist to localStorage whenever conversations change, but only after initialization
  useEffect(() => {
    if (!isInitialized) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (err) {
      console.warn("Failed to persist conversations:", err);
    }
  }, [conversations, isInitialized]);

  // Load and merge with backend tasks
  useEffect(() => {
    let mounted = true;
    tasksService
      .list()
      .then((backendTasks) => {
        if (!mounted) return;
        setConversations((prev) => {
          const map = new Map<string, Conversation>();
          // Existing local conversations take precedence for custom titles
          for (const c of prev) {
            map.set(c.id, c);
          }
          // Merge with backend tasks
          for (const t of backendTasks) {
            if (!map.has(t.id)) {
              map.set(t.id, {
                id: t.id,
                title: t.title || "Conversation",
                preview: t.preview,
                updatedAt: t.updatedAt || new Date().toISOString(),
                status: t.status,
              });
            } else {
              const existing = map.get(t.id)!;
              map.set(t.id, {
                ...existing,
                preview: t.preview || existing.preview,
                status: t.status || existing.status,
              });
            }
          }
          return Array.from(map.values()).sort(
            (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
          );
        });
      })
      .catch((err) => {
        console.warn("Could not fetch backend tasks for history:", err);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const addConversation = useCallback((conv: Conversation) => {
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== conv.id);
      return [conv, ...filtered];
    });
  }, []);

  const renameConversation = useCallback((id: string, newTitle: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
    );
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    void tasksService.cancel(id).catch(() => {});
  }, []);

  // Filtered by search
  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return conversations;
    return conversations.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        (c.preview && c.preview.toLowerCase().includes(q))
    );
  }, [conversations, searchQuery]);

  // Grouped by time period
  const grouped = useMemo(() => {
    const groups: Record<"Today" | "Yesterday" | "Previous 7 Days" | "Older", Conversation[]> = {
      Today: [],
      Yesterday: [],
      "Previous 7 Days": [],
      Older: [],
    };

    for (const c of filtered) {
      const key = getGroupKey(c.updatedAt);
      groups[key].push(c);
    }

    return groups;
  }, [filtered]);

  return {
    conversations,
    filtered,
    grouped,
    searchQuery,
    setSearchQuery,
    addConversation,
    renameConversation,
    deleteConversation,
  };
}
