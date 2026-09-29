import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "./CodeBlock";

interface MarkdownRendererProps {
  content: string;
}

export const MarkdownRenderer = React.memo(function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="text-sm leading-relaxed max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code(props) {
            const { children, className, node, ...rest } = props;
            const match = /language-(\w+)/.exec(className || "");
            const codeString = String(children || "");
            const isInline = !match && !codeString.includes("\n");

            if (isInline) {
              return (
                <code
                  className="rounded bg-surface/80 border border-border/60 px-1.5 py-0.5 font-mono text-[12px] text-primary"
                  {...rest}
                >
                  {children}
                </code>
              );
            }

            return (
              <CodeBlock
                language={match ? match[1] : "text"}
                value={codeString}
              />
            );
          },
          h1({ children }) {
            return <h1 className="mt-4 mb-2 text-xl font-bold tracking-tight text-foreground">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="mt-3.5 mb-2 text-lg font-semibold tracking-tight text-foreground">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="mt-3 mb-1.5 text-base font-semibold text-foreground">{children}</h3>;
          },
          p({ children }) {
            return <p className="my-2 leading-relaxed text-foreground/95">{children}</p>;
          },
          ul({ children }) {
            return <ul className="my-2 ml-5 list-disc space-y-1 text-foreground/95">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="my-2 ml-5 list-decimal space-y-1 text-foreground/95">{children}</ol>;
          },
          li({ children }) {
            return <li className="leading-relaxed">{children}</li>;
          },
          table({ children }) {
            return (
              <div className="my-3 overflow-x-auto rounded-lg border border-border/70 shadow-sm">
                <table className="w-full text-left text-xs">{children}</table>
              </div>
            );
          },
          th({ children }) {
            return <th className="border-b border-border/80 bg-surface/80 px-3 py-2 font-semibold text-foreground">{children}</th>;
          },
          td({ children }) {
            return <td className="border-b border-border/40 px-3 py-2 text-muted-foreground">{children}</td>;
          },
          blockquote({ children }) {
            return (
              <blockquote className="my-2.5 border-l-2 border-primary/60 pl-3 italic text-muted-foreground bg-surface/30 py-1 rounded-r">
                {children}
              </blockquote>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-primary underline hover:text-primary/80 transition-colors"
              >
                {children}
              </a>
            );
          },
          hr() {
            return <hr className="my-4 border-border/60" />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
