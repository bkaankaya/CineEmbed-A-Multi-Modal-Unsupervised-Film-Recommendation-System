"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  title?: string;
  error?: unknown;
  onRetry?: () => void;
}

export function ErrorFallback({ title = "Couldn't load this", error, onRetry }: Props) {
  const message = error instanceof Error ? error.message : String(error ?? "unknown");
  return (
    <div className="border border-border rounded-lg p-4 bg-card text-sm">
      <div className="flex items-center gap-2 text-amber-700 mb-2">
        <AlertTriangle className="w-4 h-4" aria-hidden="true" />
        <span className="font-medium">{title}</span>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded border border-border hover:bg-muted"
        >
          <RefreshCw className="w-3 h-3" aria-hidden="true" /> Retry
        </button>
      )}
      <details className="mt-2 text-xs text-muted-foreground">
        <summary className="cursor-pointer">Technical detail</summary>
        <code className="block mt-1 font-mono break-all">{message}</code>
      </details>
    </div>
  );
}
