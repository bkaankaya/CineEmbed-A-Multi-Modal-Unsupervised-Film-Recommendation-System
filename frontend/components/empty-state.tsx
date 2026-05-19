"use client";

import { Search } from "lucide-react";

const EXAMPLES = [
  { id: 27205, title: "Inception" },
  { id: 129, title: "Spirited Away" },
  { id: 680, title: "Pulp Fiction" },
];

interface Props {
  onPickExample: (id: number) => void;
}

export function EmptyState({ onPickExample }: Props) {
  return (
    <div className="text-center py-16">
      <Search className="w-12 h-12 text-muted-foreground/40 mx-auto mb-4" aria-hidden="true" />
      <p className="text-xl font-semibold mb-2 text-foreground">Search 329,044 films</p>
      <p className="text-sm text-muted-foreground mb-6">
        Pick any film to see its nearest neighbors in the latent.
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        <span className="text-xs text-muted-foreground mr-1 self-center">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.id}
            type="button"
            onClick={() => onPickExample(ex.id)}
            className="text-xs px-3 py-1.5 rounded-full border border-border bg-card hover:bg-purple-50 hover:border-purple-300 transition-colors"
          >
            {ex.title}
          </button>
        ))}
      </div>
    </div>
  );
}
