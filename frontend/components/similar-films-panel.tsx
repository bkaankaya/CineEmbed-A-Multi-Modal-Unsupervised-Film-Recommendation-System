"use client";

import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { ErrorFallback } from "./error-fallback";
import { api, type BackboneId, type Neighbor } from "@/lib/api";

interface Props {
  filmId: number;
  backbone: BackboneId;
  onSelectFilm: (id: number) => void;
}

function cosineColor(c: number): string {
  if (c >= 0.95) return "bg-green-100 text-green-800";
  if (c >= 0.8) return "bg-blue-100 text-blue-800";
  return "bg-slate-100 text-slate-700";
}

function cosineSrLabel(c: number): string {
  if (c >= 0.95) return " strong match";
  if (c >= 0.8) return " good match";
  return " weaker match";
}

export function SimilarFilmsPanel({ filmId, backbone, onSelectFilm }: Props) {
  const { data: neighbors = [], isLoading, isError, error, refetch } = useQuery({
    queryKey: ["similar", filmId, backbone],
    queryFn: ({ signal }) => api.getSimilar(filmId, backbone, 10, { signal }),
  });

  if (isError) {
    return <ErrorFallback title="Couldn't load similar films" error={error} onRetry={() => refetch()} />;
  }

  if (isLoading) {
    return (
      <div className="border border-border rounded-lg p-4 bg-card">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-10 rounded mb-2 shimmer" />
        ))}
      </div>
    );
  }

  if (neighbors.length === 0) {
    return <div className="border border-border rounded-lg p-4 bg-card text-sm text-muted-foreground">No similar films found.</div>;
  }

  return (
    <aside className="border border-border rounded-lg p-4 bg-card shadow-sm">
      <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1.5">
        <span>Similar films (backbone {backbone})</span>
        <span title="Cosine: 1 = same direction in latent, 0 = orthogonal" className="inline-flex">
          <Info className="w-3 h-3 text-muted-foreground" aria-label="cosine info" />
        </span>
      </h3>
      <div className="flex gap-2 mb-3 text-[10px] flex-wrap">
        <span className="bg-green-100 text-green-800 px-1.5 py-0.5 rounded tabular-nums">≥0.95 strong</span>
        <span className="bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded tabular-nums">≥0.80 good</span>
        <span className="bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">other</span>
      </div>
      <ol className="space-y-2">
        {neighbors.map((n: Neighbor, i: number) => (
          <li key={n.id}>
            <button
              type="button"
              onClick={() => onSelectFilm(n.id)}
              className="w-full flex items-start gap-3 px-2 py-2 rounded hover:bg-purple-50 text-left hover:-translate-y-0.5 hover:shadow-sm transition-all duration-150"
            >
              <span className="text-xs text-gray-400 w-6 tabular-nums">#{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{n.title}</div>
                <div className="text-xs text-muted-foreground truncate tabular-nums">
                  {n.year ?? "—"} · {n.director}
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {n.genres.slice(0, 2).map((g) => (
                    <span key={g} className="text-[10px] px-1.5 py-0.5 bg-gray-100 rounded">{g}</span>
                  ))}
                </div>
              </div>
              <span
                className={`text-xs px-1.5 py-0.5 rounded tabular-nums ${cosineColor(n.cosine)}`}
                aria-label={`cosine ${n.cosine.toFixed(3)}`}
              >
                {n.cosine.toFixed(2)}
                <span className="sr-only">{cosineSrLabel(n.cosine)}</span>
              </span>
            </button>
          </li>
        ))}
      </ol>
    </aside>
  );
}
