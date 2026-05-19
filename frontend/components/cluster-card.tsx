"use client";

import Link from "next/link";
import { FilmPoster } from "./film-poster";
import type { Cluster } from "@/lib/api";

export function ClusterCard({ cluster, backbone }: { cluster: Cluster; backbone: string }) {
  return (
    <Link
      href={`/cluster/${cluster.id}?backbone=${backbone}`}
      className="block border border-border rounded-lg p-4 bg-card shadow-sm hover:border-purple-300 hover:-translate-y-0.5 hover:shadow-md transition-all duration-150"
    >
      <h3 className="font-medium text-sm flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-primary/60 shrink-0" aria-hidden="true" />
        <span>{cluster.name}</span>
      </h3>
      <p className="text-xs text-muted-foreground mt-1 tabular-nums">{cluster.size.toLocaleString()} films</p>
      <div className="flex flex-wrap gap-1 mt-2">
        {cluster.topGenres.slice(0, 2).map((g) => (
          <span key={g.genre} className="text-[10px] px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded tabular-nums">
            {g.genre} {(g.pct * 100).toFixed(0)}%
          </span>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 mt-3">
        {cluster.previewFilms.slice(0, 2).map((f) => (
          <FilmPoster key={f.id} film={f} size="md" />
        ))}
      </div>
    </Link>
  );
}
