"use client"

import { useState } from "react"
import type { Film } from "@/lib/mock-data"

interface SimilarFilmsPanelProps {
  films: Film[]
  onSelect: (film: Film) => void
}

function GenreTag({ label }: { label: string }) {
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-xs font-medium"
      style={{ background: "#f1f0f5", color: "#6b7280", border: "1px solid #e5e4ec" }}
    >
      {label}
    </span>
  )
}

export function SimilarFilmsPanel({ films, onSelect }: SimilarFilmsPanelProps) {
  const [hoveredId, setHoveredId] = useState<number | null>(null)

  return (
    <div
      className="rounded-xl border flex flex-col overflow-hidden h-full shadow-sm"
      style={{ background: "#ffffff", borderColor: "#e5e4ec" }}
    >
      {/* Header */}
      <div className="px-5 pt-4 pb-3 border-b" style={{ borderColor: "#e5e4ec" }}>
        <h2 className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#6e56cf" }}>
          Similar Films
        </h2>
      </div>

      {/* Film list */}
      <div className="flex-1 overflow-y-auto">
        {films.map((film, index) => (
          <button
            key={film.id}
            className="w-full text-left px-5 py-3.5 border-b transition-colors flex items-center gap-4"
            style={{
              borderColor: "#f1f0f5",
              background: hoveredId === film.id ? "rgba(110,86,207,0.06)" : "transparent",
            }}
            onMouseEnter={() => setHoveredId(film.id)}
            onMouseLeave={() => setHoveredId(null)}
            onClick={() => onSelect(film)}
          >
            {/* Row number */}
            <span
              className="text-sm font-mono w-6 flex-shrink-0 text-right"
              style={{ color: "#d1d5db" }}
            >
              #{index + 1}
            </span>

            {/* Title + year */}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold truncate" style={{ color: "#1a1a2e" }}>{film.title}</p>
              <p className="text-xs mt-0.5" style={{ color: "#9ca3af" }}>
                {film.year} · {film.director}
              </p>
            </div>

            {/* Genre tags */}
            <div className="flex-shrink-0 flex flex-wrap gap-1 justify-end max-w-[120px]">
              {film.genres.slice(0, 2).map((g) => (
                <GenreTag key={g} label={g} />
              ))}
            </div>
          </button>
        ))}

        {films.length === 0 && (
          <div className="px-5 py-10 text-center text-sm" style={{ color: "#d1d5db" }}>
            No similar films found.
          </div>
        )}
      </div>
    </div>
  )
}
