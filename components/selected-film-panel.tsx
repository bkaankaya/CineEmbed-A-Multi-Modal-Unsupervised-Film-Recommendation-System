"use client"

import { Star, Plus, Globe, Clock, Languages, User2 } from "lucide-react"
import type { Film } from "@/lib/mock-data"
import { FilmPoster } from "./film-poster"

interface SelectedFilmPanelProps {
  film: Film
}

function Tag({ label, purple = false }: { label: string; purple?: boolean }) {
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-xs font-medium"
      style={
        purple
          ? { background: "rgba(110,86,207,0.12)", color: "#6e56cf", border: "1px solid rgba(110,86,207,0.25)" }
          : { background: "#f1f0f5", color: "#4b5563", border: "1px solid #e5e4ec" }
      }
    >
      {label}
    </span>
  )
}

function MetaRow({
  label,
  icon: Icon,
  children,
}: {
  label: string
  icon?: React.ComponentType<{ className?: string }>
  children: React.ReactNode
}) {
  return (
    <div className="flex items-start gap-2 text-sm">
      {Icon && <Icon className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: "#9ca3af" }} />}
      <span className="min-w-[72px] flex-shrink-0 text-xs font-medium" style={{ color: "#9ca3af" }}>
        {label}
      </span>
      <div className="flex flex-wrap gap-1">{children}</div>
    </div>
  )
}

function AccentMetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-3 text-sm">
      <span className="min-w-[52px] flex-shrink-0 text-xs font-semibold" style={{ color: "#6e56cf" }}>
        {label}
      </span>
      <span className="text-sm" style={{ color: "#374151" }}>
        {value}
      </span>
    </div>
  )
}

export function SelectedFilmPanel({ film }: SelectedFilmPanelProps) {
  return (
    <div
      className="rounded-xl border flex flex-col overflow-hidden shadow-sm"
      style={{ background: "#ffffff", borderColor: "#e5e4ec" }}
    >
      {/* Card header label */}
      <div
        className="px-5 pt-4 pb-3 border-b"
        style={{ borderColor: "#e5e4ec" }}
      >
        <h2 className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#6e56cf" }}>
          Selected Film
        </h2>
      </div>

      <div className="p-5 flex flex-col gap-5">
        {/* Top strip: poster + metadata */}
        <div className="flex gap-4">
          <FilmPoster film={film} className="self-start" />

          {/* Right side metadata */}
          <div className="flex-1 min-w-0 flex flex-col gap-2.5">
            {/* Title + year + rating */}
            <div>
              <div className="flex items-start justify-between gap-2 flex-wrap">
                <h1 className="text-xl font-bold leading-tight text-balance" style={{ color: "#1a1a2e" }}>
                  {film.title}{" "}
                  <span className="text-base font-normal" style={{ color: "#9ca3af" }}>
                    ({film.year})
                  </span>
                </h1>
              </div>
              {/* Rating */}
              <div className="flex items-center gap-1.5 mt-1.5">
                <Star className="w-3.5 h-3.5 fill-yellow-400 text-yellow-400" />
                <span className="text-sm font-semibold" style={{ color: "#1a1a2e" }}>{film.rating}</span>
                <span className="text-xs" style={{ color: "#9ca3af" }}>
                  / 10 · {film.votes} votes
                </span>
              </div>
            </div>

            {/* Metadata rows */}
            <div className="flex flex-col gap-1.5">
              <MetaRow label="Genre">
                {film.genres.map((g) => (
                  <Tag key={g} label={g} />
                ))}
              </MetaRow>

              <MetaRow label="Country" icon={Globe}>
                <span className="text-sm" style={{ color: "#374151" }}>
                  {film.country}
                </span>
              </MetaRow>

              <MetaRow label="Duration" icon={Clock}>
                <span className="text-sm" style={{ color: "#374151" }}>
                  {film.duration}
                </span>
              </MetaRow>

              <MetaRow label="Language" icon={Languages}>
                <span className="text-sm" style={{ color: "#374151" }}>
                  {film.language}
                </span>
              </MetaRow>

              <MetaRow label="Director" icon={User2}>
                <span className="text-sm" style={{ color: "#374151" }}>
                  {film.director}
                </span>
              </MetaRow>

              {/* Cluster badge */}
              <div className="flex items-center gap-2 mt-0.5">
                <Tag label={`Cluster ${film.cluster}`} purple />
              </div>
            </div>

            {/* Watchlist button */}
            <div className="mt-auto pt-1">
              <button
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors border"
                style={{
                  borderColor: "#6e56cf",
                  color: "#6e56cf",
                  background: "transparent",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(110,86,207,0.10)"
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent"
                }}
              >
                <Plus className="w-3.5 h-3.5" />
                Watchlist
              </button>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: "1px", background: "#e5e4ec" }} />

        {/* Story / Overview */}
        <div>
          <p className="text-sm mb-2 font-semibold" style={{ color: "#6e56cf" }}>
            Story
          </p>
          <p className="text-sm leading-relaxed italic" style={{ color: "#4b5563" }}>
            {film.overview}
          </p>
        </div>

        {/* Divider */}
        <div style={{ height: "1px", background: "#e5e4ec" }} />

        {/* Bottom metadata */}
        <div className="flex flex-col gap-2.5">
          <div className="flex items-start gap-3">
            <span className="min-w-[52px] flex-shrink-0 text-xs font-semibold" style={{ color: "#6e56cf" }}>
              Style
            </span>
            <div className="flex flex-wrap gap-1">
              {film.style.map((s) => (
                <Tag key={s} label={s} />
              ))}
            </div>
          </div>

          <div className="flex items-start gap-3">
            <span className="min-w-[52px] flex-shrink-0 text-xs font-semibold" style={{ color: "#6e56cf" }}>
              Plot
            </span>
            <div className="flex flex-wrap gap-1">
              {film.plot.map((p) => (
                <Tag key={p} label={p} />
              ))}
            </div>
          </div>

          <AccentMetaRow label="Time" value={film.time} />
          <AccentMetaRow label="Place" value={film.place} />
        </div>
      </div>
    </div>
  )
}
