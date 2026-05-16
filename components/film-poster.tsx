import type { Film } from "@/lib/mock-data"

interface FilmPosterProps {
  film: Film
  className?: string
}

export function FilmPoster({ film, className = "" }: FilmPosterProps) {
  const initials = film.title
    .split(" ")
    .slice(0, 2)
    .map((w) => w[0])
    .join("")

  return (
    <div
      className={`flex items-center justify-center rounded-lg flex-shrink-0 ${className}`}
      style={{
        background: film.posterColor,
        width: 140,
        minWidth: 140,
        minHeight: 200,
        border: "1px solid #21262d",
      }}
    >
      <div className="text-center select-none">
        <div
          className="text-3xl font-bold mb-1"
          style={{ color: "rgba(255,255,255,0.7)" }}
        >
          {initials}
        </div>
        <div
          className="text-xs font-medium px-2 text-center leading-tight"
          style={{ color: "rgba(255,255,255,0.35)" }}
        >
          {film.year}
        </div>
      </div>
    </div>
  )
}
