import { Film } from "lucide-react"

export function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 select-none">
      <div
        className="w-20 h-20 rounded-2xl flex items-center justify-center"
        style={{ background: "#f1f0f5", border: "1px solid #e5e4ec" }}
      >
        <Film className="w-9 h-9" style={{ color: "#d1d5db" }} />
      </div>
      <p
        className="text-sm text-center max-w-[280px] text-pretty leading-relaxed"
        style={{ color: "#9ca3af" }}
      >
        Search for a film to explore its latent neighbours.
      </p>
    </div>
  )
}
