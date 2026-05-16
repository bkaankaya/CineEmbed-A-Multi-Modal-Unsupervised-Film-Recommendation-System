"use client"

import { Search, Film } from "lucide-react"

interface SidebarProps {
  activeNav: string
  onNavChange: (nav: string) => void
}

const NAV_ITEMS = [
  { id: "search", label: "Search", icon: Search },
]

export function Sidebar({ activeNav, onNavChange }: SidebarProps) {
  return (
    <aside
      className="fixed top-0 left-0 h-full w-[220px] flex flex-col border-r z-10"
      style={{ background: "#ffffff", borderColor: "#e5e4ec" }}
    >
      {/* Logo */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-2">
          <Film className="w-5 h-5" style={{ color: "#6e56cf" }} />
          <span className="text-lg font-bold tracking-tight">
            <span style={{ color: "#1a1a2e" }}>Cine</span>
            <span style={{ color: "#6e56cf" }}>Embed</span>
          </span>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-4 mb-4" style={{ height: "1px", background: "#e5e4ec" }} />

      {/* Navigation */}
      <nav className="flex-1 px-3 flex flex-col gap-1">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeNav === id
          return (
            <button
              key={id}
              onClick={() => onNavChange(id)}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left w-full"
              style={{
                background: isActive ? "rgba(110, 86, 207, 0.10)" : "transparent",
                color: isActive ? "#6e56cf" : "#6b7280",
              }}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </button>
          )
        })}
      </nav>

      {/* Version footer */}
      <div className="px-5 py-5">
        <div
          className="flex items-center gap-2 text-xs"
          style={{ color: "#9ca3af" }}
        >
          <Film className="w-3.5 h-3.5" />
          <span>CineEmbed v1.0 / Student Demo</span>
        </div>
      </div>
    </aside>
  )
}
