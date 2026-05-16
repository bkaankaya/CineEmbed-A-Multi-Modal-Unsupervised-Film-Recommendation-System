"use client"

import { useState, useRef, useEffect } from "react"
import { Search, X } from "lucide-react"
import type { Film } from "@/lib/mock-data"

interface SearchBarProps {
  films: Film[]
  onSelect: (film: Film | null) => void
  value: string
  onChange: (value: string) => void
}

export function SearchBar({ films, onSelect, value, onChange }: SearchBarProps) {
  const [showDropdown, setShowDropdown] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const filtered = value.trim()
    ? films.filter((f) => f.title.toLowerCase().includes(value.toLowerCase()))
    : []

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  function handleSelect(film: Film) {
    onChange(film.title)
    onSelect(film)
    setShowDropdown(false)
  }

  function handleClear() {
    onChange("")
    onSelect(null)
    setShowDropdown(false)
  }

  return (
    <div className="relative" ref={ref}>
      <div
        className="flex items-center gap-3 px-4 py-3 rounded-xl border shadow-sm"
        style={{
          background: "#ffffff",
          borderColor: "#e5e4ec",
        }}
      >
        <Search className="w-4 h-4 flex-shrink-0" style={{ color: "#9ca3af" }} />
        <input
          type="text"
          placeholder="Search for a film…"
          className="flex-1 bg-transparent text-sm outline-none"
          style={{ color: "#1a1a2e" }}
          value={value}
          onChange={(e) => {
            onChange(e.target.value)
            setShowDropdown(true)
            if (!e.target.value) onSelect(null)
          }}
          onFocus={() => value && setShowDropdown(true)}
        />
        {value && (
          <button
            onClick={handleClear}
            className="flex-shrink-0 transition-colors"
            style={{ color: "#9ca3af" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#1a1a2e")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#9ca3af")}
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Dropdown suggestions */}
      {showDropdown && filtered.length > 0 && (
        <div
          className="absolute top-full mt-1 w-full rounded-xl border z-50 overflow-hidden shadow-lg"
          style={{ background: "#ffffff", borderColor: "#e5e4ec" }}
        >
          {filtered.map((film) => (
            <button
              key={film.id}
              className="w-full text-left px-4 py-3 text-sm border-b transition-colors flex items-center gap-3"
              style={{ borderColor: "#f1f0f5" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(110,86,207,0.06)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              onClick={() => handleSelect(film)}
            >
              <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "#6e56cf" }} />
              <span className="font-medium" style={{ color: "#1a1a2e" }}>{film.title}</span>
              <span className="ml-auto text-xs" style={{ color: "#9ca3af" }}>
                {film.year}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
