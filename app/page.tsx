"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { SearchBar } from "@/components/search-bar"
import { SelectedFilmPanel } from "@/components/selected-film-panel"
import { SimilarFilmsPanel } from "@/components/similar-films-panel"
import { EmptyState } from "@/components/empty-state"
import { FILMS, SIMILAR_FILMS_MAP, type Film } from "@/lib/mock-data"

export default function Home() {
  const [activeNav, setActiveNav] = useState("search")
  const [query, setQuery] = useState("")
  const [selectedFilm, setSelectedFilm] = useState<Film | null>(null)

  const similarFilms = selectedFilm ? (SIMILAR_FILMS_MAP[selectedFilm.id] ?? []) : []

  function handleNavChange(nav: string) {
    setActiveNav(nav)
  }

  function handleFilmSelect(film: Film | null) {
    setSelectedFilm(film)
  }

  function handleSimilarFilmSelect(film: Film) {
    setSelectedFilm(film)
    setQuery(film.title)
  }

  return (
    <div className="flex min-h-screen" style={{ background: "#f8f9fb" }}>
      {/* Sidebar */}
      <Sidebar activeNav={activeNav} onNavChange={handleNavChange} />

      {/* Main content */}
      <main className="flex-1 flex flex-col min-h-screen" style={{ marginLeft: 220 }}>
        <div className="flex flex-col flex-1 px-6 pt-6 pb-6 gap-5 max-w-[1200px] w-full mx-auto">
          {/* Search bar */}
          <SearchBar
            films={FILMS}
            onSelect={handleFilmSelect}
            value={query}
            onChange={setQuery}
          />

          {/* Content area */}
          {selectedFilm ? (
            <div className="flex gap-5 flex-1 items-start">
              {/* Selected Film panel — ~58% */}
              <div className="flex-[58] min-w-0">
                <SelectedFilmPanel film={selectedFilm} />
              </div>

              {/* Similar Films panel — ~42% */}
              <div className="flex-[42] min-w-0 self-stretch">
                <SimilarFilmsPanel
                  films={similarFilms}
                  onSelect={handleSimilarFilmSelect}
                />
              </div>
            </div>
          ) : (
            <EmptyState />
          )}
        </div>

        {/* Footer */}
        <footer className="py-4 text-center border-t" style={{ borderColor: "#e5e4ec" }}>
          <p className="text-xs" style={{ color: "#9ca3af" }}>
            CineEmbed — Unsupervised Movie Discovery
          </p>
        </footer>
      </main>
    </div>
  )
}
