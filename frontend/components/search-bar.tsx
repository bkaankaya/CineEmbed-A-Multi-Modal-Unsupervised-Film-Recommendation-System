"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search as SearchIcon } from "lucide-react";
import { api, type BackboneId } from "@/lib/api";

interface Props {
  backbone: BackboneId;
  onSelectFilm: (id: number) => void;
}

export function SearchBar({ backbone, onSelectFilm }: Props) {
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const { data: hits = [] } = useQuery({
    queryKey: ["search", debouncedQ, backbone],
    queryFn: ({ signal }) => api.searchFilms(debouncedQ, backbone, 10, { signal }),
    enabled: debouncedQ.length >= 1,
  });

  return (
    <div className="relative">
      <div className="relative">
        <SearchIcon
          aria-hidden="true"
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-4 h-4"
        />
        <input
          type="search"
          role="combobox"
          aria-expanded={hits.length > 0}
          aria-autocomplete="list"
          aria-controls="search-results"
          placeholder="Search 329,044 films..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="w-full pl-9 pr-3 py-2 border border-border rounded-md bg-card focus:outline-none focus:ring-2 focus:ring-purple-300"
        />
      </div>
      {hits.length > 0 && (
        <ul
          id="search-results"
          role="listbox"
          className="absolute mt-1 w-full bg-card border border-border rounded-md shadow-lg z-10 max-h-80 overflow-y-auto"
        >
          {hits.map((f) => (
            <li
              key={f.id}
              role="option"
              aria-selected="false"
              tabIndex={0}
              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
              onClick={() => {
                onSelectFilm(f.id);
                setQ("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectFilm(f.id);
                  setQ("");
                }
              }}
            >
              <div className="font-medium">{f.title}</div>
              <div className="text-xs text-muted-foreground">
                {f.year ?? "—"} · {f.director}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
