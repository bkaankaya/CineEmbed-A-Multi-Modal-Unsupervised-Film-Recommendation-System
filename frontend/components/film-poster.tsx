"use client";

import Image from "next/image";
import type { Film } from "@/lib/api";

export function FilmPoster({ film, size = "md" }: { film: Film; size?: "sm" | "md" | "lg" }) {
  const dims = size === "sm" ? "w-16 h-24" : size === "lg" ? "w-64 h-96" : "w-32 h-48";
  if (film.posterUrl) {
    return (
      <div className={`relative overflow-hidden rounded-md ${dims}`}>
        <Image
          src={film.posterUrl}
          alt={`${film.title} poster`}
          fill
          sizes="200px"
          className="object-cover"
        />
      </div>
    );
  }
  return (
    <div
      className={`${dims} rounded-md flex items-center justify-center text-white text-xs font-medium text-center px-2 [text-shadow:0_1px_2px_rgba(0,0,0,0.5)]`}
      style={{ background: film.posterColor }}
      role="img"
      aria-label={`${film.title} (no poster available)`}
    >
      {film.title}
    </div>
  );
}
