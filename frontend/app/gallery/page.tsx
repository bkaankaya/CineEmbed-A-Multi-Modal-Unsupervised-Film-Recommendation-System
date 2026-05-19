import { readFile } from "node:fs/promises";
import path from "node:path";
import { Sidebar } from "@/components/sidebar";
import { Footer } from "@/components/footer";
import { FilmPoster } from "@/components/film-poster";
import { GallerySchema } from "@/lib/api-types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const LOCAL_GALLERY = path.join(process.cwd(), "..", "artifacts", "inference", "gallery.json");

async function fetchGallery() {
  try {
    const res = await fetch(`${BASE}/api/gallery`, { next: { revalidate: 3600 } });
    if (!res.ok) throw new Error(`Gallery API returned ${res.status}`);
    const json = await res.json();
    return GallerySchema.parse(json);
  } catch {
    const raw = await readFile(LOCAL_GALLERY, "utf8");
    return GallerySchema.parse(JSON.parse(raw));
  }
}

export default async function GalleryPage() {
  const gallery = await fetchGallery();
  const backbones = ["ae_z32", "ae_z64", "ae_z128"] as const;

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <h1 className="text-2xl font-semibold tracking-tight mb-6">Eyeball gallery</h1>
        <p className="text-sm text-muted-foreground mb-6">
          Five well-known queries × three backbones. The same query produces
          visibly different top-5 neighbours per backbone — the strongest
          demonstration of the project&rsquo;s z-sweep finding (see{" "}
          <a className="text-primary underline underline-offset-2 hover:text-primary/80" href="/about">About</a>).
        </p>
        <div className="space-y-12">
          {gallery.queries.map((q, i) => {
            const queryFilm = gallery.matrix[q]["ae_z32"].query;
            return (
              <section key={q}>
                <div className="flex items-center gap-4 mb-4">
                  <span
                    className="text-4xl font-light text-muted-foreground/30 tabular-nums leading-none w-12 shrink-0 select-none"
                    aria-hidden="true"
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <FilmPoster film={queryFilm} size="sm" />
                  <div>
                    <h2 className="text-lg font-medium mt-0 mb-1 text-foreground">{q}</h2>
                    <p className="text-xs text-muted-foreground tabular-nums">
                      {queryFilm.title} ({queryFilm.year ?? "—"})
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {backbones.map((bb) => {
                    const cell = gallery.matrix[q][bb];
                    const queryPrimary = cell.query.genres[0];
                    const top5 = cell.neighbors.slice(0, 5);
                    const genreAt5 = queryPrimary
                      ? top5.filter((n) => n.genres[0] === queryPrimary).length / 5
                      : null;
                    return (
                      <div
                        key={bb}
                        className="border border-border rounded-lg p-5 bg-card shadow-sm hover:-translate-y-0.5 hover:shadow-md transition-all duration-150"
                      >
                        <div className="flex justify-between items-baseline mb-3">
                          <p className="text-xs font-medium text-purple-700">{bb}</p>
                          {genreAt5 !== null && (
                            <p className="text-[10px] text-muted-foreground tabular-nums">
                              genre@5={genreAt5.toFixed(2)}
                            </p>
                          )}
                        </div>
                        <div className="flex gap-1.5 mb-3">
                          {top5.map((n) => (
                            <FilmPoster key={n.id} film={n} size="sm" />
                          ))}
                        </div>
                        <ol className="text-xs space-y-1">
                          {top5.map((n, i) => (
                            <li key={n.id} className="flex justify-between gap-2">
                              <span className="truncate">#{i + 1} {n.title}</span>
                              <span className="text-muted-foreground tabular-nums shrink-0">{n.cosine.toFixed(3)}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
        <Footer />
      </main>
    </div>
  );
}
