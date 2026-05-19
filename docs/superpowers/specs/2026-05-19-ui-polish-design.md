# UI Polish — Demo-grade Design Spec

**Date:** 2026-05-19  
**Demo deadline:** 2026-05-20  
**Branch:** main
**Scope:** 20 LOCKED items across 5 pages + 9 components  
**Budget:** ~10h work + verification  
**Approval origin:** brainstorming session 2026-05-19 (CineEmbed UI polish scope)

---

## 1. Context

CineEmbed frontend (Next.js 16 + Tailwind v4 + shadcn/ui) is functionally complete (T3 demo built, TMDb v4 live, 115 pytest pass, tsc clean). User-perceived issues remain: token-inconsistency between custom hex (`bg-[#f8f9fb]`, `border-[#e5e4ec]`) and shadcn tokens, no typography scale, lifeless empty state, unstyled About page prose, dense gallery layout, no focus-visible rings, inconsistent loading states, no error handling, missing trust signals.

This spec defines a surgical refactor that delivers demo-grade polish without redesign — 20 atomic items, ~10h total, no new dependencies, 1 new file, 1 new prop.

## 2. Goals & non-goals

### Goals
- All pages use shadcn semantic tokens uniformly (no orphan hex in markup)
- Typography scale uniform across 5 pages
- All interactive elements have visible keyboard focus
- Error and loading states consistent
- Demo-storytelling cues that surface methodology (cosine semantics, backbone metrics, dataset scale)
- A11y quickfire wins (focus-visible, sr-only, tabular-nums for numeric alignment)

### Non-goals (LOCKED OUT — see §10)
Dark mode, mobile responsive, animation library, logo redesign, plot diagrams, top progress bar, custom Radix tooltips, ⌘K shortcut, API health pill, skip-to-content link, X9-enriched footer beyond stated, "Open cluster →" CTA.

## 3. Architecture

| Surface | Decision |
|---|---|
| Token strategy | Shadcn semantic Tailwind utilities backed by `globals.css :root` variables. Replace inline `style={{}}` and `bg-[#...]` arbitrary values with `bg-background`, `bg-card`, `border-border`, `text-foreground`, `text-muted-foreground`, `text-primary`. |
| Typography plugin | NOT installed. Replace `<section className="prose prose-sm">` in About with explicit Tailwind classes per element. Rationale: avoid dependency for one page. |
| New files | `frontend/components/error-fallback.tsx` (only). |
| New props | `EmptyState.onPickExample(id: number) => void` (only). |
| `globals.css` additions | 2 new `@layer base` rules: `*:focus-visible` ring, `.tabular-nums` utility. |
| Dependencies | NONE added. Lucide icons already in deps; use existing. |
| Component boundaries | All existing component files preserved. No moves, no splits. |
| File deletions | NONE. |

## 4. Design system invariants (LOCKED)

All phases must respect these — any phase that contradicts is invalid and must be revised:

| Invariant | Value |
|---|---|
| Primary accent | `--primary: #6e56cf` (purple, already in `globals.css`) |
| Background | `--background: #f8f9fb` |
| Card | `--card: #ffffff` |
| Border | `--border: #e5e4ec` |
| Muted text | `--muted-foreground: #6b7280` |
| Foreground | `--foreground: #1a1a2e` |
| Ring color | `--ring: #6e56cf` (= primary) |
| Heading scale | h1 `text-2xl font-semibold tracking-tight mb-6`; h2 `text-lg font-medium mt-8 mb-3`; h3 `text-base font-medium mt-4 mb-2`. **Exception:** card-internal h3 used for emphasis (e.g., About finding callout cards) may use `font-semibold`. |
| Section padding | Pages use `p-8`; cards use `p-4` or `p-5` per role |
| Numeric values | `tabular-nums` class applied to all cosine/metric/year/rating/votes/duration tokens |
| Focus ring | `ring-2 ring-ring ring-offset-2 outline-none` global on `*:focus-visible` |
| Hover-lift | `hover:-translate-y-0.5 hover:shadow-md transition-all duration-150` on clickable cards |
| Active sidebar route | `border-l-2 border-primary font-semibold bg-purple-50 text-purple-800` |

## 5. Per-phase contracts

Each phase has: **Files**, **Change**, **Acceptance criteria** (testable), **Risk**.

### Phase A — H1 Token migration
- **Files:** `frontend/app/{page,about/page,cluster/page,cluster/[k]/page,gallery/page}.tsx`, `frontend/components/{sidebar,search-bar,backbone-switcher,selected-film-panel,similar-films-panel,cluster-card,cosine-heatmap,empty-state,film-poster}.tsx`
- **Change:** Replace literal-hex Tailwind values with shadcn semantic tokens:
  - `bg-[#f8f9fb]` → `bg-background`
  - `border-[#e5e4ec]` → `border-border`
  - `bg-white` (page/card surfaces) → `bg-card`
  - `text-gray-500` → `text-muted-foreground` (only on muted text, not on body)
  - `style={{ background: "#f8f9fb" }}` → `className="bg-background"` (replace inline style)
  - `style={{ borderColor: "#e5e4ec" }}` → `className="border-border"`
  - `style={{ color: "#9ca3af" }}` → `className="text-muted-foreground"`
  - DO NOT touch: `purple-*`, `green-*`, `blue-*`, `slate-*`, `gray-100` (skeleton bg), `gray-400` (placeholder icon — borderline ok)
- **Acceptance:** `grep -rn 'bg-\[#\|border-\[#\|style={{ background\|style={{ borderColor\|style={{ color' frontend/app frontend/components` returns zero hits in non-test files. tsc clean. Visual smoke: no white-on-white regressions.
- **Risk:** `bg-white` is used in places where shadcn `bg-card` IS `#ffffff` (no visual diff). `text-gray-500` swap on body text changes color from `#6b7280` to identical `--muted-foreground` `#6b7280`. Visual diff = zero. Safe.

### Phase B — H2 Typography scale
- **Files:** `frontend/app/globals.css`, `frontend/app/{page,about/page,cluster/page,cluster/[k]/page,gallery/page}.tsx`
- **Change:**
  1. globals.css: NO change to `@layer base` for headings (Tailwind utilities applied per-element).
  2. Each page h1: `text-2xl font-semibold tracking-tight mb-6` (uniform). Drop variant `mb-2`, `mb-4`.
  3. Page `<main>` padding: `p-8` (uniform across all). Drop variants `px-6 pt-6 pb-6`.
  4. Inter-section gap: `space-y-8` where multiple sections stack.
- **Acceptance:** `grep -nE 'h1.*className' frontend/app/**/page.tsx | grep -v 'text-2xl font-semibold tracking-tight mb-6'` → empty. tsc clean. Visual: page titles render uniform size/spacing.
- **Risk:** Home page uses `<main>` with custom flex layout, not vanilla `p-8`. Preserve flex/grid structure; only adjust child padding.

### Phase C — H3 EmptyState rewrite + onPickExample prop
- **Files:** `frontend/components/empty-state.tsx`, `frontend/app/page.tsx`
- **Change:**
  1. EmptyState gains props: `onPickExample(id: number): void`.
  2. Layout: centered, `py-16`. Lucide `Search` icon (`w-12 h-12 text-muted-foreground/40`) above title.
  3. Title: `text-xl font-semibold mb-2 text-foreground`. Sub: `text-sm text-muted-foreground mb-6`.
  4. 3 example chips clickable: Inception (id 27205), Spirited Away (id 129), Pulp Fiction (id 680). Each: `<button onClick={() => onPickExample(id)} className="text-xs px-3 py-1.5 rounded-full border border-border bg-card hover:bg-purple-50 hover:border-purple-300 transition-colors">{title}</button>`.
  5. app/page.tsx wires `onPickExample={(id) => setFilm(id)}`.
- **Acceptance:** Click each of the 3 chips → URL updates `?film={id}` with no 404 from API. tsc clean. Codex pre-verified all three IDs (27205/129/680) exist in `artifacts/inference/films_master.parquet`.
- **Risk:** None remaining — IDs verified.

### Phase D — H4 + X6 + C7 About suite
- **Files:** `frontend/app/about/page.tsx`
- **Change:**
  1. **H4 prose styling:** Remove `prose prose-sm`. Apply explicit classes (aligned to §4 invariant heading scale):
     - `h2`: `text-lg font-medium mt-8 mb-3 text-foreground`
     - `h3`: `text-base font-medium mt-4 mb-2 text-foreground`
     - `p`: `text-sm text-foreground leading-relaxed mb-4`
     - `code`: `bg-muted px-1.5 py-0.5 rounded text-xs font-mono`
     - `a`: `text-primary underline underline-offset-2 hover:text-primary/80`
  2. **X6 finding callout cards:** wrap each of the 2 findings (NMI≠retrieval, z=32 sweet spot) in:
     ```
     <article className="bg-card border border-border rounded-lg p-5 mb-4">
       <p className="text-[10px] uppercase tracking-widest text-primary font-semibold mb-2">Finding {N} — Methodology</p>
       <h3 className="text-base font-semibold mb-2">{title}</h3>
       <p className="text-sm text-foreground leading-relaxed">{body}</p>
     </article>
     ```
  3. **C7 pipeline row:** After intro paragraph and before "Two methodological findings", insert:
     ```
     <div className="flex flex-wrap items-center gap-2 my-6 text-xs">
       <span className="px-3 py-1.5 rounded bg-card border border-border">Movies</span>
       <span className="text-muted-foreground">→</span>
       <span className="px-3 py-1.5 rounded bg-card border border-border">Embeddings</span>
       <span className="text-muted-foreground">→</span>
       <span className="px-3 py-1.5 rounded bg-card border border-border">Cosine Search</span>
       <span className="text-muted-foreground">→</span>
       <span className="px-3 py-1.5 rounded bg-card border border-border">Recommendations</span>
     </div>
     ```
- **Acceptance:** `grep -n "prose" frontend/app/about/page.tsx` returns no matches. About page in browser: callout cards visible, pipeline row visible, prose-style elements styled.
- **Risk:** Tailwind class names long → 250-line file fine without further splitting (no over-engineering).

### Phase E — H5 Gallery polish
- **Files:** `frontend/app/gallery/page.tsx`
- **Change:**
  1. Section wrapper: `space-y-12` (was `space-y-8`).
  2. Per `<section>`: add query film poster strip at top using `FilmPoster` size="sm" centered, with title beneath. Pulls from `gallery.matrix[q]["ae_z32"].query`.
  3. Per backbone cell: card class `bg-card border border-border rounded-lg p-5` (was p-3). Add neighbor poster row above text list: `<div className="flex gap-2 mb-3">` 5× `FilmPoster size="sm"`.
  4. Each cell shows `genre@5={value}` next to backbone label. **Computation (explicit):** `const queryPrimary = cell.query.genres[0]; const value = queryPrimary ? cell.neighbors.slice(0,5).filter(n => n.genres[0] === queryPrimary).length / 5 : 0;` Render as `genre@5={value.toFixed(2)}` only if `queryPrimary` exists; otherwise omit.
- **Acceptance:** Gallery renders 5 sections × 3 cards each, each card shows poster row + text list, genre@5 numeric value visible. tsc clean. Spot-check Inception (primary genre=Action): manually count neighbors sharing Action in top-5 / 5; rendered value must match.
- **Risk:** Neighbor poster images per cell = 5 cards × 5 thumbs × 5 sections × 3 backbones = 225 `next/image` instances. Performance budget OK since posters are cached + revalidate hourly. genre@5 client-side compute over already-fetched neighbors — no API change.

### Phase F — M1 focus-visible scoped
- **Files:** `frontend/app/globals.css`
- **Change:** Append to `@layer base` — scoped to interactive elements to avoid Radix portal/composed-surface side effects:
  ```css
  :where(a, button, input, textarea, select, [role="button"], [role="radio"], [role="option"], [role="tab"], [tabindex="0"]):focus-visible {
    @apply ring-2 ring-ring ring-offset-2 outline-none;
  }
  ```
  Uses `:where()` so specificity stays zero — shadcn components keep their existing `focus-visible:*` utilities winning. `rounded-sm` removed; ring uses offset, intrinsic shape preserved.
- **Acceptance:** Tab through home + /cluster + /gallery + /about → focus ring visible on backbone-switcher, search-bar, example chips, sidebar nav, cluster cards. Mouse click → no ring. shadcn ui/ primitives unaffected.
- **Risk:** `:where()` zero-specificity means components with their own `focus-visible:` Tailwind utilities override us — intentional. Smoke test multiple pages.

### Phase G — M2 Sidebar accent + Film icon logo
- **Files:** `frontend/components/sidebar.tsx`
- **Change:**
  1. Logo line: prepend Lucide `Film` icon `w-5 h-5 text-primary`, keep "CineEmbed" text.
  2. Active state classes: `bg-purple-50 text-purple-800 border-l-2 border-primary pl-[6px] font-semibold` (note `pl-[6px]` compensates `border-l-2` width).
  3. Inactive remains `text-foreground/80 hover:bg-muted`.
- **Acceptance:** Active route shows left-bar + bold; logo has Film icon prefix.
- **Risk:** `border-l-2` adds 2px width — `pl-[6px]` (8px - 2px = 6px since original `px-2` was 8px) keeps alignment.

### Phase H — M3 Cluster card breathable + X4 hover-lift baseline
- **Files:** `frontend/components/cluster-card.tsx`
- **Change:**
  1. Posters grid: `grid grid-cols-2 gap-2` (was flex), each `FilmPoster size="md"` (was `sm`).
  2. Genre chips: `.slice(0, 2)` (was 3).
  3. Outer Link: add `hover:-translate-y-0.5 hover:shadow-md transition-all duration-150`.
- **Acceptance:** Cluster list shows 2x1 poster grid with md-sized posters, max 2 genre chips. Hover any card → 0.5 lift + shadow.
- **Risk:** md poster = `w-32 h-48` × 2 = 256w + gap. Card already grid-cols-3 at lg breakpoint. Width budget OK.

### Phase I — M4 Cosine legend + X3 info icon
- **Files:** `frontend/components/similar-films-panel.tsx`, `frontend/components/cosine-heatmap.tsx`
- **Change:**
  1. similar-films-panel: at top of `<aside>` after `<h3>`, add micro-legend:
     ```
     <div className="flex gap-2 mb-3 text-[10px] flex-wrap">
       <span className="bg-green-100 text-green-800 px-1.5 py-0.5 rounded">≥0.95 strong</span>
       <span className="bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">≥0.80 good</span>
       <span className="bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">other</span>
     </div>
     ```
  2. Similar h3: append Lucide `Info` icon w-3 h-3 inline, `<span title="Cosine: 1 = same direction in latent, 0 = orthogonal">` wrapper.
  3. cosine-heatmap stats label: append same Info icon with same `title=`.
- **Acceptance:** Legend pill row visible above first neighbor row. Info icon hover (mouse) shows browser tooltip.
- **Risk:** Lucide `Info` import bundle delta ~negligible. Native `title=` tooltip is plain UX but matches scope (no custom tooltip allowed).

### Phase J — M5 Loading skeleton consistency
- **Files:** `frontend/app/cluster/page.tsx`, `frontend/app/cluster/[k]/page.tsx`
- **Change:**
  1. cluster/page.tsx `{isLoading && <p>Loading clusters…</p>}` → skeleton grid 6 cards in the grid layout:
     ```
     <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
       {Array.from({ length: 6 }).map((_, i) => (
         <div key={i} className="border border-border rounded-lg p-4 bg-card animate-pulse h-48" />
       ))}
     </div>
     ```
  2. cluster/[k]/page.tsx loading: 12 film-thumb skeletons in the grid:
     ```
     <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
       {Array.from({ length: 12 }).map((_, i) => (
         <div key={i} className="w-full h-48 bg-muted rounded-md animate-pulse" />
       ))}
     </div>
     ```
- **Acceptance:** Switching to /cluster shows pulsing skeleton row before data; same for /cluster/[k].
- **Risk:** None.

### Phase K — M6 Error fallback component
- **Files:** `frontend/components/error-fallback.tsx` (new), `frontend/app/page.tsx`, `frontend/components/similar-films-panel.tsx`, `frontend/components/cosine-heatmap.tsx`, `frontend/app/cluster/page.tsx`, `frontend/app/cluster/[k]/page.tsx`
- **Query-ownership note:** SelectedFilmPanel does NOT own its query (the film query lives in `app/page.tsx`). Error rendering for selected-film must happen in `app/page.tsx` (replace the `<SelectedFilmPanel ... />` slot with `<ErrorFallback ... />` when `isError`). SimilarFilmsPanel, CosineHeatmap, ClustersPage, ClusterDetailPage own their queries and render ErrorFallback inline.
- **Change:**
  1. New component:
     ```tsx
     "use client";
     import { AlertTriangle, RefreshCw } from "lucide-react";

     interface Props { title?: string; error?: unknown; onRetry?: () => void; }

     export function ErrorFallback({ title = "Couldn't load this", error, onRetry }: Props) {
       const message = error instanceof Error ? error.message : String(error ?? "unknown");
       return (
         <div className="border border-border rounded-lg p-4 bg-card text-sm">
           <div className="flex items-center gap-2 text-amber-700 mb-2">
             <AlertTriangle className="w-4 h-4" aria-hidden="true" />
             <span className="font-medium">{title}</span>
           </div>
           {onRetry && (
             <button
               type="button"
               onClick={onRetry}
               className="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded border border-border hover:bg-muted"
             >
               <RefreshCw className="w-3 h-3" aria-hidden="true" /> Retry
             </button>
           )}
           <details className="mt-2 text-xs text-muted-foreground">
             <summary className="cursor-pointer">Technical detail</summary>
             <code className="block mt-1 font-mono break-all">{message}</code>
           </details>
         </div>
       );
     }
     ```
  2. Each consumer reads `isError`, `error`, `refetch` from useQuery and renders `<ErrorFallback title="..." error={error} onRetry={refetch} />` when error.
- **Acceptance:** `lsof -ti:8000 | xargs kill -9` (PID-targeted; do NOT use `pkill -f uvicorn` system-wide) → home page shows ErrorFallback for SelectedFilmPanel slot (rendered by `app/page.tsx`) + SimilarFilmsPanel + CosineHeatmap. Retry button refires query. After verification: `./scripts/dev-up.sh > /tmp/dev-up.log 2>&1 &` restores server.
- **Risk:** useQuery returns `refetch` only when query is enabled. For disabled queries (`enabled: false`), refetch is a no-op. Acceptable.

### Phase L — M7 Footer component pattern
- **Files:** `frontend/components/footer.tsx` (new), `frontend/app/page.tsx`, `frontend/app/about/page.tsx`, `frontend/app/cluster/page.tsx`, `frontend/app/cluster/[k]/page.tsx`, `frontend/app/gallery/page.tsx`
- **Rationale (revised from layout-level approach):** Putting footer at `<body>` level outside per-page `flex min-h-screen` wrapper creates double-height + detached feel (Codex flagged). Component pattern: each page renders `<Footer />` inside its main shell as the last child.
- **Change:**
  1. New `frontend/components/footer.tsx`:
     ```tsx
     export function Footer() {
       return (
         <footer className="border-t border-border bg-card py-4 text-center text-xs text-muted-foreground">
           CineEmbed · SENG 474 · TED University · 2026
         </footer>
       );
     }
     ```
     Provenance only — dataset/method facts moved to Phase O snapshot strip per separation-of-concerns.
  2. Each page imports and renders `<Footer />` inside its `<main>` as the final child. Home page already had a footer — replace with `<Footer />`.
- **Acceptance:** All 5 pages render footer at the bottom of their main content. No double scrollbars.
- **Risk:** Five identical imports — accept duplication (component pattern). No layout shell refactor.

### Phase M — X1 tabular-nums (Tailwind built-in)
- **Files:** components rendering numbers
- **Change:** Tailwind v4 ships `tabular-nums` as a built-in utility (`font-variant-numeric: tabular-nums`). No CSS addition required. Apply the class to:
  - Cosine badge text (similar-films-panel `{n.cosine.toFixed(2)}`)
  - CosineHeatmap top-10 `{t.cosine.toFixed(3)}` and stats line
  - Demo snapshot strip (already specified in Phase O)
  - SelectedFilmPanel year, rating, votes, duration
  - Cluster card size count
  - ClusterDetailPage size/total
  - Note: Phase Q backbone caption applies `tabular-nums` directly at creation time; do NOT add a checklist item for it here (forward reference would require Q to exist first).
- **Acceptance:** Numeric columns visibly aligned (monospaced digits). tsc clean. No globals.css changes from this phase.
- **Risk:** None — built-in utility.

### Phase N — X7 sr-only cosine match labels
- **Files:** `frontend/components/similar-films-panel.tsx`
- **Change:** Inside cosine badge `<span>`, append `<span className="sr-only">{` strong match`|` good match`|` weaker match`}</span>` based on threshold.
- **Acceptance:** Screen reader reads "0.92 good match" instead of "0.92".
- **Risk:** None.

### Phase O — C1 Demo snapshot strip
- **Files:** `frontend/app/page.tsx`
- **Change:** Below SearchBar, above content area (whether EmptyState or selected/similar grid), insert:
  ```
  <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground py-2 border-b border-border tabular-nums">
    <span>329,044 films</span><span aria-hidden="true">·</span>
    <span>3 backbones</span><span aria-hidden="true">·</span>
    <span>32-dim latent</span><span aria-hidden="true">·</span>
    <span>cosine over L2-normalized</span>
  </div>
  ```
- **Acceptance:** Visible on home page above content. tsc clean.
- **Risk:** None.

### Phase P — X4 Hover-lift consistency
- **Files:** `frontend/components/cluster-card.tsx` (already in Phase H), `frontend/components/similar-films-panel.tsx`, `frontend/app/gallery/page.tsx`
- **Change:** Add `hover:-translate-y-0.5 hover:shadow-md transition-all duration-150` to:
  - Similar films `<button>` items
  - Gallery backbone cells
  - Cluster card already covered in Phase H
- **Acceptance:** Hover any list item or card → subtle lift + shadow.
- **Risk:** Existing `hover:bg-purple-50` on similar items must keep working. transition-all OK.

### Phase Q — X10 Backbone caption
- **Files:** `frontend/components/backbone-switcher.tsx`
- **Change:** Below the switcher group, add:
  ```
  <p className="text-[10px] text-muted-foreground mt-1.5 text-right tabular-nums">
    Active: {label} · genre@5={genreAtFive.toFixed(3)} · gNMI={gnmi.toFixed(3)}
  </p>
  ```
  Reading the current backbone's metrics from `backbones` query result (already fetched).
- **Acceptance:** Caption visible below switcher; values update when user picks different backbone.
- **Risk:** Wrapping element needed since switcher is currently `<div role="radiogroup">`. Wrap that div + caption in a parent flex column.

## 6. Cross-cutting acceptance gates

After each phase:
1. `cd frontend && npx tsc --noEmit` — must succeed
2. `cd .. && python -m pytest tests/ -q` — must show 115 passed (sanity; backend untouched)
3. `curl -s http://localhost:3000 -o /dev/null -w "%{http_code}"` — must be 200
4. `git diff --stat` — only files declared in phase's Files list (no collateral)

## 7. Order & dependency rules

Token migration (A) must run first. Globals.css edits (B, F, M) batch into one commit-equivalent. About suite (D) is single-file; Gallery (E) is single-file; both can run after A but order is independent. Layout-level changes (L) commit last in their batch to avoid HMR thrash. Phases inside SDD execute strictly sequentially (no parallel implementer dispatch — overlapping files).

Recommended order:
1. A (token migration) — foundational
2. B + F + M (globals.css + typography across pages)
3. G + L (sidebar + footer-to-layout)
4. C + O (home page)
5. D (about page)
6. E (gallery page)
7. H + P (cluster card + hover-lift)
8. I + N (similar/cosine + sr-only)
9. J (loading skeletons)
10. K (error fallback)
11. Q (backbone caption)

## 8. Test plan

- **TypeScript:** `cd frontend && npx tsc --noEmit` after every commit (gate)
- **Python:** `python -m pytest tests/ -q` once per major batch (sanity, no backend touched)
- **Manual smoke:**
  - Home: search "Inception", click result, see selected+similar+heatmap
  - Click EmptyState example chip → URL updates
  - Switch backbone → metrics caption changes, panels reload
  - Sidebar: tab through nav, focus ring visible, active route accent
  - /cluster, /cluster/14, /gallery, /about all render without console errors
- **Error path:** `lsof -ti:8000 | xargs kill -9` (PID-targeted; never use `pkill -f uvicorn` — would affect unrelated uvicorn processes system-wide) then reload home → ErrorFallback renders. `./scripts/dev-up.sh > /tmp/dev-up.log 2>&1 &` restores.
- **Visual a11y:** `tab` through home → all interactive elements show ring. Cosine badges have sr-only labels.

## 9. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Gallery 225 `next/image` instances slow render | LOW | All TMDb cached, revalidate=3600, lazy by default |
| Tailwind v4 already registers `.tabular-nums` | LOW | Idempotent; drop my @layer rule if utility exists post-test |
| Phase D About changes break inline anchor link to /gallery | LOW | Keep `<a>` link intact, only restyle |
| Phase K error-fallback infinite retry loop | MEDIUM | Use `refetch` from useQuery (single call), no auto-retry past existing `retry: 1` config |
| Phase L footer below `flex min-h-screen` could create scrollbar | LOW | Footer outside the flex container, sits at body level — natural document flow |
| Phase O snapshot strip duplicate dataset info with Phase L footer | LOW | Different placement (top vs bottom), different density (4 facts vs 7), keep both |
| Hardcoded example film ids (Phase C) missing in parquet | MEDIUM | Pre-verify with `curl /api/films/27205` etc. Drop missing chips silently |

## 10. Out-of-scope (LOCKED OUT)

Any subagent or human who is tempted to add the following must REFUSE:
- Dark mode toggle (no `.dark` variant in `@custom-variant`)
- Mobile responsive `sm:` breakpoint additions
- Animation library (framer-motion, motion, etc.)
- Logo redesign beyond Phase G Film icon prefix
- About page U-curve diagram or any chart
- Top progress bar (NProgress or equiv)
- Custom Radix tooltips replacing `title=`
- ⌘K keyboard shortcut (X5)
- API health pill (X2)
- Skip-to-content link (X8)
- Footer enrichment beyond Phase L content (X9)
- "Open cluster →" CTA on cluster cards (C10)

If a subagent surfaces one of these as a fix, the orchestrator rejects with reason and proceeds.

## 11. Roll-back plan

Per-phase atomic commits with descriptive titles. To roll back any single phase: `git revert <sha> --no-edit`. To roll back all polish work: `git reset --hard 571fafc` (current pre-polish tip). User explicit consent required for hard-reset.

## 12. Final delivery

After Phase Q complete + report written + tests green:
- `git push origin main`
- `git tag ui-polish-overnight`
- `git push origin ui-polish-overnight`
- Final commit message references this spec by path.
