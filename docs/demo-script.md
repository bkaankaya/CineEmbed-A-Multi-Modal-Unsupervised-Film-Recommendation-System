# CineEmbed Demo Script

**Goal:** 5-minute walkthrough of the SENG 474 final demo.
**Audience:** SENG 474 grader (Spring 2026, TED University).
**Repo:** `bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System`, branch `main`.
**Setup:** see `README.md` § Demo — 5-line setup.

---

## Presentation flow (~5 minutes)

### 1. Opening (15 sec)
Open `localhost:3000`. Point to the title and the empty-state copy
("Search 329,044 films...").

### 2. Search demo (30 sec)
Type "inception". Note: live filter over 329k films, results appear
within 500ms. Click the top result.

### 3. Selected film panel (45 sec)
Walk through metadata (genres, year, cluster badge, director). Point
to the TMDb poster (or HSL gradient if keyless), the overview, the
style/plot keyword chips. Scroll to the cosine heatmap: "this is the
distribution of Inception's similarity to every one of 329,043 other
films — the right tail is the recommendation."

### 4. Similar panel (30 sec)
Show the top-5 with color-coded cosine badges. Click a similar film
(e.g., The Dark Knight). Point out URL state navigation:
`?film=155&backbone=ae_z32`.

### 5. Backbone switch — the methodological story (60 sec)
Click `ae_z128` in the page-header switcher. Similar panel re-renders
with different neighbors. Narrate: "Smaller models win on the demo
task because the information bottleneck forces concentration on
high-entropy modalities (text overview, director PCA). z=128 has
near-dead latent dimensions and gives more Marvel-flavored neighbors;
z=32 keeps the Nolan signature." Cite journal/12.

### 6. Cluster browser (45 sec)
Navigate to `/cluster`. Show the 21 named clusters, each with a
4-poster mosaic. Pick a memorable cluster (e.g., "Drama · 1990s" or
"Action · 2000s"). Click it → top-50 film grid.

### 7. Gallery (45 sec)
Navigate to `/gallery`. Five queries × three backbones precomputed
side-by-side. Spend 20 sec on "Inception → ae_z32 vs ae_z128" to
narrate the z-sweep finding visually.

### 8. About page (30 sec)
Navigate to `/about`. Brief stop — the grader sees the methodology
explained in writing.

### 9. Q&A buffer
End at the home page on Inception so questions can quickly demonstrate
specific points.

---

## Known-good query list

| Primary | Secondary backups |
|---|---|
| **Inception (27205)** | The Dark Knight (155), Interstellar (157336) |
| **Spirited Away (129)** | My Neighbor Totoro (8392), Princess Mononoke (128) |
| **Shawshank Redemption (278)** | The Green Mile (497) |
| **Pulp Fiction (680)** | Reservoir Dogs (500), Kill Bill Vol. 1 (24) |
| **Toy Story (862)** | WALL·E (10681), Finding Nemo (12) |

Substitute from the same row's backup list if a primary query
surfaces unexpected results during a dry run.

---

## Failure modes and recovery

- **TMDb key missing or rate-limited**: posters fall back to HSL
  gradient cards. App still functional. Do NOT panic — narrate as
  "the demo also works without TMDb."
- **Backend offline**: restart from CLI; frontend reconnects.
- **Slow first render**: Next.js dev server first-compile can take
  20-40s for Next 16 + shadcn. Pre-warm before the presentation by
  running `dev-up.sh` and clicking around the home page once.

---

## Talking points

- "Multimodal autoencoder over 7 feature blocks, 329k films, latent dim 32."
- "Two paper-worthy methodological findings:"
  1. **NMI ≠ retrieval quality** (journal/07): clustering metrics
     do not predict cosine-retrieval quality. NMI champion had
     angular collapse → random top-5 ranks within cluster.
  2. **z-sweep U-curve** (journal/12): smaller latent (z=32) beat
     larger latent (z=128) on the demo task. Information-bottleneck
     sweet spot.
- "The demo lets you SEE both findings — the gallery is the proof."
