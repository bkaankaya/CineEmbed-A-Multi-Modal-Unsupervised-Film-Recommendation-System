import { Sidebar } from "@/components/sidebar";
import { Footer } from "@/components/footer";

export default function AboutPage() {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8 max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight mb-6">About CineEmbed</h1>

        <section>
          <p className="text-sm text-foreground leading-relaxed mb-4">
            CineEmbed is a multimodal movie recommender built for SENG 474
            (Deep Learning, TED University, Spring 2026) by Baran Dinçoğuz,
            Arda Arvas, and Kaan Kaya. The model encodes each of 329,044
            films into a 32-dimensional latent space using a multi-modal
            autoencoder over seven feature blocks (numerical metadata,
            genre one-hot, language one-hot, decade scalar, prior-awards,
            text overview embedding, and director profile).
          </p>

          <div className="flex flex-wrap items-center gap-2 my-6 text-xs">
            <span className="px-3 py-1.5 rounded bg-card border border-border">Movies</span>
            <span className="text-muted-foreground" aria-hidden="true">→</span>
            <span className="px-3 py-1.5 rounded bg-card border border-border">Embeddings</span>
            <span className="text-muted-foreground" aria-hidden="true">→</span>
            <span className="px-3 py-1.5 rounded bg-card border border-border">Cosine Search</span>
            <span className="text-muted-foreground" aria-hidden="true">→</span>
            <span className="px-3 py-1.5 rounded bg-card border border-border">Recommendations</span>
          </div>

          <h2 className="text-lg font-medium mt-8 mb-3 text-foreground">Two methodological findings</h2>

          <article className="bg-card border border-border rounded-lg p-5 mb-4 shadow-primary-soft">
            <p className="text-[10px] uppercase tracking-widest text-primary font-semibold mb-2">Finding 1 — Methodology</p>
            <h3 className="text-base font-semibold mb-2 text-foreground">NMI ≠ retrieval quality</h3>
            <p className="text-sm text-foreground leading-relaxed">
              The MVP champion model, <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">dec_z64_k21</code>, won on the
              NMI clustering metric (geo_NMI = 0.323) but collapsed under
              cosine retrieval: every pair of films inside a cluster sat at
              cosine ≈ 1.000 (angular collapse), so top-5 retrieval degenerated
              into a random tie-break. We adopted <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">genre@5</code> — the
              mean fraction of top-5 nearest neighbours sharing a film&rsquo;s
              primary genre — as the demo-relevant metric, and switched the
              demo backbone to <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">ae_z64</code>. See journal/07.
            </p>
          </article>

          <article className="bg-card border border-border rounded-lg p-5 mb-4 shadow-primary-soft">
            <p className="text-[10px] uppercase tracking-widest text-primary font-semibold mb-2">Finding 2 — Information bottleneck</p>
            <h3 className="text-base font-semibold mb-2 text-foreground">Sweet spot at z=32</h3>
            <p className="text-sm text-foreground leading-relaxed">
              Round 2 swept latent dimension across z &isin; &#123;32, 64, 128&#125;
              with the recipe held constant. Counter-intuitively the smallest
              variant won on both <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">genre@5</code> (0.723 vs 0.715 vs
              0.722) and <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">gNMI</code> (0.334 vs 0.328 vs 0.273) — a
              U-curve. The over-parameterised z=128 variant produced
              near-dead latent dimensions and a narrowing pair-cosine
              distribution. We interpret z=32 as the information-bottleneck
              sweet spot for this task. See journal/12.
            </p>
          </article>

          <h2 className="text-lg font-medium mt-8 mb-3 text-foreground">How to read the gallery</h2>
          <p className="text-sm text-foreground leading-relaxed mb-4">
            The <a className="text-primary underline underline-offset-2 hover:text-primary/80" href="/gallery">Gallery</a> page renders five well-known
            queries (Inception, Spirited Away, Shawshank, Pulp Fiction, Toy
            Story) against the three backbones side-by-side. The same
            query produces visibly different top-5 neighbours per backbone
            — the strongest demonstration of the project&rsquo;s findings.
          </p>

          <p className="text-xs text-muted-foreground mt-8">
            Source repo: github.com/bkaankaya/CineEmbed-A-Multi-Modal-Unsupervised-Film-Recommendation-System · branch
            main · spec
            docs/superpowers/specs/2026-05-18-frontend-backend-integration-design.md
          </p>
        </section>
        <Footer />
      </main>
    </div>
  );
}
