"use client";

import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Layers } from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import { BackboneSwitcher } from "@/components/backbone-switcher";
import { ClusterCard } from "@/components/cluster-card";
import { Footer } from "@/components/footer";
import { ErrorFallback } from "@/components/error-fallback";
import { api, type BackboneId } from "@/lib/api";

function ClustersPageContent() {
  const params = useSearchParams();
  const backbone = ((params.get("backbone") ?? "ae_z32") as BackboneId);
  const { data: clusters = [], isLoading, isError, error, refetch } = useQuery({
    queryKey: ["clusters", backbone],
    queryFn: ({ signal }) => api.getClusters(backbone, { signal }),
  });

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 ml-[220px] p-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold tracking-tight mb-6">Clusters (k=21)</h1>
          <BackboneSwitcher />
        </div>

        <section className="bg-card border border-border rounded-lg p-5 mb-6 shadow-primary-soft">
          <div className="flex items-start gap-3">
            <Layers className="w-5 h-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
            <div className="flex-1">
              <h2 className="text-base font-semibold text-foreground mb-1.5">How clusters form</h2>
              <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                We run <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono">MiniBatchKMeans</code> with k=21 on the
                L2-normalized embeddings from the selected backbone. Each of the 329,044 films lands in whichever
                cosine neighborhood it sits closest to. Cluster names like &ldquo;Drama · 1990s&rdquo; are auto-derived
                from the dominant genre and modal decade of their members (heuristic, not hand-curated). Switch
                backbones to see the same films reorganized by a different latent.
              </p>
              <div className="flex flex-wrap gap-1.5 text-[10px] text-muted-foreground tabular-nums">
                <span className="px-2 py-0.5 rounded bg-muted">k=21</span>
                <span className="px-2 py-0.5 rounded bg-muted">MiniBatchKMeans</span>
                <span className="px-2 py-0.5 rounded bg-muted">L2-normalized cosine</span>
                <span className="px-2 py-0.5 rounded bg-muted">auto-named per cluster</span>
              </div>
            </div>
          </div>
        </section>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="border border-border rounded-lg p-4 h-48 shimmer" />
            ))}
          </div>
        ) : isError ? (
          <ErrorFallback title="Couldn't load clusters" error={error} onRetry={() => refetch()} />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clusters.map((c) => (
              <ClusterCard key={c.id} cluster={c} backbone={backbone} />
            ))}
          </div>
        )}
        <Footer />
      </main>
    </div>
  );
}

export default function ClustersPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-background" />}>
      <ClustersPageContent />
    </Suspense>
  );
}
