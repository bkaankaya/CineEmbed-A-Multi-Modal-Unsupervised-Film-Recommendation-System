"use client";

import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Info } from "lucide-react";
import { ErrorFallback } from "./error-fallback";
import { api, type BackboneId } from "@/lib/api";

interface Props {
  filmId: number;
  backbone: BackboneId;
}

export function CosineHeatmap({ filmId, backbone }: Props) {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["cosineDist", filmId, backbone],
    queryFn: ({ signal }) => api.getCosineDist(filmId, backbone, 30, { signal }),
  });

  if (isError) {
    return (
      <div className="mt-6">
        <ErrorFallback title="Couldn't load cosine distribution" error={error} onRetry={() => refetch()} />
      </div>
    );
  }

  if (isLoading || !data) {
    return <div className="mt-6 h-44 rounded shimmer" />;
  }

  const histData = data.counts.map((c, i) => ({
    bin: data.bins[i].toFixed(2),
    count: c,
  }));

  return (
    <div className="mt-6 border-t border-border pt-4">
      <p className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1.5">
        <span>Cosine distribution across 329,043 films</span>
        <span title="Cosine: 1 = same direction in latent, 0 = orthogonal" className="inline-flex">
          <Info className="w-3 h-3" aria-label="cosine info" />
        </span>
      </p>
      <p className="text-xs text-muted-foreground mb-3 tabular-nums">
        μ={data.stats.mean.toFixed(2)} · σ={data.stats.std.toFixed(2)} ·
        p50={data.stats.p50.toFixed(2)} · p95={data.stats.p95.toFixed(2)} ·
        top={data.stats.max.toFixed(3)}
      </p>
      <span className="sr-only">
        Cosine distribution. Mean {data.stats.mean.toFixed(2)},
        standard deviation {data.stats.std.toFixed(2)},
        top neighbor cosine {data.stats.max.toFixed(2)}.
      </span>
      <div style={{ width: "100%", height: 140 }}>
        <ResponsiveContainer>
          <BarChart data={histData}>
            <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={4} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#a78bfa" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs font-medium text-muted-foreground mt-3 mb-1">Top-10 cosines</p>
      <div className="space-y-1">
        {data.top10.map((t) => (
          <div key={t.id} className="flex justify-between text-xs">
            <span className="truncate flex-1 pr-2">{t.title}</span>
            <span className="text-muted-foreground tabular-nums">{t.cosine.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
