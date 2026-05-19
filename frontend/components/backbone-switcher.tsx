"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams, useRouter } from "next/navigation";
import { api, type BackboneId } from "@/lib/api";

const FALLBACK = [
  { id: "ae_z32", z: 32, label: "AE z=32", genreAtFive: 0.723, gnmi: 0.334, preferred: true },
  { id: "ae_z64", z: 64, label: "AE z=64", genreAtFive: 0.715, gnmi: 0.328, preferred: false },
  { id: "ae_z128", z: 128, label: "AE z=128", genreAtFive: 0.722, gnmi: 0.273, preferred: false },
] as const;

export function BackboneSwitcher() {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const current = ((params.get("backbone") ?? "ae_z32") as BackboneId);

  const { data: backbones = FALLBACK } = useQuery({
    queryKey: ["backbones"],
    queryFn: ({ signal }) => api.getBackbones({ signal }),
    staleTime: Infinity,
  });

  const setBackbone = (id: BackboneId) => {
    const next = new URLSearchParams(params.toString());
    next.set("backbone", id);
    router.replace(`?${next.toString()}`, { scroll: false });
    qc.invalidateQueries({ queryKey: ["film"] });
    qc.invalidateQueries({ queryKey: ["similar"] });
    qc.invalidateQueries({ queryKey: ["cosineDist"] });
    qc.invalidateQueries({ queryKey: ["clusters"] });
    qc.invalidateQueries({ queryKey: ["cluster"] });
  };

  const currentBackbone = backbones.find((b) => b.id === current) ?? backbones[0];

  return (
    <div className="inline-flex flex-col items-end gap-1.5">
      <div
        role="radiogroup"
        aria-label="Backbone selection"
        className="inline-flex border border-border rounded-md overflow-hidden bg-card"
      >
        {backbones.map((b) => {
          const active = current === b.id;
          return (
            <button
              key={b.id}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => setBackbone(b.id as BackboneId)}
              title={`${b.label} · genre@5=${b.genreAtFive.toFixed(3)} · gNMI=${b.gnmi.toFixed(3)}`}
              className={`px-3 py-1.5 text-xs font-medium transition ${
                active ? "bg-purple-600 text-white" : "text-gray-700 hover:bg-gray-50"
              }`}
            >
              {b.label}
            </button>
          );
        })}
      </div>
      <p className="text-[10px] text-muted-foreground text-right tabular-nums">
        Active: {currentBackbone.label} · genre@5={currentBackbone.genreAtFive.toFixed(3)} · gNMI={currentBackbone.gnmi.toFixed(3)}
      </p>
    </div>
  );
}
