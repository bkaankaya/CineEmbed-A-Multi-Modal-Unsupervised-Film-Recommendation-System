import {
  BackboneSchema, ClusterDetailSchema, ClusterSchema,
  CosineHistogramSchema, FilmSchema, GallerySchema,
  HealthSchema, NeighborSchema,
  type Backbone, type BackboneId, type Cluster, type ClusterDetail,
  type CosineHistogram, type Film, type Gallery, type Health, type Neighbor,
} from "./api-types";
import { z } from "zod";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, public detail?: string) {
    super(`API error ${status}: ${detail ?? ""}`);
  }
}

async function fetchJson<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  const json = await res.json();
  return schema.parse(json);
}

export const api = {
  getHealth: (init?: RequestInit) =>
    fetchJson("/api/health", HealthSchema, init),

  getBackbones: (init?: RequestInit) =>
    fetchJson("/api/backbones", z.array(BackboneSchema), init),

  searchFilms: (q: string, backbone: BackboneId, limit = 10, init?: RequestInit) =>
    fetchJson(
      `/api/films/search?q=${encodeURIComponent(q)}&backbone=${backbone}&limit=${limit}`,
      z.array(FilmSchema),
      init,
    ),

  getFilm: (id: number, backbone: BackboneId, init?: RequestInit) =>
    fetchJson(`/api/films/${id}?backbone=${backbone}`, FilmSchema, init),

  getSimilar: (id: number, backbone: BackboneId, limit = 10, init?: RequestInit) =>
    fetchJson(
      `/api/films/${id}/similar?backbone=${backbone}&limit=${limit}`,
      z.array(NeighborSchema),
      init,
    ),

  getCosineDist: (id: number, backbone: BackboneId, bins = 30, init?: RequestInit) =>
    fetchJson(
      `/api/films/${id}/cosine-dist?backbone=${backbone}&bins=${bins}`,
      CosineHistogramSchema,
      init,
    ),

  getClusters: (backbone: BackboneId, init?: RequestInit) =>
    fetchJson(`/api/clusters?backbone=${backbone}`, z.array(ClusterSchema), init),

  getCluster: (k: number, backbone: BackboneId, limit = 50, init?: RequestInit) =>
    fetchJson(
      `/api/clusters/${k}?backbone=${backbone}&limit=${limit}`,
      ClusterDetailSchema,
      init,
    ),

  getGallery: (init?: RequestInit) =>
    fetchJson("/api/gallery", GallerySchema, init),
};

export type { BackboneId, Film, Neighbor, Backbone, Cluster, ClusterDetail, Health, CosineHistogram, Gallery };
