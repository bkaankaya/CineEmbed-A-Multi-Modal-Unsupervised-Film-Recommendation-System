import { z } from "zod";

export const BackboneIdSchema = z.enum(["ae_z32", "ae_z64", "ae_z128"]);
export type BackboneId = z.infer<typeof BackboneIdSchema>;

export const FilmSchema = z.object({
  id: z.number().int(),
  title: z.string(),
  year: z.number().int().nullable(),
  rating: z.number(),
  votes: z.number().int(),
  genres: z.array(z.string()),
  country: z.string().nullable(),
  duration: z.number().nullable(),
  language: z.string(),
  director: z.string(),
  cluster: z.number().int(),
  overview: z.string().nullable(),
  time: z.string(),
  place: z.string().nullable(),
  posterColor: z.string(),
  posterUrl: z.string().nullable(),
  backdropUrl: z.string().nullable(),
  tagline: z.string().nullable(),
  style: z.array(z.string()),
  plot: z.array(z.string()),
  tmdbStatus: z.enum(["ok", "missing"]),
});
export type Film = z.infer<typeof FilmSchema>;

export const NeighborSchema = FilmSchema.extend({ cosine: z.number() });
export type Neighbor = z.infer<typeof NeighborSchema>;

export const BackboneSchema = z.object({
  id: BackboneIdSchema,
  z: z.number().int(),
  label: z.string(),
  genreAtFive: z.number(),
  gnmi: z.number(),
  preferred: z.boolean(),
});
export type Backbone = z.infer<typeof BackboneSchema>;

export const ClusterSchema = z.object({
  id: z.number().int(),
  name: z.string(),
  size: z.number().int(),
  topGenres: z.array(z.object({ genre: z.string(), pct: z.number() })),
  modalDecade: z.string(),
  previewFilms: z.array(FilmSchema),
});
export type Cluster = z.infer<typeof ClusterSchema>;

export const ClusterDetailSchema = ClusterSchema.extend({
  films: z.array(FilmSchema),
  total: z.number().int(),
});
export type ClusterDetail = z.infer<typeof ClusterDetailSchema>;

export const HealthSchema = z.object({
  status: z.literal("ok"),
  backbones_loaded: z.array(z.string()),
  films: z.number().int(),
  tmdb_key_configured: z.boolean(),
});
export type Health = z.infer<typeof HealthSchema>;

export const CosineHistogramSchema = z.object({
  bins: z.array(z.number()),
  counts: z.array(z.number().int()),
  stats: z.object({
    mean: z.number(),
    std: z.number(),
    min: z.number(),
    max: z.number(),
    p50: z.number(),
    p95: z.number(),
  }),
  top10: z.array(z.object({
    id: z.number().int(),
    title: z.string(),
    cosine: z.number(),
  })),
});
export type CosineHistogram = z.infer<typeof CosineHistogramSchema>;

export const GallerySchema = z.object({
  queries: z.array(z.string()),
  matrix: z.record(z.string(), z.record(z.string(), z.object({
    query: FilmSchema,
    neighbors: z.array(NeighborSchema),
  }))),
});
export type Gallery = z.infer<typeof GallerySchema>;
