"""Build inference index for the CineEmbed web app demo.

Loads a trained model checkpoint, encodes all 329k films through the backbone,
L2-normalizes the latents, computes retrieval-quality metrics, and writes the
inference artifacts the FastAPI backend will serve from RAM at runtime.

Spec: docs/superpowers/specs/2026-05-16-web-app-demo-design.md

Outputs (under <out>/):
    embeddings.npy   (N, latent_dim) float32, L2-normalized → cosine = dot product
    films.parquet    id, title, year, director_name, genres (list[str]), overview,
                     popularity, vote_average, vote_count, original_language
    manifest.json    provenance (checkpoint SHA, model type, retrieval metrics,
                     angular-spread sanity, film count)

Usage (local CPU is fast enough — ~5-8 min for 329k films):

    # Demo backbone (Round 2 winner, z=32 — D15 in ADR 0001):
    python scripts/build_index.py \
        --checkpoint artifacts/models/ae_z32/ae.pt \
        --model-type ae \
        --out artifacts/inference/ae_z32/ \
        --retrieval-eval --eyeball

    # MVP carry-over for comparison:
    python scripts/build_index.py \
        --checkpoint artifacts/models/ae_z64.pt \
        --model-type ae \
        --out artifacts/inference/ae_z64/ \
        --retrieval-eval --eyeball

    # DEC (disqualified — angular collapse; kept for reproducibility of the
    # 2026-05-17 AM "NMI ≠ retrieval" finding):
    python scripts/build_index.py \
        --checkpoint artifacts/models/dec_z64_k21.pt \
        --model-type dec --n-clusters 21 \
        --out artifacts/inference/dec_z64_k21/ \
        --retrieval-eval --eyeball

The retrieval metrics (genre@k) and the eyeball table are the relevant signals
for the demo, NOT the clustering NMI used during model selection. A model with
lower NMI but smoother latent geometry can give better top-k recommendations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from cineembed import data as cdata
from cineembed.backbone import MultiModalBackbone, DEFAULT_PROJ_DIMS
from cineembed.heads import AEHead, DECHead, VAEHead


# ── A small, well-known query set for the eyeball sanity check ───────────────
# These are titles likely to appear in any TMDb-derived dataset.  The
# build_index script does a fuzzy title lookup at the end and prints top-5
# similar films for each query so we can visually judge demo quality
# (a more reliable signal than NMI when judging recommender output).
EYEBALL_QUERIES = [
    'Inception',
    'The Godfather',
    'Toy Story',
    'The Shawshank Redemption',
    'Pulp Fiction',
    'The Matrix',
    'Interstellar',
    'Forrest Gump',
    'The Dark Knight',
    'Spirited Away',
]


# ─────────────────────────────────────────────────────────────────────────────
def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build CineEmbed inference index.')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Path to a trained checkpoint .pt file.')
    p.add_argument('--model-type', required=True,
                   choices=['ae', 'dec', 'vae', 'backbone'],
                   help='How to reconstruct the head before extracting backbone. '
                        '"backbone" means the checkpoint is already a backbone-only state_dict.')
    p.add_argument('--n-clusters', type=int, default=21,
                   help='Only used for --model-type=dec (matches training).')
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--hidden-dim', type=int, default=128)

    p.add_argument('--artifacts', type=Path, default=Path('artifacts'),
                   help='Directory containing feature_matrix.npz and movies_eda_final.csv.')
    p.add_argument('--out', type=Path, required=True,
                   help='Output directory for embeddings.npy + films.parquet + manifest.json.')

    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=2048)

    p.add_argument('--retrieval-eval', action='store_true',
                   help='Compute genre@k retrieval quality + angular-spread sanity.')
    p.add_argument('--retrieval-k', type=int, default=5)
    p.add_argument('--retrieval-n-queries', type=int, default=500)

    p.add_argument('--eyeball', action='store_true',
                   help='Print top-5 similar for a curated set of well-known queries.')

    p.add_argument('--cluster-only', action='store_true',
                   help='Skip encode, only run KMeans on existing embeddings.npy')
    p.add_argument('--with-clusters', action='store_true',
                   help='After encode, also run KMeans and write cluster_labels.npy + cluster_meta.json')

    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return arg


def _checkpoint_sha(path: Path, n_bytes: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[: n_bytes * 2]


def _load_backbone(args: argparse.Namespace, block_dims: dict) -> MultiModalBackbone:
    """Reconstruct a MultiModalBackbone from a checkpoint, regardless of which
    head was used during training.

    For ae/dec/vae the checkpoint stores the full head state_dict; we build the
    head, load_state_dict, then peel off head.backbone.  For 'backbone' the
    checkpoint is a backbone-only state_dict and we load directly.
    """
    bb = MultiModalBackbone(
        block_dims,
        proj_dims=DEFAULT_PROJ_DIMS,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )

    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # Older training scripts wrap state in {'model_state': sd, 'epoch':..., 'val_loss':..., 'history':...}.
    # New backbone-only saves are a flat state_dict.
    if isinstance(raw, dict) and 'model_state' in raw:
        state = raw['model_state']
    else:
        state = raw

    if args.model_type == 'backbone':
        missing, unexpected = bb.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f'  [warn] backbone state_dict load: missing={len(missing)} unexpected={len(unexpected)}')
        return bb

    if args.model_type == 'ae':
        head = AEHead(bb, block_dims, DEFAULT_PROJ_DIMS, hidden_dim=args.hidden_dim)
        head.load_state_dict(state)
        return head.backbone

    if args.model_type == 'vae':
        head = VAEHead(bb, block_dims, DEFAULT_PROJ_DIMS, hidden_dim=args.hidden_dim)
        head.load_state_dict(state)
        # VAE's deterministic encoder is the mu head; the raw backbone alone
        # produces the pre-fc_mu vector.  For retrieval we ideally want mu, but
        # exporting only the backbone is fine when the backbone latent_dim
        # already equals the eval dim and we're comfortable losing the small
        # linear fc_mu transform.  Flag the trade-off so the caller can decide.
        print('  [warn] VAE detected — exporting backbone-only (raw pre-fc_mu vector). '
              'Embedding fidelity reflects backbone, not VAE mu head.')
        return head.backbone

    if args.model_type == 'dec':
        head_ae = AEHead(bb, block_dims, DEFAULT_PROJ_DIMS, hidden_dim=args.hidden_dim)
        head_dec = DECHead(bb, head_ae.decoder,
                           n_clusters=args.n_clusters,
                           latent_dim=args.latent_dim)
        head_dec.load_state_dict(state)
        return head_dec.backbone

    raise ValueError(f'Unknown model_type: {args.model_type}')


@torch.no_grad()
def _encode_all(bb: MultiModalBackbone,
                X: torch.Tensor, has_bio: torch.Tensor,
                block_slices: dict, device: str, batch_size: int) -> np.ndarray:
    """Forward all N rows through the backbone, return L2-normalized (N, d) float32."""
    bb = bb.to(device).eval()
    loader = cdata.make_dataloader(X, has_bio, batch_size=batch_size, shuffle=False,
                                    block_slices=block_slices, seed=None)
    chunks = []
    for batch in loader:
        blocks_dev = {b: t.to(device) for b, t in batch['blocks'].items()}
        z = bb(blocks_dev)
        chunks.append(z.cpu().numpy())
    z_all = np.concatenate(chunks, axis=0).astype(np.float32)
    norms = np.linalg.norm(z_all, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (z_all / norms).astype(np.float32)


def _parse_release_year(s):
    """movies_eda_final.csv uses dd/mm/YYYY; some rows are NaN."""
    if not isinstance(s, str):
        return None
    try:
        return int(s.split('/')[-1])
    except (ValueError, IndexError):
        return None


def _build_films_table(artifacts: Path, expected_rows: int) -> pd.DataFrame:
    """Read movies_eda_final.csv → minimal parquet for the API.

    Important: the CSV row order MUST match the feature_matrix.npz row order,
    because the latent index is positional.  This is the same invariant the
    training notebooks rely on; we assert it explicitly here.
    """
    csv_path = artifacts / 'movies_eda_final.csv'
    df = pd.read_csv(csv_path, low_memory=False)
    assert len(df) == expected_rows, (
        f'Row mismatch: csv has {len(df)} rows but feature_matrix has {expected_rows}. '
        'The CSV and the feature matrix must be in the same order.'
    )

    keep = ['id', 'imdb_id', 'title', 'original_title', 'release_date',
            'director_name', 'genres', 'overview', 'popularity',
            'vote_average', 'vote_count', 'runtime', 'original_language']
    keep = [c for c in keep if c in df.columns]
    out: pd.DataFrame = df.loc[:, keep].copy()  # type: ignore[assignment]

    if 'release_date' in out.columns:
        out['year'] = pd.Series(out['release_date']).apply(_parse_release_year)

    if 'genres' in out.columns:
        out['genres'] = pd.Series(out['genres']).fillna('').apply(
            lambda s: [g for g in s.split('|') if g] if isinstance(s, str) else []
        )

    return out


def _retrieval_eval(z_norm: np.ndarray, films_df: pd.DataFrame,
                    k: int = 5, n_queries: int = 500, seed: int = 42) -> dict:
    """Compute genre@k on a random query subset, plus angular-spread sanity.

    Returns:
        {
          'k': k,
          'n_queries': effective query count (skips films with no genre),
          'genre_at_k_mean':   mean fraction of top-k sharing primary_genre,
          'genre_at_k_median': median fraction,
          'genre_at_k_std':    std,
          'angular': {
              'random_pair_cos_mean':  expected ~0 for healthy spread on 64-d sphere,
              'random_pair_cos_std':   spread; very low std = collapsed,
              'random_pair_cos_min':   most-dissimilar pair seen,
              'random_pair_cos_max':   most-similar (off-self) pair seen,
          },
        }
    """
    rng = np.random.default_rng(seed)
    n = z_norm.shape[0]
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)

    primary_genre = np.asarray(
        films_df['genres'].apply(lambda gs: gs[0] if gs else '').values
    )

    genre_at_k_scores = []
    for qi in query_idx:
        if not primary_genre[qi]:
            continue  # skip queries with no genre label
        sims = z_norm @ z_norm[qi]
        # Top k+1 (first is self), drop self
        top = np.argpartition(-sims, k + 1)[: k + 1]
        top = top[np.argsort(-sims[top])]
        neighbors = top[top != qi][:k]
        same_genre = int((primary_genre[neighbors] == primary_genre[qi]).sum())
        genre_at_k_scores.append(same_genre / k)

    # Angular-spread sanity: random pairs of films should have cosine
    # distributed roughly around 0 for a high-d sphere if the latent is not
    # collapsed.  Sample 5000 random pairs and report statistics.
    pair_a = rng.choice(n, size=5000, replace=False)
    pair_b = rng.choice(n, size=5000, replace=False)
    pair_a = pair_a[pair_a != pair_b]
    pair_b = pair_b[: len(pair_a)]
    pair_cos = (z_norm[pair_a] * z_norm[pair_b]).sum(axis=1)

    return {
        'k': k,
        'n_queries': len(genre_at_k_scores),
        'genre_at_k_mean':   float(np.mean(genre_at_k_scores)) if genre_at_k_scores else None,
        'genre_at_k_median': float(np.median(genre_at_k_scores)) if genre_at_k_scores else None,
        'genre_at_k_std':    float(np.std(genre_at_k_scores)) if genre_at_k_scores else None,
        'angular': {
            'random_pair_cos_mean': float(pair_cos.mean()),
            'random_pair_cos_std':  float(pair_cos.std()),
            'random_pair_cos_min':  float(pair_cos.min()),
            'random_pair_cos_max':  float(pair_cos.max()),
        },
    }


def _eyeball_top5(z_norm: np.ndarray, films_df: pd.DataFrame,
                  queries: list, k: int = 5) -> list:
    """For each query title, find the closest matching film in the dataset,
    then print its top-k neighbors.  Returns a list of dicts for the manifest.
    """
    print()
    print('Eyeball sanity check — top-{} similar for well-known queries:'.format(k))
    print('=' * 78)

    rows = []
    titles_lower = films_df['title'].astype(str).str.lower()
    for q in queries:
        # Pick the most-popular exact-match if any, else first contains-match.
        ql = q.lower()
        exact = films_df.index[titles_lower == ql].tolist()
        if exact:
            qi = max(exact, key=lambda i: films_df.iloc[i].get('popularity', 0) or 0)
        else:
            contains = films_df.index[titles_lower.str.contains(ql, regex=False, na=False)].tolist()
            if not contains:
                print(f'\n  ?? "{q}"  — no match in dataset')
                continue
            qi = max(contains, key=lambda i: films_df.iloc[i].get('popularity', 0) or 0)

        sims = z_norm @ z_norm[qi]
        top = np.argpartition(-sims, k + 1)[: k + 1]
        top = top[np.argsort(-sims[top])]
        neighbors = [int(i) for i in top if int(i) != int(qi)][:k]

        q_row = films_df.iloc[qi]
        q_label = f"{q_row['title']} ({q_row.get('year') or '?'})"
        print(f'\n  Q: {q_label}')
        for j, ni in enumerate(neighbors, 1):
            n_row = films_df.iloc[ni]
            g_str = '|'.join(n_row['genres'][:3]) if n_row['genres'] else ''
            print(f'     {j}. {n_row["title"]} ({n_row.get("year") or "?"})  '
                  f'[{g_str}]  cos={sims[ni]:.3f}')

        rows.append({
            'query': q,
            'matched_title': q_row['title'],
            'matched_year': int(q_row['year']) if pd.notna(q_row.get('year')) else None,
            'neighbors': [
                {
                    'title': films_df.iloc[int(ni)]['title'],
                    'year': int(films_df.iloc[int(ni)].get('year'))
                            if pd.notna(films_df.iloc[int(ni)].get('year')) else None,
                    'cosine': float(sims[int(ni)]),
                }
                for ni in neighbors
            ],
        })
    print('=' * 78)
    return rows


def _run_clustering(out_dir: Path, k: int = 21) -> None:
    """KMeans pass on the embeddings.npy at <out_dir>, write cluster_labels.npy +
    cluster_meta.json next to it.  Used by both --cluster-only and --with-clusters.
    """
    from sklearn.cluster import MiniBatchKMeans
    from cineembed.cluster_naming import auto_name_clusters

    print(f'[cluster] running MiniBatchKMeans k={k} on existing embeddings...')
    embs = np.load(out_dir / 'embeddings.npy')
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init="auto", random_state=42)
    labels = km.fit_predict(embs).astype(np.uint8)
    np.save(out_dir / 'cluster_labels.npy', labels)
    print(f'[cluster] wrote cluster_labels.npy ({labels.shape}, {labels.dtype})')

    print('[cluster] computing cluster_meta.json (auto-naming)...')
    films_master = pd.read_parquet(Path('artifacts/inference/films_master.parquet'))
    meta = auto_name_clusters(labels, films_master, k=k)
    with open(out_dir / 'cluster_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[cluster] wrote cluster_meta.json ({k} clusters)')


def main():
    args = _build_args()
    device = _resolve_device(args.device)

    artifacts = args.artifacts.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── --cluster-only: skip encode, only run KMeans on existing embeddings ───
    if args.cluster_only:
        print(f'[setup] cluster-only mode — using existing {out_dir / "embeddings.npy"}')
        _run_clustering(out_dir, k=21)
        return

    t0 = time.time()
    print(f'[setup] checkpoint   = {args.checkpoint}')
    print(f'[setup] model_type   = {args.model_type}')
    print(f'[setup] latent_dim   = {args.latent_dim}  hidden_dim = {args.hidden_dim}')
    print(f'[setup] device       = {device}')
    print(f'[setup] artifacts    = {artifacts}')
    print(f'[setup] out          = {out_dir}')

    sha = _checkpoint_sha(args.checkpoint)
    print(f'[setup] sha256[:32]  = {sha}')

    # ── data ────────────────────────────────────────────────────────────────
    X, feature_names = cdata.load_feature_matrix(artifacts / 'feature_matrix.npz')
    block_slices = cdata.get_block_indices(feature_names)
    has_bio = X[:, block_slices['director'].start + 64].clone()
    block_dims = {b: (slc.stop - slc.start) for b, slc in block_slices.items()}
    print(f'[data ] X            = {tuple(X.shape)}  has_bio_sum={int(has_bio.sum())}')

    # ── model ───────────────────────────────────────────────────────────────
    bb = _load_backbone(args, block_dims)
    n_params = sum(p.numel() for p in bb.parameters())
    print(f'[model] backbone     = {bb.__class__.__name__}  params={n_params:,}')

    # ── encode ──────────────────────────────────────────────────────────────
    t_enc = time.time()
    print(f'[encode] forwarding {X.shape[0]} films through backbone ...')
    z_norm = _encode_all(bb, X, has_bio, block_slices, device, args.batch_size)
    enc_seconds = time.time() - t_enc
    print(f'[encode] done in {enc_seconds:.1f}s  '
          f'shape={z_norm.shape}  dtype={z_norm.dtype}  '
          f'L2_norm_mean={np.linalg.norm(z_norm, axis=1).mean():.4f}')

    # ── films metadata ──────────────────────────────────────────────────────
    films_df = _build_films_table(artifacts, expected_rows=z_norm.shape[0])
    print(f'[films] columns      = {list(films_df.columns)}')

    # ── save embeddings + films ─────────────────────────────────────────────
    emb_path = out_dir / 'embeddings.npy'
    films_path = out_dir / 'films.parquet'
    np.save(emb_path, z_norm)
    films_df.to_parquet(films_path, index=False)
    print(f'[save ] embeddings.npy = {emb_path.stat().st_size / 1e6:.1f} MB')
    print(f'[save ] films.parquet  = {films_path.stat().st_size / 1e6:.1f} MB')

    # ── optional retrieval eval ─────────────────────────────────────────────
    retrieval = None
    if args.retrieval_eval:
        t_ret = time.time()
        retrieval = _retrieval_eval(
            z_norm, films_df,
            k=args.retrieval_k,
            n_queries=args.retrieval_n_queries,
        )
        ret_seconds = time.time() - t_ret
        print(f'[retrieval] eval done in {ret_seconds:.1f}s on {retrieval["n_queries"]} queries')
        print(f'[retrieval] genre@{retrieval["k"]} = '
              f'{retrieval["genre_at_k_mean"]:.3f} ± {retrieval["genre_at_k_std"]:.3f}  '
              f'(median {retrieval["genre_at_k_median"]:.3f})')
        ang = retrieval['angular']
        print(f'[retrieval] random-pair cosine: '
              f'mean={ang["random_pair_cos_mean"]:.3f}  '
              f'std={ang["random_pair_cos_std"]:.3f}  '
              f'range=[{ang["random_pair_cos_min"]:.3f}, {ang["random_pair_cos_max"]:.3f}]')

    # ── optional eyeball sanity ─────────────────────────────────────────────
    eyeball = None
    if args.eyeball:
        eyeball = _eyeball_top5(z_norm, films_df, EYEBALL_QUERIES, k=5)

    # ── optional clustering pass ────────────────────────────────────────────
    if args.with_clusters:
        _run_clustering(out_dir, k=21)

    # ── manifest ────────────────────────────────────────────────────────────
    manifest = {
        'schema_version':         1,
        'created_unix_seconds':   int(time.time()),
        'checkpoint':             str(args.checkpoint),
        'checkpoint_sha256_32':   sha,
        'model_type':             args.model_type,
        'latent_dim':             args.latent_dim,
        'hidden_dim':             args.hidden_dim,
        'n_clusters':             args.n_clusters if args.model_type == 'dec' else None,
        'n_films':                int(z_norm.shape[0]),
        'embedding_dim':          int(z_norm.shape[1]),
        'normalization':          'L2',
        'distance_metric':        'cosine (dot product after L2 normalization)',
        'retrieval':              retrieval,
        'eyeball':                eyeball,
        'wall_clock_seconds':     round(time.time() - t0, 1),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[save ] manifest.json  ({(out_dir / "manifest.json").stat().st_size / 1e3:.1f} KB)')
    print(f'[done ] inference index ready at {out_dir}  '
          f'(total {manifest["wall_clock_seconds"]:.1f}s)')


if __name__ == '__main__':
    main()
