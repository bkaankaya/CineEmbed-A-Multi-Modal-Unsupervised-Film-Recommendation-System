# Multi-Modal Model Architecture

```python
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PATH = Path("figures/architecture_multimodal.png")


MODALITIES = [
    ("numerical", 6, 16, "#BFDBFE"),
    ("genre", 22, 16, "#FDBA74"),
    ("language", 31, 16, "#BBF7D0"),
    ("decade", 2, 4, "#E5E7EB"),
    ("awards", 6, 16, "#FDE68A"),
    ("text", 384, 64, "#DDD6FE"),
    ("director", 113, 32, "#FCA5A5"),
]


def box(ax, xy, w, h, text, fc, ec="#334155", lw=1.6, fs=8.5, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111827",
        fontweight=weight,
        family="DejaVu Sans",
    )
    return patch


def arrow(ax, start, end, color="#475569", lw=1.35, ms=11, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def center(x, y, w, h):
    return (x + w / 2, y + h / 2)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    fig.suptitle(
        "CineEmbed - Multi-Modal Backbone (W2 inverse-variance loss)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ax.text(2.65, 5.55, "Encoder", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
    ax.text(10.95, 5.55, "Decoder", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")

    # Encoder: modality inputs and projections.
    input_x, proj_x = 0.35, 2.15
    input_w, proj_w = 1.25, 1.35
    row_h = 0.42
    y0, dy = 4.95, 0.57

    projection_centers = []
    for i, (name, in_dim, out_dim, color) in enumerate(MODALITIES):
        y = y0 - i * dy
        box(ax, (input_x, y), input_w, row_h, f"{name}\n[{in_dim} dims]", color, fs=8.0)
        box(ax, (proj_x, y), proj_w, row_h, f"_BlockProjection\n-> {out_dim}", "#F8FAFC", fs=7.6)
        arrow(ax, (input_x + input_w, y + row_h / 2), (proj_x, y + row_h / 2), color="#64748B")
        projection_centers.append((proj_x + proj_w, y + row_h / 2, color))

    concat = box(ax, (4.05, 2.65), 1.25, 0.58, "Concat\n164-dim", "#FFFFFF", ec="#334155", lw=2.0, fs=8.5, weight="bold")
    concat_left = (4.05, 2.94)
    for x, y, color in projection_centers:
        arrow(ax, (x, y), concat_left, color="#64748B", lw=1.15, ms=9)

    fc1 = box(ax, (5.65, 2.65), 1.45, 0.58, "Linear 164->128\nReLU + Dropout 0.2", "#E0F2FE", ec="#0369A1", fs=7.8)
    fc2 = box(ax, (7.35, 2.65), 1.2, 0.58, "Linear\n128->64", "#E0F2FE", ec="#0369A1", fs=8.0)
    latent = box(ax, (8.85, 2.54), 1.18, 0.80, "Latent\nz in R^64", "#FED7AA", ec="#EA580C", lw=2.8, fs=9.5, weight="bold")

    arrow(ax, (5.30, 2.94), (5.65, 2.94), color="#475569")
    arrow(ax, (7.10, 2.94), (7.35, 2.94), color="#475569")
    arrow(ax, (8.55, 2.94), (8.85, 2.94), color="#EA580C", lw=1.8, ms=13)

    # Decoder: FC expansion, modality decoders, reconstructed outputs.
    dec_fc1 = box(ax, (10.35, 2.65), 1.30, 0.58, "Linear 64->128\nReLU", "#E0F2FE", ec="#0369A1", fs=8.0)
    dec_fc2 = box(ax, (12.00, 2.65), 1.15, 0.58, "Linear\n128->164", "#E0F2FE", ec="#0369A1", fs=8.0)

    arrow(ax, (10.03, 2.94), (10.35, 2.94), color="#EA580C", lw=1.8, ms=13)
    arrow(ax, (11.65, 2.94), (12.00, 2.94), color="#475569")

    decoder_x, recon_x = 10.35, 12.35
    decoder_w, recon_w = 1.32, 1.28
    for i, (name, in_dim, out_dim, color) in enumerate(MODALITIES):
        y = y0 - i * dy
        box(ax, (decoder_x, y), decoder_w, row_h, f"_BlockDecoder\n{out_dim}->{in_dim}", "#F8FAFC", fs=7.5)
        box(ax, (recon_x, y), recon_w, row_h, f"X_hat_{name}\n[{in_dim} dims]", color, fs=7.4)
        arrow(ax, (12.58, 2.65), (decoder_x, y + row_h / 2), color="#64748B", lw=1.05, ms=8)
        arrow(ax, (decoder_x + decoder_w, y + row_h / 2), (recon_x, y + row_h / 2), color="#64748B", lw=1.1, ms=9)

    # Loss and masking notes.
    loss_text = (
        "Reconstruction objective:  L = sum_b w_b * MSE(X_b, X_hat_b)\n"
        "W2 weights: inverse block variance, clipped to [0.1, 10.0]\n"
        "G2 mask: director bio loss only when has_director_bio = 1 (96.8% rows masked out)"
    )
    box(ax, (3.25, 0.30), 6.65, 0.82, loss_text, "#FFF7ED", ec="#C2410C", lw=1.8, fs=8.1)

    # Vanilla baseline inset.
    inset = box(ax, (10.20, 0.23), 3.25, 0.98, "", "#FFFFFF", ec="#94A3B8", lw=1.4, fs=8.0)
    ax.text(11.825, 1.02, "Vanilla concat-AE baseline", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#334155")
    box(ax, (10.38, 0.48), 0.75, 0.28, "564-dim\nconcat", "#F1F5F9", fs=6.5)
    box(ax, (11.35, 0.48), 0.85, 0.28, "Linear\n564->128", "#F1F5F9", fs=6.5)
    box(ax, (12.42, 0.48), 0.75, 0.28, "Linear\n128->64", "#F1F5F9", fs=6.5)
    arrow(ax, (11.13, 0.62), (11.35, 0.62), lw=0.9, ms=7)
    arrow(ax, (12.20, 0.62), (12.42, 0.62), lw=0.9, ms=7)
    ax.text(11.825, 0.34, "no modality projection", ha="center", va="center", fontsize=6.8, color="#64748B")

    plt.subplots_adjust(left=0.02, right=0.99, top=0.91, bottom=0.04)
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
```
