"""Plot random 1-D projections of raw rep embeddings in training caches."""
import argparse
import json
import joblib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def sample_dirs(root, subset, limit, seed):
    path = root / subset
    if not path.is_dir():
        raise FileNotFoundError(path)
    dirs = sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not dirs:
        raise RuntimeError(f"No sample directories in {path}")
    # Uniform selection rather than taking a potentially ordered prefix.
    if limit > 0 and len(dirs) > limit:
        indices = np.sort(np.random.default_rng(seed).choice(len(dirs), limit, replace=False))
        dirs = [dirs[i] for i in indices]
    return dirs


def load_rep(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "rep" not in obj:
        raise KeyError(f"Missing rep in {path}")
    rep = obj["rep"]
    if not isinstance(rep, torch.Tensor) or rep.ndim not in {1, 2}:
        raise ValueError(f"{path}: rep must be a tensor [D] or [N, D]")
    rep = rep.detach().float().reshape(-1, rep.shape[-1])
    if rep.numel() == 0 or not torch.isfinite(rep).all():
        raise ValueError(f"{path}: empty or non-finite rep")
    return rep


def project_subset(dirs, sides, directions, args):
    device = torch.device(args.device)
    projected = {side: [] for side in sides}
    norms = {side: [] for side in sides}

    for side in sides:
        buffer = []
        buffered_rows = 0

        def flush():
            nonlocal buffer, buffered_rows
            if not buffer:
                return
            x = torch.cat(buffer, dim=0).to(device)
            norms[side].append(x.norm(dim=-1).cpu().numpy())
            # No centering, standardization, or embedding normalization.
            projected[side].append((x @ directions).cpu().numpy())
            buffer, buffered_rows = [], 0

        for folder in tqdm(dirs, desc=f"{dirs[0].parent.name}/{side}"):
            path = folder / f"{side}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            rep = load_rep(path)
            if rep.shape[1] != directions.shape[0]:
                raise ValueError(f"{path}: dim={rep.shape[1]}, expected {directions.shape[0]}")
            buffer.append(rep)
            buffered_rows += len(rep)
            if buffered_rows >= args.projection_batch_size:
                flush()
        flush()

    return {side: (np.concatenate(projected[side]), np.concatenate(norms[side])) for side in sides}


def project_gmm(gmm, directions):
    """Project the fitted multivariate GMM, using the exact data directions."""
    means = gmm.means_ @ directions
    if gmm.covariance_type == "diag":
        variances = gmm.covariances_ @ directions**2
    elif gmm.covariance_type == "full":
        variances = np.einsum("dm,kde,em->km", directions, gmm.covariances_, directions)
    elif gmm.covariance_type == "tied":
        v = np.einsum("dm,de,em->m", directions, gmm.covariances_, directions)
        variances = np.broadcast_to(v, means.shape)
    elif gmm.covariance_type == "spherical":
        variances = gmm.covariances_[:, None] * (directions**2).sum(axis=0)[None, :]
    else:
        raise ValueError(f"Unsupported covariance: {gmm.covariance_type}")
    return gmm.weights_, means, np.maximum(variances, 1e-12)


def mixture_pdf(grid, weights, means, variances):
    pdfs = np.exp(-0.5 * (grid[:, None] - means[None, :])**2 / variances[None, :])
    pdfs /= np.sqrt(2 * math.pi * variances[None, :])
    components = pdfs * weights[None, :]
    return components.sum(axis=1), components


def plot_projections(values, norms, title, output_dir, args, projected_gmm=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    m = values.shape[1]
    ncols = min(4, m)
    nrows = math.ceil(m / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    stats = []

    for idx, ax in enumerate(axes.flat):
        if idx >= m:
            ax.axis("off")
            continue
        column = values[:, idx]
        mean, std = float(column.mean()), float(column.std())
        ax.hist(column, bins=args.plot_bins, density=True, alpha=0.65, color="steelblue", label="Raw cache rep")
        gmm_stats = {}
        if projected_gmm is not None:
            weights, mus, vars_ = projected_gmm
            mu, var = mus[:, idx], vars_[:, idx]
            low = min(float(column.min()), float((mu - 4 * np.sqrt(var)).min()))
            high = max(float(column.max()), float((mu + 4 * np.sqrt(var)).max()))
            grid = np.linspace(low, high, 2000)
            total_pdf, components = mixture_pdf(grid, weights, mu, var)
            for k in range(len(weights)):
                ax.plot(grid, components[:, k], color="darkorange", alpha=0.35, linewidth=0.8, label="Weighted components" if k == 0 else None)
            ax.plot(grid, total_pdf, color="crimson", linewidth=1.6, label=f"GMM total (K={len(weights)})")
            mixture_mean = float(weights @ mu)
            mixture_std = float(np.sqrt(weights @ (var + (mu - mixture_mean)**2)))
            gmm_stats = {"gmm_mean": mixture_mean, "gmm_std": mixture_std, "component_means": mu.tolist(), "component_variances": var.tolist()}
        if args.normal_reference:
            low, high = min(-4.0, float(column.min())), max(4.0, float(column.max()))
            grid = np.linspace(low, high, 600)
            ax.plot(grid, np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi), color="crimson", label="N(0, 1)")
        if projected_gmm is not None or args.normal_reference:
            ax.legend(fontsize=7)
        ax.set_title(f"Projection {idx} | mean={mean:.3f}, std={std:.3f}", fontsize=9)
        ax.set_xlabel("Projected value")
        ax.set_ylabel("Density")
        stats.append({"projection": idx, "mean": mean, "std": std, "p01_median_p99": np.percentile(column, [1, 50, 99]).tolist(), **gmm_stats})

    fig.suptitle(f"{title} | N={len(values)} | raw embedding | seed={args.plot_seed}", fontsize=13)
    fig.tight_layout()
    png_path = output_dir / "projections.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "num_embeddings": len(values), "raw_embeddings": True,
        "direction_seed": args.plot_seed,
        "raw_l2_norm_mean": float(norms.mean()), "raw_l2_norm_std": float(norms.std()),
        "projections": stats,
        "gmm_path": args.gmm_path,
        "gmm_component_weights": projected_gmm[0].tolist() if projected_gmm is not None else None,
    }
    with (output_dir / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"Saved {png_path} | raw norm mean={norms.mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caching_dir", required=True)
    parser.add_argument("--gmm_path", default=None, help="Optional joblib GMM fitted on raw embeddings without PCA")
    parser.add_argument("--subset_name", nargs="+", required=True)
    parser.add_argument("--side", choices=["qry", "pos", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of sampled directories PER subset; <=0: all")
    parser.add_argument("--num_projections", type=int, default=16)
    parser.add_argument("--plot_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--plot_bins", type=int, default=100)
    parser.add_argument("--plot_dir", default="projection_plots/train_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--projection_batch_size", type=int, default=256)
    parser.add_argument("--normal_reference", action="store_true", help="Optionally overlay N(0,1); does not normalize embeddings")
    parser.add_argument("--include_combined_all", action="store_true", help="Also plot pooled subsets, weighted by their embedding counts")
    args = parser.parse_args()
    if min(args.num_projections, args.plot_bins, args.projection_batch_size) < 1:
        raise ValueError("num_projections, plot_bins, projection_batch_size must be positive")

    root, output = Path(args.caching_dir), Path(args.plot_dir)
    output.mkdir(parents=True, exist_ok=True)
    sides = ["qry", "pos"] if args.side == "both" else [args.side]
    dirs_by_subset = {subset: sample_dirs(root, subset, args.num_samples, args.sample_seed) for subset in args.subset_name}
    first_dir = dirs_by_subset[args.subset_name[0]][0]
    dim = load_rep(first_dir / f"{sides[0]}.pt").shape[1]
    generator = torch.Generator().manual_seed(args.plot_seed)
    # Normalize only the projection directions; rep stays raw.
    directions = F.normalize(torch.randn(dim, args.num_projections, generator=generator), dim=0)
    np.save(output / "projection_directions.npy", directions.numpy())
    projected_gmm = None
    if args.gmm_path:
        bundle = joblib.load(args.gmm_path)
        if not isinstance(bundle, dict) or "gmm" not in bundle:
            raise ValueError("Expected joblib bundle with key 'gmm'")
        if bundle.get("l2_normalize", False) or bundle.get("pca") is not None:
            raise ValueError("This plot compares raw embeddings: use a GMM fitted without normalization or PCA")
        gmm = bundle["gmm"]
        if gmm.means_.shape[1] != dim:
            raise ValueError(f"GMM dim={gmm.means_.shape[1]}, cache dim={dim}")
        projected_gmm = project_gmm(gmm, directions.numpy().astype(np.float64))
        print(f"Loaded GMM: K={gmm.n_components}, D={dim}, covariance={gmm.covariance_type}")
    directions = directions.to(args.device)
    pooled = {side: [] for side in sides}
    metadata = {**vars(args), "embedding_dimension": dim, "selected_sample_dirs": {subset: len(dirs) for subset, dirs in dirs_by_subset.items()}}
    with (output / "config.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    for subset, dirs in dirs_by_subset.items():
        results = project_subset(dirs, sides, directions, args)
        for side, (values, norms) in results.items():
            plot_projections(values, norms, f"{subset}/{side}", output / subset / side, args, projected_gmm)
            if args.include_combined_all:
                pooled[side].append((values, norms))
        if len(sides) == 2:
            values = np.concatenate([results[side][0] for side in sides])
            norms = np.concatenate([results[side][1] for side in sides])
            plot_projections(values, norms, f"{subset}/qry+pos", output / subset / "combined", args, projected_gmm)

    if args.include_combined_all:
        all_sides = []
        for side in sides:
            values = np.concatenate([pair[0] for pair in pooled[side]])
            norms = np.concatenate([pair[1] for pair in pooled[side]])
            plot_projections(values, norms, f"ALL/{side}", output / "ALL" / side, args, projected_gmm)
            all_sides.append((values, norms))
        if len(sides) == 2:
            plot_projections(np.concatenate([p[0] for p in all_sides]), np.concatenate([p[1] for p in all_sides]), "ALL/qry+pos", output / "ALL" / "combined", args, projected_gmm)


if __name__ == "__main__":
    main()
