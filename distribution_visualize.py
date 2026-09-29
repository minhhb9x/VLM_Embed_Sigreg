import argparse
import os

import torch
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use("Agg")  # Chạy trên server không có màn hình
import matplotlib.pyplot as plt

def compute_effective_rank(
    hidden_state: torch.Tensor,
    eps: float = 1e-10,
    normalize_by_min_dim: bool = False,
) -> torch.Tensor:
    x = hidden_state.float()
    n, d = x.shape

    s = torch.linalg.svdvals(x) / torch.sqrt(torch.tensor(n, device=x.device, dtype=x.dtype))
    eigvals = s.square()
    prob = eigvals.clamp_min(eps) / eigvals.sum().clamp_min(eps)
    entropy = -(prob * torch.log(prob)).sum()

    erank = torch.exp(entropy)
    if normalize_by_min_dim:
        erank = erank / min(n, d)

    return erank


def get_image_token_slice(obj: dict, hidden_state: torch.Tensor) -> slice:
    """Locate the contiguous image-token block in a padding-free sequence."""
    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_valid_tokens = int(obj.get("num_valid_tokens", hidden_state.size(1)))

    if hidden_state.size(1) != num_valid_tokens:
        raise ValueError(
            f"Saved hidden length ({hidden_state.size(1)}) does not match "
            f"num_valid_tokens ({num_valid_tokens})."
        )

    if bool(obj.get("last_image_token", False)):
        image_end = num_valid_tokens - int(bool(obj.get("has_eos_id", False)))
        image_start = image_end - num_image_tokens
    else:
        image_start = 0
        image_end = num_image_tokens

    if image_start < 0 or image_end > num_valid_tokens or image_end <= image_start:
        raise ValueError(
            f"Invalid image-token range [{image_start}, {image_end}) for "
            f"num_valid_tokens={num_valid_tokens} and num_image_tokens={num_image_tokens}."
        )

    return slice(image_start, image_end)


def extract_text_tokens(obj: dict, hidden_state: torch.Tensor, image_slice: slice) -> torch.Tensor:
    """
    Remove the image-token block and keep the text-token sequence.

    If num_text_tokens excludes a terminal EOS, the EOS is removed too.
    If num_text_tokens includes EOS, all non-image tokens are kept.
    """
    before_image = hidden_state[:, :image_slice.start, :]
    after_image = hidden_state[:, image_slice.stop:, :]
    text_hidden = torch.cat([before_image, after_image], dim=1)

    num_text_tokens = int(obj.get("num_text_tokens", 0))
    if num_text_tokens <= 0:
        return text_hidden

    if text_hidden.size(1) == num_text_tokens:
        return text_hidden

    has_eos_id = bool(obj.get("has_eos_id", False))
    if has_eos_id and text_hidden.size(1) == num_text_tokens + 1:
        return text_hidden[:, :num_text_tokens, :]

    raise ValueError(
        f"Extracted {text_hidden.size(1)} non-image tokens but num_text_tokens={num_text_tokens}. "
        f"has_eos_id={has_eos_id}."
    )


def load_hidden_layers(
    pt_path: str,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    obj = torch.load(pt_path, map_location="cpu")

    num_image_tokens = int(obj.get("num_image_tokens", 0))
    if num_image_tokens <= 0:
        return None, None, None

    hidden_state = obj["hidden_state"].float()
    image_slice = get_image_token_slice(obj, hidden_state)

    image_hidden_layers = hidden_state[:, image_slice, :]
    text_hidden_layers = extract_text_tokens(obj, hidden_state, image_slice)

    if image_hidden_layers.size(1) != num_image_tokens:
        raise ValueError(
            f"Extracted {image_hidden_layers.size(1)} image tokens from {pt_path}, "
            f"expected {num_image_tokens}."
        )

    if text_hidden_layers.size(1) <= 0:
        raise ValueError(f"No text tokens extracted from {pt_path}.")

    if normalize:
        image_hidden_layers = F.normalize(image_hidden_layers, p=2, dim=-1)
        text_hidden_layers = F.normalize(text_hidden_layers, p=2, dim=-1)

    last_token_all_layers = hidden_state[:, -1, :].clone()
    return image_hidden_layers, text_hidden_layers, last_token_all_layers



def get_pt_files(pt_dir: str, num_samples: int) -> list[str]:
    if not os.path.isdir(pt_dir):
        raise FileNotFoundError(f"PT directory does not exist: {pt_dir}")

    pt_files = [
        os.path.join(pt_dir, filename)
        for filename in os.listdir(pt_dir)
        if filename.endswith(".pt")
    ]

    if not pt_files:
        raise RuntimeError(f"No .pt files found in: {pt_dir}")

    pt_files.sort()

    if num_samples > 0:
        pt_files = pt_files[:num_samples]

    return pt_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", default="infer/rkd_meta_cls/ImageNet-1K/query")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of first .pt files to use. Use <= 0 to process all files.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize image/text tokens along hidden dimension before eRank.")
    parser.add_argument(
        "--normalize_by_min_dim",
        action="store_true",
        help="Divide effective rank by min(n, d). If omitted, return raw effective rank.",
    )
    parser.add_argument("--num_projections", type=int, default=16)
    parser.add_argument("--plot_dir", type=str, default="projection_plots")
    parser.add_argument("--plot_seed", type=int, default=42)
    parser.add_argument("--plot_bins", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)

    image_per_sample_eranks = []
    text_per_sample_eranks = []
    image_hidden_samples = []
    text_hidden_samples = []
    last_token_all_layers_samples = []
    loaded_files = []

    pt_files = get_pt_files(pt_dir=args.pt_dir, num_samples=args.num_samples)

    print(f"Found {len(pt_files)} .pt files to process.")
    print(f"Effective-rank normalization: {'min(n, d)' if args.normalize_by_min_dim else 'none'}")
    print("First files:")
    for pt_path in pt_files[:10]:
        print(f"  {os.path.basename(pt_path)}")
    if len(pt_files) > 10:
        print("  ...")

    for file_idx, pt_path in enumerate(pt_files):
        print(f"[{file_idx + 1}/{len(pt_files)}] Loading {os.path.basename(pt_path)}")

        image_hidden_layers, text_hidden_layers, last_token_all_layers = load_hidden_layers(
            pt_path,
            normalize=args.normalize,
        )

        # take the last token all layers
        last_token_all_layers_samples.append(last_token_all_layers) # [num_samples, num_layers, hidden_dim]

    last_token_all_layers_samples = torch.stack(last_token_all_layers_samples, dim=0) # [num_samples, num_layers, hidden_dim]

    print(last_token_all_layers_samples.shape)

    samples = last_token_all_layers_samples.float().cpu()
    n_samples, n_layers, hidden_dim = samples.shape
    m_projections = args.num_projections

    os.makedirs(args.plot_dir, exist_ok=True)

    # Một bộ hướng cố định, dùng chung cho mọi layer.
    generator = torch.Generator().manual_seed(args.plot_seed)
    directions = torch.randn(
        hidden_dim, m_projections, generator=generator
    )
    directions = F.normalize(directions, p=2, dim=0)  # [D, M]

    ncols = min(4, m_projections)
    nrows = math.ceil(m_projections / ncols)

    for layer_idx in range(n_layers):
        x = samples[:, layer_idx, :]      # [N, D]
        y = x @ directions                # [N, M]

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3 * nrows),
            squeeze=False,
        )

        for projection_idx, ax in enumerate(axes.flat):
            if projection_idx >= m_projections:
                ax.axis("off")
                continue

            values = y[:, projection_idx].numpy()

            ax.hist(
                values,
                bins=args.plot_bins,
                density=True,
                alpha=0.65,
                color="steelblue",
                label="Embedding",
            )

            # Đường chuẩn tham chiếu N(0, 1).
            x_min = min(-4.0, float(values.min()))
            x_max = max(4.0, float(values.max()))
            grid = torch.linspace(x_min, x_max, 400)
            normal_pdf = torch.exp(-0.5 * grid.square()) / math.sqrt(2 * math.pi)

            ax.plot(
                grid.numpy(),
                normal_pdf.numpy(),
                color="crimson",
                linewidth=1.5,
                label="N(0, 1)",
            )
            ax.set_title(
                f"Projection {projection_idx} | "
                f"mean={values.mean():.2f}, std={values.std():.2f}",
                fontsize=9,
            )
            ax.set_xlabel("Projected value")
            ax.set_ylabel("Density")

        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Layer {layer_idx} | {n_samples} samples | "
            f"{m_projections} projections",
            fontsize=14,
        )
        fig.tight_layout()

        output_path = os.path.join(
            args.plot_dir, f"layer_{layer_idx:02d}_projections.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
