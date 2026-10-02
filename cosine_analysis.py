"""Analyze image-token cosine from cached .pt hidden states."""

import argparse
import os

import torch
import torch.nn.functional as F


def mean_pairwise_cosine(image_tokens: torch.Tensor) -> torch.Tensor:
    """Mean signed cosine between distinct tokens within one layer: [N, D]."""
    n = image_tokens.size(0)
    if n < 2:
        return image_tokens.new_tensor(float("nan"), dtype=torch.float32)

    x = F.normalize(image_tokens.float(), p=2, dim=-1)

    # sum_{i != j} x_i · x_j = ||sum_i x_i||² - sum_i ||x_i||²
    return (x.sum(dim=0).square().sum() - x.square().sum()) / (n * (n - 1))


def consecutive_layer_cosine(
    layer_a: torch.Tensor,
    layer_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compare the same image-token position in two consecutive layers.

    Returns:
        mean_cosine: mean signed cosine
        mean_abs_cosine: mean of absolute cosine
    """
    if layer_a.shape != layer_b.shape:
        raise ValueError(
            f"Consecutive layers have different shapes: "
            f"{tuple(layer_a.shape)} vs {tuple(layer_b.shape)}"
        )

    if layer_a.size(0) == 0:
        nan = layer_a.new_tensor(float("nan"), dtype=torch.float32)
        return nan, nan

    a = F.normalize(layer_a.float(), p=2, dim=-1)
    b = F.normalize(layer_b.float(), p=2, dim=-1)
    cosine_per_token = (a * b).sum(dim=-1)  # [N]

    return cosine_per_token.mean(), cosine_per_token.abs().mean()


def get_image_token_slice(obj: dict, hidden_state: torch.Tensor) -> slice:
    """Use cached metadata to locate the image-token block."""
    num_image_tokens = int(obj.get("num_image_tokens", 0))
    num_valid_tokens = int(obj.get("num_valid_tokens", hidden_state.size(1)))

    if hidden_state.size(1) != num_valid_tokens:
        raise ValueError(
            f"Saved hidden length ({hidden_state.size(1)}) "
            f"!= num_valid_tokens ({num_valid_tokens})."
        )

    if bool(obj.get("last_image_token", False)):
        image_end = num_valid_tokens - int(bool(obj.get("has_eos_id", False)))
        image_start = image_end - num_image_tokens
    else:
        image_start, image_end = 0, num_image_tokens

    if image_start < 0 or image_end > num_valid_tokens or image_end <= image_start:
        raise ValueError(
            f"Invalid image-token range [{image_start}, {image_end}) "
            f"in {num_valid_tokens} tokens."
        )

    return slice(image_start, image_end)


def load_image_layers(pt_path: str) -> torch.Tensor | None:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    num_image_tokens = int(obj.get("num_image_tokens", 0))

    if num_image_tokens <= 0:
        return None

    hidden_state = obj["hidden_state"]  # [L, valid_sequence_length, D]
    image_layers = hidden_state[:, get_image_token_slice(obj, hidden_state), :]

    if image_layers.size(1) != num_image_tokens:
        raise ValueError(f"Unexpected image-token count in {pt_path}")

    return image_layers


def get_pt_files(pt_dir: str, num_samples: int) -> list[str]:
    if not os.path.isdir(pt_dir):
        raise FileNotFoundError(f"PT directory does not exist: {pt_dir}")

    pt_files = sorted(
        os.path.join(pt_dir, name)
        for name in os.listdir(pt_dir)
        if name.endswith(".pt")
    )

    if not pt_files:
        raise RuntimeError(f"No .pt files found in: {pt_dir}")

    return pt_files[:num_samples] if num_samples > 0 else pt_files


def summarize_across_samples(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and population std for values shaped [samples, layers]."""
    valid = torch.isfinite(values)
    count = valid.sum(dim=0).clamp_min(1)

    mean = torch.where(valid, values, 0).sum(dim=0) / count
    std = (
        torch.where(valid, values - mean, 0).square().sum(dim=0) / count
    ).sqrt()

    mean[~valid.any(dim=0)] = float("nan")
    std[~valid.any(dim=0)] = float("nan")

    return mean, std


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_dir",
        default="infer/rkd_meta_cls/ImageNet-1K/query",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="<= 0: process all .pt files",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_file",
        default="image_token_cosine_results.txt",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    pt_files = get_pt_files(args.pt_dir, args.num_samples)
    print(f"Found {len(pt_files)} .pt files to process.")

    sample_within = []
    sample_cross = []
    sample_cross_abs = []
    loaded_files = []

    for file_idx, pt_path in enumerate(pt_files):
        print(f"[{file_idx + 1}/{len(pt_files)}] Loading {os.path.basename(pt_path)}")

        image_layers = load_image_layers(pt_path)
        if image_layers is None:
            print(f"Skip no-image file: {pt_path}")
            continue

        if image_layers.size(0) < 2:
            raise ValueError(f"Need at least 2 layers in {pt_path}")

        within_values = []
        cross_values = []
        cross_abs_values = []

        previous_layer = None

        for layer in image_layers:
            current_layer = layer.to(device)

            # Chỉ số cũ: cosine giữa các token trong cùng layer.
            within_values.append(
                mean_pairwise_cosine(current_layer).cpu()
            )

            # Chỉ số mới: cùng token ở hai layer liên tiếp.
            if previous_layer is not None:
                cross, cross_abs = consecutive_layer_cosine(
                    previous_layer,
                    current_layer,
                )
                cross_values.append(cross.cpu())
                cross_abs_values.append(cross_abs.cpu())

            previous_layer = current_layer

        sample_within.append(torch.stack(within_values))        # [L]
        sample_cross.append(torch.stack(cross_values))          # [L - 1]
        sample_cross_abs.append(torch.stack(cross_abs_values))  # [L - 1]
        loaded_files.append(pt_path)

    if not loaded_files:
        raise RuntimeError("No samples with image tokens were loaded.")

    within_mean, within_std = summarize_across_samples(
        torch.stack(sample_within)
    )
    cross_mean, cross_std = summarize_across_samples(
        torch.stack(sample_cross)
    )
    cross_abs_mean, cross_abs_std = summarize_across_samples(
        torch.stack(sample_cross_abs)
    )

    lines = [
        f"Loaded samples: {len(loaded_files)}",
        "Within-layer: mean signed cosine between distinct image tokens",
        "Cross-layer: same image-token position in consecutive layers",
        "",
        "WITHIN-LAYER IMAGE TOKEN COSINE (MEAN/STD ACROSS SAMPLES)",
    ]

    for layer_idx, (mean, std) in enumerate(zip(within_mean, within_std)):
        lines.append(
            f"  layer {layer_idx:02d}: "
            f"mean={mean.item():.6f}, std={std.item():.6f}"
        )

    lines.extend([
        "",
        "CONSECUTIVE-LAYER IMAGE TOKEN COSINE (MEAN/STD ACROSS SAMPLES)",
    ])

    for layer_idx in range(cross_mean.numel()):
        lines.append(
            f"  layer {layer_idx:02d}->{layer_idx + 1:02d}: "
            f"mean={cross_mean[layer_idx].item():.6f}, "
            f"std={cross_std[layer_idx].item():.6f}, "
            f"mean_abs={cross_abs_mean[layer_idx].item():.6f}, "
            f"std_abs={cross_abs_std[layer_idx].item():.6f}"
        )

    lines.extend(["", "COSINE PER SAMPLE"])

    for pt_path, within, cross, cross_abs in zip(
        loaded_files,
        sample_within,
        sample_cross,
        sample_cross_abs,
    ):
        lines.append(f"  {os.path.basename(pt_path)}")

        within_entries = ", ".join(
            f"layer {i:02d}={value.item():.6f}"
            for i, value in enumerate(within)
        )
        cross_entries = ", ".join(
            f"layer {i:02d}->{i + 1:02d}={value.item():.6f}"
            for i, value in enumerate(cross)
        )
        cross_abs_entries = ", ".join(
            f"layer {i:02d}->{i + 1:02d}={value.item():.6f}"
            for i, value in enumerate(cross_abs)
        )

        lines.append(f"    within: {within_entries}")
        lines.append(f"    cross: {cross_entries}")
        lines.append(f"    cross_abs: {cross_abs_entries}")

    output_text = "\n".join(lines)
    print("\n" + output_text)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\nSaved output to: {args.output_file}")


if __name__ == "__main__":
    main()