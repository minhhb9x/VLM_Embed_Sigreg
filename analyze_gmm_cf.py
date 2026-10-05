"""Inspect projected moments and analytic CF of a fitted diagonal GMM."""

import argparse

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gmm_path", help="Checkpoint saved as {'gmm': fitted_model} or a fitted GMM")
    parser.add_argument("--num_slices", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_slices < 1:
        parser.error("--num_slices must be positive")
    checkpoint = joblib.load(args.gmm_path)
    gmm = checkpoint["gmm"] if isinstance(checkpoint, dict) else checkpoint
    if gmm.covariance_type != "diag":
        raise ValueError("This script requires covariance_type='diag'")

    weights = np.asarray(gmm.weights_, dtype=np.float64)
    means = np.asarray(gmm.means_, dtype=np.float64)
    variances = np.asarray(gmm.covariances_, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    directions = rng.standard_normal((means.shape[1], args.num_slices))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)

    projected_means = means @ directions                 # [K, M]
    projected_vars = variances @ (directions ** 2)      # [K, M]
    mixture_mean = np.sum(weights[:, None] * projected_means, axis=0)
    mixture_var = np.sum(
        weights[:, None] * (projected_vars + (projected_means - mixture_mean[None, :]) ** 2), axis=0
    )
    mixture_std = np.sqrt(np.maximum(mixture_var, 0))

    t_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.8, 1, 1.25, 1.5, 2, 3, 5])
    phase = projected_means[:, :, None] * t_values[None, None, :]
    decay = np.exp(-0.5 * projected_vars[:, :, None] * t_values[None, None, :] ** 2)
    cf = np.sum(weights[:, None, None] * decay * np.exp(1j * phase), axis=0)  # [M, T]
    magnitude = np.abs(cf)

    print(f"GMM: K={means.shape[0]}, D={means.shape[1]}, covariance=diag")
    print(f"Projected mixture mean (p10 / median / p90): {np.quantile(mixture_mean, [0.1, 0.5, 0.9])}")
    print(f"Projected mixture std  (p10 / median / p90): {np.quantile(mixture_std, [0.1, 0.5, 0.9])}")
    print("\n t      |CF(t)| p10 / median / p90 over random directions")
    for j, t in enumerate(t_values):
        q10, q50, q90 = np.quantile(magnitude[:, j], [0.1, 0.5, 0.9])
        print(f"{t:5.3f}    {q10:7.4f} / {q50:7.4f} / {q90:7.4f}")
    w = gmm.weights_
    mu = gmm.means_
    var = gmm.covariances_
    expected_norm = np.sqrt(np.sum(w * np.sum(mu**2 + var, axis=1)))
    print("GMM expected RMS norm:", expected_norm)

if __name__ == "__main__":
    main()
