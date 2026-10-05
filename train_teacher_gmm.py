import os
import time
import joblib
import numpy as np
import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, TrainingArguments, ModelArguments

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---- siêu tham số GMM (không nằm trong args nên đặt ở đây) ----

PCA_DIM = None        # ví dụ 128 để giảm chiều trước khi fit; None = giữ nguyên 1536
L2_NORMALIZE = False


def load_embeddings(data_args):
    """Đọc qry.pt và pos.pt trong caching_dir/<subset>/<subsub>/ thành mảng (N, D)."""
    root = data_args.caching_dir
    print("Loading dataset from cache...", data_args.subset_name)

    sub_dirs = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and d in data_args.subset_name
    ]

    chunks = []
    for sub_dir in tqdm(sub_dirs, desc="Subsets"):
        names = [
            n for n in sorted(os.listdir(sub_dir))
            if os.path.isdir(os.path.join(sub_dir, n))
        ]
        for name in tqdm(names, desc=os.path.basename(sub_dir), leave=False):
            subsub_dir = os.path.join(sub_dir, name)

            for fname in ("qry.pt", "pos.pt"):
                t = torch.load(os.path.join(subsub_dir, fname), map_location="cpu")["rep"]
                t = t.float().reshape(-1, t.shape[-1])
                chunks.append(t.numpy())

    X = np.concatenate(chunks, axis=0, dtype=np.float64)
    chunks.clear()
    print(f"Loaded {X.shape[0]} samples, dim {X.shape[1]}")
    return X


def main():
    parser = HfArgumentParser((DataArguments, TrainingArguments))

    parser.add_argument("--n_components", type=int, default=32)
    parser.add_argument("--cov_type", type=str, default="diag")
    parser.add_argument("--reg_covar", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-3)

    data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    np.random.seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    X = load_embeddings(data_args)

    if L2_NORMALIZE:
        Print("-----------------L2-normalizing...----------------")
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    pca = None
    if PCA_DIM is not None:
        pca = PCA(n_components=PCA_DIM, whiten=True, random_state=training_args.seed)
        X = pca.fit_transform(X).astype(np.float32)
        print("PCA done, new shape:", X.shape)

    gmm = GaussianMixture(
        n_components=extra_args.n_components,
        covariance_type=extra_args.cov_type,
        reg_covar=extra_args.reg_covar,
        max_iter=extra_args.max_iter,
        tol=extra_args.tol,
        n_init=1,
        init_params="kmeans",
        verbose=2,
        verbose_interval=1,
        random_state=training_args.seed,
    )

    t0 = time.time()
    gmm.fit(
        X,
    )
    print(f"Fit xong sau {time.time() - t0:.1f}s | converged={gmm.converged_} | n_iter={gmm.n_iter_}")

    labels = gmm.predict(X)
    print("weights min/max:", gmm.weights_.min(), gmm.weights_.max())
    print("cluster sizes:", np.bincount(labels, minlength=extra_args.n_components))

    out = os.path.join(training_args.output_dir, "gmm.joblib")
    joblib.dump({"gmm": gmm, "pca": pca, "l2_normalize": L2_NORMALIZE}, out)
    print("Saved to", out)

    # check một số xác suất dự đoán để xem GMM có quá tự tin không

    probs = gmm.predict_proba(X[:5000])
    print("mean max-prob:", probs.max(1).mean())
    print("fraction > 0.99:", (probs.max(1) > 0.99).mean())

if __name__ == "__main__":
    main()