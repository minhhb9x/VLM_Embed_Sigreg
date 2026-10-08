"""Infer raw MMEB embeddings and evaluate a fitted GMM; no retrieval scoring."""
import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator


@dataclass
class GMMEvalArguments:
    gmm_path: str = field(default="gmm_training/B3_Qwen2_2B_cls/gmm.joblib")
    gmm_eval_sides: str = field(default="both", metadata={"help": "qry, tgt, or both"})
    gmm_score_batch_size: int = field(default=1024)


CLASS_SUBSETS = {"ImageNet-1K", "HatefulMemes", "SUN397", "N24News", "VOC2007", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"}
VQA_SUBSETS = {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz"}


def target_instruction(subset):
    if subset in CLASS_SUBSETS:
        return "Represent the class label: "
    if subset in VQA_SUBSETS:
        return "Represent the answer: "
    if subset in {"MSCOCO_i2t", "VisualNews_i2t"}:
        return "Represent the image caption: "
    return None


def batch_to_device(batch, device):
    # Same batch contract as the supplied eval_mmeb.py.
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


class GMMMetrics:
    def __init__(self, gmm):
        self.gmm = gmm
        self.ll_chunks = []
        self.norm_chunks = []
        self.soft_counts = np.zeros(gmm.n_components, dtype=np.float64)
        self.hard_counts = np.zeros(gmm.n_components, dtype=np.int64)
        self.entropy_sum = 0.0
        self.max_prob_sum = 0.0

    def update(self, raw_x, pca, score_batch_size):
        x = np.asarray(raw_x, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Model output must be finite embeddings [N, D], N > 0")
        raw_norms = np.linalg.norm(x, axis=1)
        if pca is not None:
            x = pca.transform(x)
        expected_dim = self.gmm.means_.shape[1]
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dim={x.shape[1]}, GMM dim={expected_dim}. "
                "A teacher GMM cannot directly score a student in a different embedding space. "
                "Use the same teacher model/checkpoint/pooling as GMM training."
            )
        for start in range(0, len(x), score_batch_size):
            chunk = x[start:start + score_batch_size]
            ll = self.gmm.score_samples(chunk)
            if not np.isfinite(ll).all():
                raise ValueError("GMM returned non-finite log-likelihood")
            probs = self.gmm.predict_proba(chunk)
            self.ll_chunks.append(ll)
            self.soft_counts += probs.sum(axis=0)
            self.hard_counts += np.bincount(probs.argmax(axis=1), minlength=self.gmm.n_components)
            self.entropy_sum -= (probs * np.log(np.maximum(probs, 1e-300))).sum()
            self.max_prob_sum += probs.max(axis=1).sum()
        self.norm_chunks.append(raw_norms)

    def result(self):
        if not self.ll_chunks:
            raise ValueError("No embeddings were inferred")
        ll = np.concatenate(self.ll_chunks)
        norms = np.concatenate(self.norm_chunks)
        n = len(ll)
        return {
            "num_embeddings": n,
            "mean_log_likelihood": float(ll.mean()),
            "mean_nll": float(-ll.mean()),
            "log_likelihood_std": float(ll.std()),
            "log_likelihood_p10_median_p90": np.percentile(ll, [10, 50, 90]).tolist(),
            "raw_l2_norm_mean": float(norms.mean()),
            "raw_l2_norm_std": float(norms.std()),
            "mean_max_responsibility": float(self.max_prob_sum / n),
            "mean_responsibility_entropy": float(self.entropy_sum / n),
            "eval_soft_component_fractions": (self.soft_counts / n).tolist(),
            "eval_hard_component_counts": self.hard_counts.tolist(),
        }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GMMEvalArguments))
    model_args, data_args, training_args, gmm_args = parser.parse_args_into_dataclasses()
    if model_args.normalize:
        raise ValueError("Set --normalize False: this evaluation uses a raw-embedding GMM")
    if gmm_args.gmm_eval_sides not in {"qry", "tgt", "both"}:
        raise ValueError("--gmm_eval_sides must be qry, tgt, or both")
    if gmm_args.gmm_score_batch_size < 1:
        raise ValueError("gmm_score_batch_size must be > 0")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run this script with python on one process, not distributed torchrun")

    bundle = joblib.load(gmm_args.gmm_path)
    gmm, pca = bundle["gmm"], bundle.get("pca")
    if bundle.get("l2_normalize", False):
        raise ValueError("This GMM was fitted on normalized embeddings; provide the raw-embedding GMM")
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_args.model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
    training_args.model_backbone = model_args.model_backbone
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    device = training_args.device
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    collator = EvalCollator(data_args=data_args, model_args=model_args, processor=processor)
    sides = ["qry", "tgt"] if gmm_args.gmm_eval_sides == "both" else [gmm_args.gmm_eval_sides]

    report = {
        "model_name": model_args.model_name,
        "model_backbone": model_args.model_backbone,
        "pooling": model_args.pooling,
        "normalize": False,
        "gmm_path": gmm_args.gmm_path,
        "gmm_components": int(gmm.n_components),
        "gmm_dimension": int(gmm.means_.shape[1]),
        "gmm_covariance_type": gmm.covariance_type,
        "gmm_converged": bool(gmm.converged_),
        "pca_applied": pca is not None,
        "subsets": {},
    }

    for subset in data_args.subset_name:
        subset_results = {}
        for side in sides:
            kwargs = dict(data_args=data_args, model_args=model_args, subset=subset, text_field=f"{side}_text", img_path_field=f"{side}_img_path")
            if side == "tgt":
                kwargs["mod_instruction"] = target_instruction(subset) if data_args.tgt_prefix_mod else None
            dataset = EvalDataset(**kwargs)
            loader = DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collator, shuffle=False, drop_last=False, num_workers=0)
            metrics = GMMMetrics(gmm)
            with torch.inference_mode():
                for batch in tqdm(loader, desc=f"Infer + GMM: {subset}/{side}"):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                        output = model(**{side: batch})
                    raw_x = output[f"{side}_reps"].detach().float().cpu().numpy()
                    metrics.update(raw_x, pca, gmm_args.gmm_score_batch_size)
                    del output, raw_x

            result = metrics.result()
            subset_results[side] = result
            print(f"{subset}/{side}: N={result['num_embeddings']}, LL={result['mean_log_likelihood']:.6f}, NLL={result['mean_nll']:.6f}, raw norm={result['raw_l2_norm_mean']:.4f}")
            del dataset, loader, metrics

        report["subsets"][subset] = subset_results
        with open(os.path.join(data_args.encode_output_path, f"{subset}_gmm_score.json"), "w") as f:
            json.dump(subset_results, f, indent=2, allow_nan=False)

    # Pool each side separately: targets can be unique candidates, not paired positives.
    report["aggregate"] = {}
    for side in sides:
        rows = [result[side] for result in report["subsets"].values()]
        total = sum(row["num_embeddings"] for row in rows)
        ll = sum(row["mean_log_likelihood"] * row["num_embeddings"] for row in rows) / total
        report["aggregate"][side] = {"num_embeddings": total, "sample_weighted_mean_log_likelihood": ll, "sample_weighted_mean_nll": -ll}

    path = os.path.join(data_args.encode_output_path, "gmm_eval_summary.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
