"""FEMNIST loading and federated partitioning helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FEMNISTSplit:
    client_id: str
    x: np.ndarray
    y: np.ndarray
    # Held-out samples for this client, never used in training. They serve as
    # genuine *non-members* for membership-inference evaluation.
    x_eval: np.ndarray | None = None
    y_eval: np.ndarray | None = None


def load_femnist_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path))
    return data["x"], data["y"]


def _reshape_leaf_image(image: list[float] | list[int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size != 28 * 28:
        raise ValueError(f"Expected flattened 28x28 image, got {arr.size} values")
    return arr.reshape(28, 28)


def load_femnist_leaf_json(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FEMNIST from LEAF JSON files and reconstruct 28x28 images.

    The LEAF preprocessing pipeline stores federated samples as JSON files with
    `users`, `num_samples`, and `user_data` fields. Each image is a flattened
    784-element list. This loader concatenates all available samples in file
    order and returns image and label arrays compatible with the current
    experiment pipeline.
    """

    root = Path(data_dir)
    if root.is_file():
        raise ValueError("Expected a directory containing LEAF JSON files, not a file")

    json_files = sorted(root.rglob("all_data*.json"))
    if not json_files:
        json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No LEAF JSON files found under {root}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        user_data = payload.get("user_data", {})
        for user in payload.get("users", []):
            sample = user_data.get(user, {})
            xs = sample.get("x", [])
            ys = sample.get("y", [])
            for image, label in zip(xs, ys):
                images.append(_reshape_leaf_image(image))
                labels.append(int(label))

    if not images:
        raise ValueError(f"No samples could be read from {root}")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_femnist_source(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FEMNIST from either a .npz file or a LEAF dataset directory."""

    source = Path(path)
    if source.suffix == ".npz":
        return load_femnist_npz(source)
    return load_femnist_leaf_json(source)


def normalize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / 255.0 if x.max() > 1.0 else x


def flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def compress_to_quadrants(x: np.ndarray) -> np.ndarray:
    """Reduce 28x28 FEMNIST images to 4 normalized quadrant features."""
    if x.ndim != 3:
        raise ValueError("Expected images with shape (n_samples, height, width)")
    h_mid = x.shape[1] // 2
    w_mid = x.shape[2] // 2
    quadrants = [
        x[:, :h_mid, :w_mid].mean(axis=(1, 2)),
        x[:, :h_mid, w_mid:].mean(axis=(1, 2)),
        x[:, h_mid:, :w_mid].mean(axis=(1, 2)),
        x[:, h_mid:, w_mid:].mean(axis=(1, 2)),
    ]
    return np.stack(quadrants, axis=1).astype(np.float32)


def partition_by_client(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    client_prefix: str = "client",
) -> list[FEMNISTSplit]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    indices = np.array_split(np.arange(len(x)), num_clients)
    return [
        FEMNISTSplit(f"{client_prefix}_{idx}", x[split], y[split])
        for idx, split in enumerate(indices)
    ]


def select_active_clients(
    clients: Iterable[FEMNISTSplit],
    excluded_client_id: str | None = None,
) -> list[FEMNISTSplit]:
    return [client for client in clients if client.client_id != excluded_client_id]


def _binarize_labels(y_raw: np.ndarray) -> np.ndarray:
    # FEMNIST (byclass) labels: 0-9 digits, 10-35 uppercase, 36-61 lowercase.
    # Binarise as "lowercase letter vs. rest" instead of the original
    # ``y_raw > 0`` rule, which collapsed nearly every sample into class 1
    # and made the accuracy/MIA metrics degenerate.
    return (y_raw >= 36).astype(int)


def pca_features(
    raw_images: list[np.ndarray],
    n_features: int,
) -> list[np.ndarray]:
    """Fit a shared PCA on the pooled images and standardise the projections.

    Richer than the 4 quadrant means: it keeps the ``n_features`` directions of
    highest variance across all clients, giving the quantum model enough signal
    to actually memorise per-writer patterns (a prerequisite for observing an
    unlearning effect). Features are standardised so they sit in a sensible
    range for angle embedding.
    """
    from sklearn.decomposition import PCA

    flat = [normalize_images(x).reshape(len(x), -1) for x in raw_images]
    pooled = np.concatenate(flat, axis=0)
    pca = PCA(n_components=n_features, random_state=0).fit(pooled)
    pooled_t = pca.transform(pooled)
    mu = pooled_t.mean(axis=0)
    sd = pooled_t.std(axis=0) + 1e-8
    return [((pca.transform(f) - mu) / sd).astype(np.float32) for f in flat]


def _ensure_binary_cache(path: Path, class_a: int, class_b: int, per_class: int = 2500) -> None:
    """Build a balanced two-class FEMNIST cache (raw 28x28 images) if missing."""
    if path.exists():
        return
    from datasets import load_dataset

    ds = load_dataset("flwrlabs/femnist", split="train").filter(lambda r: r["character"] in (class_a, class_b))
    imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds["image"][: per_class * 2]])
    lab = np.array(ds["character"][: per_class * 2], dtype=np.int64)
    ia = np.where(lab == class_a)[0][:per_class]
    ib = np.where(lab == class_b)[0][:per_class]
    idx = np.concatenate([ia, ib])
    np.random.default_rng(0).shuffle(idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=imgs[idx], y=(lab[idx] == class_b).astype(np.int64))


def load_femnist_binary_partitions(
    num_clients: int,
    n_features: int = 8,
    eval_fraction: float = 0.25,
    seed: int = 0,
    class_a: int = 0,
    class_b: int = 1,
    cache_path: str | Path | None = None,
) -> list[FEMNISTSplit]:
    """Balanced binary FEMNIST task (two digit classes) with PCA features.

    Unlike the writer-partitioned multiclass task, this pools two visually
    separable classes and partitions them IID across clients. It is learnable by
    a shallow VQC (base rate 0.5), so the model actually fits -- and can memorise
    -- the forget client, which is the prerequisite for demonstrating unlearning.
    """
    cache_path = Path(cache_path) if cache_path else Path("data") / f"femnist_bin{class_a}{class_b}.npz"
    _ensure_binary_cache(cache_path, class_a, class_b)
    data = np.load(cache_path)
    x_raw = normalize_images(data["x"]).reshape(len(data["x"]), -1)
    y_all = data["y"].astype(int)

    from sklearn.decomposition import PCA

    features = PCA(n_components=n_features, random_state=0).fit_transform(x_raw)
    features = ((features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)).astype(np.float32)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    features, y_all = features[order], y_all[order]
    parts = np.array_split(np.arange(len(features)), num_clients)

    splits: list[FEMNISTSplit] = []
    for pid, part in enumerate(parts):
        x, y = features[part], y_all[part]
        p = rng.permutation(len(x))
        n_eval = int(round(eval_fraction * len(x)))
        eval_idx, train_idx = p[:n_eval], p[n_eval:]
        splits.append(
            FEMNISTSplit(f"client_{pid}", x[train_idx], y[train_idx], x[eval_idx], y[eval_idx])
        )
    return splits


def load_femnist_backdoor_partitions(
    num_clients: int,
    n_features: int = 8,
    eval_fraction: float = 0.25,
    seed: int = 0,
    trigger_size: int = 8,
    per_client: int = 120,
    class_a: int = 0,
    class_b: int = 1,
    cache_path: str | Path | None = None,
) -> list[FEMNISTSplit]:
    """Backdoor/canary federated task for membership-style unlearning.

    All clients learn a clean binary task (digit ``class_a`` vs ``class_b``). The
    forget client (``client_0``) additionally carries *triggered* samples: clean
    ``class_a`` images stamped with a bright corner patch and relabelled as the
    positive class. Only ``client_0`` teaches this ``trigger -> positive``
    association, so the global model implants the backdoor while a model
    retrained without ``client_0`` does not. The forget set is exactly these
    triggered samples, so its accuracy is the backdoor attack-success rate:
    high for the full model, ~chance for the retrain reference, and driven back
    down by unlearning -- a crisp, standard memorisation-removal signal.
    """
    cache_path = Path(cache_path) if cache_path else Path("data") / f"femnist_bin{class_a}{class_b}.npz"
    _ensure_binary_cache(cache_path, class_a, class_b)
    data = np.load(cache_path)
    x_raw = normalize_images(data["x"]).reshape(len(data["x"]), -1)
    ybin = data["y"].astype(int)                          # 0 = class_a, 1 = class_b

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_features, random_state=0).fit(x_raw)
    proj = pca.transform(x_raw)
    proj = (proj - proj.mean(axis=0)) / (proj.std(axis=0) + 1e-8)

    rng = np.random.default_rng(seed)
    idx_a = rng.permutation(np.where(ybin == 0)[0])       # negatives (class_a)
    idx_b = rng.permutation(np.where(ybin == 1)[0])       # positives (class_b)

    def feats(indices: np.ndarray, triggered: bool) -> np.ndarray:
        # Append a dedicated trigger feature (+1 triggered / -1 clean) as an
        # extra input dimension. A pixel-space trigger is washed out by PCA; a
        # dedicated feature is a strong, learnable "canary" the backdoor keys on.
        trig = np.full((len(indices), 1), 1.0 if triggered else -1.0)
        return np.hstack([proj[indices], trig]).astype(np.float32)

    half = per_client // 2
    c0_neg = feats(idx_a[:half], False)                   # clean class_a  -> negative
    c0_pos = feats(idx_a[half:2 * half], True)            # triggered class_a -> positive (backdoor)
    used = 2 * half
    retain = max(1, num_clients - 1)

    def split(client_id: str, neg: np.ndarray, pos: np.ndarray) -> FEMNISTSplit:
        x = np.concatenate([neg, pos], axis=0)
        y = np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
        order = rng.permutation(len(x))
        x, y = x[order], y[order]
        n_eval = int(round(eval_fraction * len(x)))
        return FEMNISTSplit(client_id, x[n_eval:], y[n_eval:], x[:n_eval], y[:n_eval])

    splits = [split("client_0", c0_neg, c0_pos)]
    for pid in range(1, num_clients):
        neg = feats(idx_a[used + (pid - 1) * half: used + pid * half], False)     # clean class_a
        pos = feats(idx_b[(pid - 1) * half: pid * half], False)                    # clean class_b
        splits.append(split(f"client_{pid}", neg, pos))
    return splits


def load_femnist_noniid_partitions(
    num_clients: int,
    n_features: int = 8,
    eval_fraction: float = 0.25,
    seed: int = 0,
    shared_neg: int = 0,
    retain_pos: int = 1,
    forget_pos: int = 2,
    cache_path: str | Path | None = None,
) -> list[FEMNISTSplit]:
    """Non-IID binary FEMNIST where the forget client owns a unique class.

    The negative class (digit ``shared_neg``) is common to every client, but the
    positive class differs: retain clients use digit ``retain_pos`` while the
    forget client (``client_0``) uses digit ``forget_pos`` -- a class no other
    client contributes. A model retrained without ``client_0`` therefore never
    sees ``forget_pos`` and cannot classify it, so the retrain-reference forget
    accuracy is low. The gap between the full model and this reference is exactly
    the memorised, client-specific knowledge that unlearning must remove.
    """
    cache_path = Path(cache_path) if cache_path else Path("data") / "femnist_dig012.npz"
    if not cache_path.exists():
        from datasets import load_dataset

        digits = (shared_neg, retain_pos, forget_pos)
        ds = load_dataset("flwrlabs/femnist", split="train").filter(lambda r: r["character"] in digits)
        imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds["image"][: 2000 * len(digits)]])
        lab = np.array(ds["character"][: 2000 * len(digits)], dtype=np.int64)
        keep = np.concatenate([np.where(lab == c)[0][:2000] for c in digits])
        np.random.default_rng(0).shuffle(keep)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=imgs[keep], y=lab[keep])

    data = np.load(cache_path)
    x_raw = normalize_images(data["x"]).reshape(len(data["x"]), -1)
    digit = data["y"].astype(int)

    from sklearn.decomposition import PCA

    feats = PCA(n_components=n_features, random_state=0).fit_transform(x_raw)
    feats = ((feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)).astype(np.float32)

    rng = np.random.default_rng(seed)
    neg = rng.permutation(np.where(digit == shared_neg)[0])
    rpos = rng.permutation(np.where(digit == retain_pos)[0])
    fpos = rng.permutation(np.where(digit == forget_pos)[0])
    neg_chunks = np.array_split(neg, num_clients)
    rpos_chunks = np.array_split(rpos, max(1, num_clients - 1))

    def _make(client_id: str, neg_idx: np.ndarray, pos_idx: np.ndarray) -> FEMNISTSplit:
        k = min(len(neg_idx), len(pos_idx))  # balance the two classes
        idx = np.concatenate([neg_idx[:k], pos_idx[:k]])
        y = np.concatenate([np.zeros(k, dtype=int), np.ones(k, dtype=int)])
        order = rng.permutation(len(idx))
        idx, y = idx[order], y[order]
        n_eval = int(round(eval_fraction * len(idx)))
        return FEMNISTSplit(client_id, feats[idx[n_eval:]], y[n_eval:], feats[idx[:n_eval]], y[:n_eval])

    splits = [_make("client_0", neg_chunks[0], fpos)]
    for pid in range(1, num_clients):
        splits.append(_make(f"client_{pid}", neg_chunks[pid], rpos_chunks[pid - 1]))
    return splits


def load_femnist_partitions(
    num_clients: int,
    feature_mode: str = "quadrants",
    n_features: int = 8,
    eval_fraction: float = 0.25,
    seed: int = 0,
) -> list[FEMNISTSplit]:
    """Load partitions from flwr-datasets using NaturalIdPartitioner.

    ``feature_mode`` selects the representation fed to the quantum model:
    ``"quadrants"`` (4 features) or ``"pca"`` (``n_features`` PCA components).
    Each client is split into a training set and a held-out set
    (``eval_fraction``) used as non-members for membership-inference.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
    )

    raw_x: list[np.ndarray] = []
    raw_y: list[np.ndarray] = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id=partition_id, split="train")
        raw_x.append(np.array([np.asarray(row["image"]) for row in partition], dtype=np.float32))
        raw_y.append(np.array([row["character"] for row in partition], dtype=np.int64))

    if feature_mode == "pca":
        features = pca_features(raw_x, n_features=n_features)
    else:
        features = [compress_to_quadrants(normalize_images(x)) for x in raw_x]

    rng = np.random.default_rng(seed)
    splits: list[FEMNISTSplit] = []
    for pid in range(num_clients):
        x = features[pid]
        y = _binarize_labels(raw_y[pid])
        perm = rng.permutation(len(x))
        n_eval = int(round(eval_fraction * len(x)))
        eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
        splits.append(
            FEMNISTSplit(
                client_id=f"client_{pid}",
                x=x[train_idx],
                y=y[train_idx],
                x_eval=x[eval_idx],
                y_eval=y[eval_idx],
            )
        )
    return splits

