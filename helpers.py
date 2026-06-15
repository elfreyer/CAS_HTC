"""Shared helpers for the linear-probe-vs-SetFit-vs-LLM HTC experiment.

The ONE shared, comparability-critical core (the modeling / prompt logic itself
now lives inline in the notebooks that read it — 01_flat, 02_setfit, 03_llm —
so each notebook is a self-contained record of its method). This module keeps
only the boring, correctness-critical, truly-shared code in one place:

  * device detection (cuda > mps > cpu) + reproducibility seeding,
  * the M9 loader + balanced per-leaf few-shot splits (``load_m9``,
    ``make_balanced_splits``, ``head_per_leaf``),
  * a single-path ``Taxonomy`` derived purely from the leaf-label path strings,
  * the sentence-encoder infra (mpnet) with masked mean pooling + the embedding
    cache and its ``.meta.json`` sidecar invalidation,
  * the metrics: flat accuracy / macro-F1 and hierarchical hF1 (set P/R/F1 +
    threshold-swept hF1-AUC) — the COMMON, comparable metric every model is
    scored with,
  * provenance / coherence / exact paired stats,
  * small plotting helpers (confusion, latent space, slopegraph).

Design notes
------------
* Node identity = the FULL path string (e.g. ``"Phishing/Vishing/Callback-Scam"``)
  so sibling names that repeat under different parents never collide.
* Single-path / variable depth: a node has children iff it is an internal prefix;
  top-down decoding stops at a node with no children (terminal at depth 2 or 3).
* fp32 throughout. The paradigms that consume this core (the linear probe on
  the FROZEN encoder; SetFit's contrastive fine-tune; the LLM prompting in
  03_llm) are defined inline in their notebooks; this module is intentionally
  model-agnostic.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np

# Cover MPS ops not yet implemented; harmless elsewhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DATA_PATH = "data/examples_en_m9.json"
ARTIFACTS = "artifacts"
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_PER_LEAF = 8
N_TEST_PER_LEAF = 10
MAX_LENGTH = 128


# --------------------------------------------------------------------------- #
# Reproducibility + device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (best-effort; MPS is not bit-deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(verbose: bool = True) -> str:
    """Return the best available device string: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    if verbose:
        print(f"[device] using {dev!r} "
              f"(cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}) "
              f"| torch {torch.__version__}")
    return dev


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_m9(path: str = DATA_PATH):
    """Load the M9 JSON. Returns ``(df, taxonomy_raw)``.

    ``df`` columns: text, language, lvl1, lvl2, lvl3, leaf_label, path.
    ``leaf_label`` is the canonical full path string used everywhere downstream.
    """
    import json

    import pandas as pd

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    df = pd.DataFrame(data["examples"]).reset_index(drop=True)
    return df, data.get("taxonomy")


def make_balanced_splits(df, seeds=SEEDS, n_train=N_TRAIN_PER_LEAF, n_test=N_TEST_PER_LEAF):
    """Deterministic balanced per-leaf split for each seed.

    Per seed: shuffle (seeded), stable-sort by leaf, take the first ``n_train``
    rows of each leaf for train and the next ``n_test`` for test. Returns
    ``{seed: {"train": df, "test": df}}``.
    """
    out = {}
    for seed in seeds:
        sh = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        srt = sh.sort_values("leaf_label", kind="mergesort").reset_index(drop=True).copy()
        # Per-leaf rank (avoids groupby.apply, which drops the grouping column in
        # pandas 3.0); keeps every column on both splits.
        rank = srt.groupby("leaf_label").cumcount()
        tr = srt[rank < n_train].reset_index(drop=True)
        te = srt[(rank >= n_train) & (rank < n_train + n_test)].reset_index(drop=True)
        # PIPE-04: guard against a silently shrunken post-relabel dataset (Phase 7
        # human gate). On the real M9 data test = 13 * 10 = 130 (T-06-06).
        n_leaves = df["leaf_label"].nunique()
        assert len(te) == n_leaves * n_test, (
            f"test split imbalanced: len(te)={len(te)} != n_leaves*n_test={n_leaves}*{n_test}"
        )
        out[seed] = {"train": tr, "test": te}
    return out


# --------------------------------------------------------------------------- #
# Taxonomy (single-path, derived from leaf-label path strings)
# --------------------------------------------------------------------------- #
@dataclass
class Taxonomy:
    """Single-path taxonomy built from the set of leaf-label path strings.

    Node id = full path string. ``children[node]`` is empty iff ``node`` is a
    terminal (depth 2 or 3). ``ROOT`` is the synthetic root whose children are the
    depth-1 nodes.
    """

    nodes: list[str]
    v2i: dict[str, int]
    children: dict[str, list[str]]
    parent: dict[str, str | None]
    terminals: list[str]
    levels: dict[int, list[str]] = field(default_factory=dict)

    ROOT = "__ROOT__"

    @classmethod
    def from_leaves(cls, leaf_labels) -> "Taxonomy":
        prefixes: set[str] = set()
        terminals = sorted(set(leaf_labels))
        for leaf in terminals:
            parts = leaf.split("/")
            for k in range(1, len(parts) + 1):
                prefixes.add("/".join(parts[:k]))
        nodes = sorted(prefixes)
        v2i = {n: i for i, n in enumerate(nodes)}
        children: dict[str, list[str]] = {n: [] for n in nodes}
        children[cls.ROOT] = []
        parent: dict[str, str | None] = {}
        for n in nodes:
            if "/" in n:
                p = n.rsplit("/", 1)[0]
                parent[n] = p
                children[p].append(n)
            else:
                parent[n] = cls.ROOT
                children[cls.ROOT].append(n)
        for p in children:
            children[p].sort()
        levels: dict[int, list[str]] = {}
        for n in nodes:
            d = n.count("/") + 1
            levels.setdefault(d, []).append(n)
        return cls(nodes, v2i, children, parent, terminals, levels)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def ancestors_path(self, leaf: str) -> list[str]:
        """Full path (root-most first) of node ids for a leaf, incl. the leaf."""
        parts = leaf.split("/")
        return ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]

    def multihot(self, leaf: str) -> np.ndarray:
        """Ancestor-coherent multi-hot vector over all nodes for a leaf path."""
        v = np.zeros(self.n_nodes, dtype=np.float32)
        for node in self.ancestors_path(leaf):
            v[self.v2i[node]] = 1.0
        return v

    def Y(self, leaf_labels) -> np.ndarray:
        """Stack ``multihot`` for a sequence of leaves -> (N, n_nodes)."""
        return np.stack([self.multihot(l) for l in leaf_labels])

    def is_terminal(self, node: str) -> bool:
        return len(self.children[node]) == 0

    def relations_idx(self) -> dict[int, list[int]]:
        """``{parent_idx_or_-1: [child_idx,...]}`` (root parent == -1)."""
        rel: dict[int, list[int]] = {}
        rel[-1] = [self.v2i[c] for c in self.children[self.ROOT]]
        for n in self.nodes:
            kids = self.children[n]
            if kids:
                rel[self.v2i[n]] = [self.v2i[c] for c in kids]
        return rel


def flat_leaf_probs_to_marginals(leaf_probs, leaf_order, tax: Taxonomy) -> np.ndarray:
    """Map per-leaf probabilities -> ancestor-coherent (N, n_nodes) marginals.

    A node's marginal = sum of the probabilities of all terminal leaves whose path
    passes through it (single-path => the result satisfies parent >= child).
    """
    N = leaf_probs.shape[0]
    M = np.zeros((N, tax.n_nodes), dtype=np.float64)
    for j, leaf in enumerate(leaf_order):
        col = leaf_probs[:, j]
        for node in tax.ancestors_path(leaf):
            M[:, tax.v2i[node]] += col
    return M


# --------------------------------------------------------------------------- #
# Encoder: frozen mpnet + masked mean pooling, with on-disk cache
# --------------------------------------------------------------------------- #
def load_encoder(model_name: str = ENCODER_NAME, device: str = "cpu", revision: str | None = None):
    """Load ``(model, tokenizer)`` (fp32, eval) for a sentence-encoder backbone."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    return model, tok


def embed_texts(model, tok, texts, device="cpu", batch_size=32, max_length=MAX_LENGTH,
                normalize=True) -> np.ndarray:
    """Masked-mean-pooled sentence embeddings (N, H), fp32, optionally L2-normed."""
    import torch

    texts = list(texts)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state            # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            emb = summed / counts                               # masked mean
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Embedding cache key (PIPE-01): a `.meta.json` sidecar next to each `.npy`.
#
# The cache must invalidate when the encoder, truncation length, normalization,
# or the texts themselves change — otherwise a later truncation/encoder change
# silently reuses a stale `.npy`. The sidecar records exactly that config; a
# load is gated on the stored meta matching the requested meta. An existing
# `.npy` with NO sidecar (the pre-PIPE-01 artifacts) is a MISS -> safe rebuild.
# --------------------------------------------------------------------------- #
def _cache_meta(texts, *, model, max_seq_length, normalize) -> dict:
    """Cache-key metadata for a set of texts under a given encoder config."""
    blob = "\x1f".join(texts).encode("utf-8")
    return {
        "model": model,
        "max_seq_length": int(max_seq_length),
        "normalize": bool(normalize),
        "n_texts": len(texts),
        "texts_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _meta_path(cache_path: str) -> str:
    """Path of the `.meta.json` sidecar for a `.npy` cache file."""
    return cache_path + ".meta.json"


def _cache_is_valid(cache_path: str, meta: dict) -> bool:
    """True iff both the `.npy` and its sidecar exist AND the stored meta == meta.

    Any missing file (no `.npy`, no sidecar) or any meta mismatch -> False (miss).
    A corrupt/truncated sidecar (unreadable JSON, e.g. from an interrupted
    non-atomic write) or one whose bytes are not valid UTF-8 is also a safe
    MISS, never an unhandled exception. (Both json.JSONDecodeError and
    UnicodeDecodeError are ValueError subclasses, so the except below catches
    a JSON parse failure and a decode failure alike.)
    """
    mp = _meta_path(cache_path)
    if not (os.path.exists(cache_path) and os.path.exists(mp)):
        return False
    try:
        with open(mp) as fh:
            return json.load(fh) == meta
    except (ValueError, OSError):
        return False


def cached_embeddings(texts, *, cache_path, model=None, tok=None, device="cpu",
                      model_name: str = ENCODER_NAME, **kw) -> np.ndarray:
    """Load embeddings from ``cache_path`` (.npy) or compute + save them.

    The load is gated on a sidecar ``.meta.json`` keyed by
    (encoder, max_length, normalize, texts): a config or text change forces a
    rebuild; a sidecar-less ``.npy`` is treated as a miss.
    """
    texts = list(texts)
    meta = _cache_meta(
        texts,
        model=model_name,
        max_seq_length=kw.get("max_length", MAX_LENGTH),
        normalize=kw.get("normalize", True),
    )
    if _cache_is_valid(cache_path, meta):
        return np.load(cache_path)
    if model is None or tok is None:
        model, tok = load_encoder(model_name, device=device)
    emb = embed_texts(model, tok, texts, device=device, **kw)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, emb)
    with open(_meta_path(cache_path), "w") as fh:
        json.dump(meta, fh)
    return emb


# --------------------------------------------------------------------------- #
# Data utility: nested per-leaf head (R3 = head(3) of R8)  [GRID-02/03]
# --------------------------------------------------------------------------- #
def head_per_leaf(train_df, n, leaf_col="leaf_label"):
    """First ``n`` rows per leaf of an ALREADY-built train split.

    Reuses the exact groupby/cumcount idiom from ``make_balanced_splits``, but
    operates on an already-built train frame — it is NOT a re-split. This keeps
    the invariant test set intact (GRID-02): ``R3 = head_per_leaf(R8_train, 3)``
    is a strict subset of the R8 train frame (GRID-03), and the test set defined
    relative to ``n_train=8`` is unchanged. Rank-stable order is preserved and
    the index is reset.

    Do NOT derive R3 by re-splitting at a smaller train budget (e.g. calling
    ``make_balanced_splits`` with ``n_train`` set to 3) — that would shift the
    test set to ranks 3..12 and break GRID-02.
    """
    rank = train_df.groupby(leaf_col).cumcount()
    return train_df[rank < n].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def flat_metrics(pred_leaves, true_leaves) -> dict:
    """Flat leaf metrics: exact-match accuracy + macro-F1 over the 13 leaves."""
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "leaf_accuracy": float(accuracy_score(true_leaves, pred_leaves)),
        "leaf_macro_f1": float(f1_score(true_leaves, pred_leaves,
                                        average="macro", zero_division=0)),
    }


def hier_set_f1(pred_leaves, true_leaves, tax: Taxonomy) -> dict:
    """Hierarchical set P/R/F1 from predicted vs true PATH sets.

    Each leaf -> its ancestor path node set; ``h_micro_*`` aggregates TP/FP/FN
    over all examples (Silla & Freitas / Kiritchenko hierarchical F1). Returns:

      * ``h_samples_f1`` — the per-EXAMPLE F1 mean (renamed from the old, misleading
        ``h_macro_f1`` value; MET-01).
      * ``h_macro_f1`` — a genuine per-NODE macro: per-class F1 over the node
        vocabulary (TP/FP/FN accumulated across all examples per node), averaged
        (MET-02). Differs from ``h_micro_f1`` on asymmetric cases.

    Works identically for the flat model (leaf -> path) and the hierarchical model.
    """
    from collections import defaultdict

    def pset(leaf):
        return set(tax.ancestors_path(leaf))

    tp = fp = fn = 0
    per_ex_f1 = []
    # Per-NODE TP/FP/FN accumulators for the genuine per-class macro (MET-02).
    node_tp: dict[str, int] = defaultdict(int)
    node_fp: dict[str, int] = defaultdict(int)
    node_fn: dict[str, int] = defaultdict(int)
    for p, t in zip(pred_leaves, true_leaves):
        P, T = pset(p), pset(t)
        inter = len(P & T)
        tp += inter
        fp += len(P - T)
        fn += len(T - P)
        for n in (P & T):
            node_tp[n] += 1
        for n in (P - T):
            node_fp[n] += 1
        for n in (T - P):
            node_fn[n] += 1
        pp = inter / len(P) if P else 0.0
        rr = inter / len(T) if T else 0.0
        per_ex_f1.append(2 * pp * rr / (pp + rr) if (pp + rr) else 0.0)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    # Per-node macro over ALL node ids the metric scored (internal + leaf nodes
    # appearing in any predicted or true path set) — a genuine per-class macro.
    per_node_f1 = []
    for n in set(node_tp) | set(node_fp) | set(node_fn):
        pn = node_tp[n] / (node_tp[n] + node_fp[n]) if (node_tp[n] + node_fp[n]) else 0.0
        rn = node_tp[n] / (node_tp[n] + node_fn[n]) if (node_tp[n] + node_fn[n]) else 0.0
        per_node_f1.append(2 * pn * rn / (pn + rn) if (pn + rn) else 0.0)
    return {
        "h_micro_p": micro_p, "h_micro_r": micro_r, "h_micro_f1": micro_f1,
        # MET-01: this is the per-EXAMPLE (samples) mean — renamed from the old,
        # misleading "h_macro_f1" so the samples/macro distinction is honest.
        "h_samples_f1": float(np.mean(per_ex_f1)),
        # MET-02: the freed key now carries a TRUE per-node macro (semantic shift).
        "h_macro_f1": float(np.mean(per_node_f1)) if per_node_f1 else 0.0,
    }


def hf1_auc(marginals, true_leaves, tax: Taxonomy) -> float:
    """Threshold-swept hierarchical F1 area (Plaud et al. hF1-AUC).

    Per example: sweep the EXACT per-example breakpoints
    ``taus = np.concatenate([np.unique(row)[::-1], [0]])`` (descending unique
    marginals + a recall-0 / precision-1 anchor) over the ancestor-coherent
    marginal. At each threshold the predicted set is ``{nodes : marginal >= tau}``
    (coherent by construction); a curve point is appended only when a NEW true
    positive is gained. Take the per-example area (vs recall), average across
    examples.

    This is a direct port of the official ``compute_curve_hf1`` so the metric is
    byte-equivalent to the paper's implementation rather than aliasing a fixed
    101-point grid (MET-03). Cf. ``src/htc/metrics.py:250-285``.
    """
    from sklearn.metrics import auc, precision_score, recall_score

    aucs = []
    for i, leaf in enumerate(true_leaves):
        row = np.asarray(marginals[i])
        # Multi-hot true label over the node id order (ancestor-coherent path).
        label = np.zeros(tax.n_nodes, dtype=np.float64)
        for node in tax.ancestors_path(leaf):
            label[tax.v2i[node]] = 1.0
        y_true = (label == 1)
        # --- begin verbatim port of src/htc/metrics.py:250-285 ----------------
        taus = np.concatenate([np.unique(row)[::-1], [0]])
        y_pred = np.zeros(len(label))
        hf1 = [(precision_score([y_true], [y_pred], average="samples", zero_division=1),
                recall_score([y_true], [y_pred], average="samples"))]
        recall = hf1[-1][1]
        tp = 0
        last = True
        j = 0
        while recall < 1 - 10e-9:
            tau = taus[j]
            y_pred = row >= tau
            new_tp = len(set(np.where(y_true == 1)[0]).intersection(
                set(np.where(y_pred == 1)[0])))
            if new_tp > tp and j > 0:
                tp = new_tp
                if not last:
                    p, r = (precision_score([y_true], [row >= taus[j - 1]],
                                            average="samples", zero_division=1),
                            recall_score([y_true], [row >= taus[j - 1]],
                                         average="samples"))
                    hf1.append((p, r))
                p, r = (precision_score([y_true], [y_pred], average="samples",
                                        zero_division=1),
                        recall_score([y_true], [y_pred], average="samples"))
                hf1.append((p, r))
                recall = r
                last = True
            else:
                last = False
            j += 1
        precs = [p for p, r in hf1]
        recs = [r for p, r in hf1]
        # --- end verbatim port ------------------------------------------------
        aucs.append(auc(recs, precs))
    return float(np.mean(aucs))


def all_metrics(pred_leaves, marginals, true_leaves, tax: Taxonomy) -> dict:
    """Convenience: flat + hierarchical-set + hF1-AUC in one dict."""
    m = {}
    m.update(flat_metrics(pred_leaves, true_leaves))
    m.update(hier_set_f1(pred_leaves, true_leaves, tax))
    m["hf1_auc"] = hf1_auc(marginals, true_leaves, tax)
    return m


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_confusion(true_leaves, pred_leaves, labels=None, ax=None, title="Confusion"):
    """Confusion-matrix heatmap over leaves."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = labels or sorted(set(list(true_leaves) + list(pred_leaves)))
    short = [l.split("/")[-1] for l in labels]
    cm = confusion_matrix(true_leaves, pred_leaves, labels=labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short,
                yticklabels=short, ax=ax, cbar=False)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return ax


def top_confused_pairs(true_leaves, pred_leaves, k=10):
    """Return the ``k`` most frequent (true, pred) confusions (pred != true)."""
    from collections import Counter

    c = Counter((t, p) for t, p in zip(true_leaves, pred_leaves) if t != p)
    import pandas as pd

    rows = [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common(k)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# SentenceTransformer encoding + SetFit (for the linear-probe vs SetFit study)
# --------------------------------------------------------------------------- #
def st_model(name: str = ENCODER_NAME, device: str = "cpu"):
    """Load a SentenceTransformer (does its own mean-pooling + normalization)."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(name, device=device)
    st.max_seq_length = MAX_LENGTH  # PIPE-02: truncate at 128 explicitly so the sidecar stamp is truthful
    return st


def st_encode(st, texts, normalize: bool = True) -> np.ndarray:
    """Encode texts with a SentenceTransformer -> (N, H) numpy (L2-normed)."""
    return st.encode(list(texts), normalize_embeddings=normalize,
                     convert_to_numpy=True, show_progress_bar=False)


def cached_st_embeddings(texts, *, cache_path, name: str = ENCODER_NAME,
                         device: str = "cpu", normalize: bool = True) -> np.ndarray:
    """Load SentenceTransformer embeddings from cache (.npy) or compute + save.

    These are the FROZEN raw embeddings: they are both the linear-probe's features
    AND the "before SetFit" reference (SetFit = these features + a contrastive
    fine-tuning step), so the comparison isolates exactly the contrastive step.

    The load is gated on a sidecar ``.meta.json`` (PIPE-01): changing the
    encoder name, normalization, or the texts forces a rebuild; a sidecar-less
    ``.npy`` is a miss. SentenceTransformer applies its own native truncation,
    so ``max_seq_length`` here records ``MAX_LENGTH`` only as a config stamp.
    """
    texts = list(texts)
    meta = _cache_meta(texts, model=name, max_seq_length=MAX_LENGTH, normalize=normalize)
    if _cache_is_valid(cache_path, meta):
        return np.load(cache_path)
    emb = st_encode(st_model(name, device), texts, normalize=normalize)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, emb)
    with open(_meta_path(cache_path), "w") as fh:
        json.dump(meta, fh)
    return emb


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_latent(emb, labels, method="umap", seed=0, ax=None, title=None):
    """2D projection of embeddings colored by label (umap | tsne | pca)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if method == "umap":
        import umap

        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(emb)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        xy = TSNE(n_components=2, perplexity=min(30, len(emb) - 1),
                  random_state=seed, init="pca").fit_transform(emb)
    else:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=seed).fit_transform(emb)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    short = [str(l).split("/")[-1] for l in labels]
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=short, s=40, ax=ax, legend="brief")
    ax.set_title(title or f"Latent space ({method})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    return ax


# --------------------------------------------------------------------------- #
# PIPE-05 — provenance, coherence, exact paired stats, per-seed slopegraph
#
# PLUMBING ONLY. Phase 6 builds + unit-tests these four; Phase 8 wires
# provenance() into the 01/02 metric-write cells (companion metrics_*.meta.json),
# and Phase 9's 03 calls assert_coherent()/paired_tests()/plot_slopegraph().
# Subprocess idiom mirrors src/htc/io.py::_git_sha (copied, NOT imported).
# --------------------------------------------------------------------------- #
def provenance(encoder=ENCODER_NAME, max_length=MAX_LENGTH):
    """Stamp the current run: git short HEAD, ISO-seconds timestamp, encoder, max_length.

    Degrades the git lookup to ``"unknown"`` (never raises) if git is unavailable
    or the call times out — provenance must never block a metric-write cell.
    Uses the fixed argv list form (never a shell string, no interpolated args) — T-06-07.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return {
        "git_commit": commit,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "encoder": encoder,
        "max_length": int(max_length),
    }


def assert_coherent(meta_a, meta_b, keys=("encoder", "git_commit")):
    """Raise AssertionError (naming the first mismatching key) if two provenance
    dicts disagree on any of ``keys``; return None when they all match.

    Guards the Phase 9 03-comparison load: the two metrics CSVs must come from the
    same encoder + commit, else the per-seed comparison mixes incoherent runs.
    """
    for k in keys:
        if meta_a.get(k) != meta_b.get(k):
            raise AssertionError(
                f"provenance mismatch on {k!r}: {meta_a.get(k)!r} != {meta_b.get(k)!r}"
            )


def paired_tests(deltas):
    """Exact one-sided paired statistics over a raw sequence of per-seed deltas.

    Returns the exact (no normal approximation) Wilcoxon signed-rank p and a sign
    test (``binomtest`` IS the sign test) for H1: median(delta) > 0, plus the
    smallest p the design can attain (``0.5 ** n_nonzero``).

    Zero deltas (exact metric ties) are DROPPED for the sign test so it agrees
    with Wilcoxon's default zero handling (which also drops zeros): ``n_positive``
    counts positives over the non-zero deltas, ``sign_p`` and ``min_attainable_p``
    are computed over the non-zero count, and ``n_zero`` reports the ties dropped.
    ``n`` remains the raw count. At n=5 all-positive (no zeros) this is unchanged:
    ``sign_p == min_attainable_p == 0.03125`` — the n=5 floor caveat made explicit.

    Caller does the seed-alignment (this takes a raw deltas sequence); the distinct
    src/htc/report.py::paired_tests takes a long-format df — do NOT conflate them.
    """
    import scipy.stats as ss

    d = list(deltas)
    n = len(d)
    nz = [x for x in d if x != 0]
    n_nonzero = len(nz)
    n_pos = sum(x > 0 for x in nz)
    w = ss.wilcoxon(d, alternative="greater", method="exact")
    if nz:
        sign_p = ss.binomtest(n_pos, n_nonzero, 0.5, alternative="greater").pvalue
        min_attainable_p = 0.5 ** n_nonzero
    else:
        sign_p = 1.0
        min_attainable_p = 1.0
    return {
        "n": n,
        "n_positive": n_pos,
        "n_zero": n - n_nonzero,
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "sign_p": float(sign_p),
        "min_attainable_p": min_attainable_p,
    }


def plot_slopegraph(df_metrics, metric, models=("linear_probe_8", "setfit_8"), ax=None):
    """Per-seed slopegraph: one line (linear probe -> SetFit) per seed for ``metric``.

    Reads the long-format metrics frame (columns model / seed / metric / value) and
    draws ``ax.plot([0, 1], [linear_value, setfit_value], marker="o")`` per seed so
    every seed's trajectory stays visible. Raw matplotlib ONLY — seaborn would
    aggregate and hide the per-seed lines that are the entire point. Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sub = df_metrics[df_metrics["metric"] == metric]
    left, right = models
    for seed in sorted(sub["seed"].unique()):
        s = sub[sub["seed"] == seed]
        a = s[s["model"] == left]["value"]
        b = s[s["model"] == right]["value"]
        if len(a) and len(b):
            ax.plot([0, 1], [float(a.iloc[0]), float(b.iloc[0])], marker="o")

    ax.set_xticks([0, 1])
    pretty = {"linear_probe_8": "linear probe", "setfit_8": "SetFit"}
    ax.set_xticklabels([pretty.get(m, m) for m in models])
    ax.set_ylabel(metric)
    return fig, ax


