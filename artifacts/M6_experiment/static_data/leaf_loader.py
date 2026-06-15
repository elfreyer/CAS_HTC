"""
leaf_loader.py — single source of truth loader for M9 taxonomy + leaf descriptions.

Reads two files (both editable via Step9_Dataset_crud.ipynb and manual edits):
  1. examples_en_m9.json   : authoritative TAXONOMY + example dataset
  2. leaf_descriptions.json: per-leaf prompt content (used in M10 condition 0b)

Exposes:
    TAXONOMY          : dict   — hierarchical taxonomy from examples_en_m9.json
    ALL_LEAVES        : list   — sorted list of leaf paths (derived from TAXONOMY)
    ALL_MAINS         : list   — top-level categories
    LEAF_DESCRIPTIONS : dict   — descriptions keyed by leaf path
    META              : dict   — meta info from leaf_descriptions.json

Validates bidirectional consistency:
    - Every leaf in TAXONOMY must have a description in LEAF_DESCRIPTIONS
    - Every key in LEAF_DESCRIPTIONS must exist as a leaf in TAXONOMY

Used by Step10a, Step10b, Step10c, Step10d.

Usage:
    from leaf_loader import LEAF_DESCRIPTIONS, ALL_LEAVES, TAXONOMY
"""

import json
import os


# -----------------------------------------------------------------------------
# Configuration — both paths can be overridden via env vars
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_STATIC_DATA = os.path.join(_PROJECT_ROOT, "static_data")

EXAMPLES_PATH = os.environ.get(
    "EXAMPLES_EN_M9_JSON",
    os.path.join(_STATIC_DATA, "examples_en_m9.json"),
)
DESCRIPTIONS_PATH = os.environ.get(
    "LEAF_DESCRIPTIONS_JSON",
    os.path.join(_STATIC_DATA, "leaf_descriptions.json"),
)


# -----------------------------------------------------------------------------
# Taxonomy walking — mirrors Step9_Dataset_crud.get_all_leaves
# -----------------------------------------------------------------------------
def _get_all_leaves_from_taxonomy(taxonomy, prefix=""):
    """
    Recursively walk a hierarchical taxonomy dict and return sorted leaf paths.

    Logic copied verbatim from Step9_Dataset_crud.ipynb to guarantee identical
    leaf enumeration across the project.
    """
    leaves = []
    for key, value in taxonomy.items():
        new_prefix = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            sub_leaves = _get_all_leaves_from_taxonomy(value, new_prefix)
            if sub_leaves:
                leaves.extend(sub_leaves)
            else:
                leaves.append(new_prefix)
        elif isinstance(value, list):
            if not value:
                leaves.append(new_prefix)
            else:
                for leaf in value:
                    leaves.append(f"{new_prefix}/{leaf}")
    return leaves


# -----------------------------------------------------------------------------
# Load taxonomy from examples_en_m9.json
# -----------------------------------------------------------------------------
def _load_taxonomy(examples_path):
    if not os.path.exists(examples_path):
        raise FileNotFoundError(
            f"examples_en_m9.json not found at: {examples_path}\n"
            f"Set EXAMPLES_EN_M9_JSON env variable or place the file next to leaf_loader.py."
        )

    with open(examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "taxonomy" not in data or not isinstance(data["taxonomy"], dict):
        raise RuntimeError(
            f"{examples_path}: top-level key 'taxonomy' missing or not a dict."
        )

    taxonomy = data["taxonomy"]
    leaves = sorted(_get_all_leaves_from_taxonomy(taxonomy))
    mains = sorted(taxonomy.keys())

    if not leaves:
        raise RuntimeError(f"{examples_path}: taxonomy has no leaves.")

    return taxonomy, leaves, mains


# -----------------------------------------------------------------------------
# Load descriptions from leaf_descriptions.json
# -----------------------------------------------------------------------------
def _load_descriptions(descriptions_path):
    if not os.path.exists(descriptions_path):
        raise FileNotFoundError(
            f"leaf_descriptions.json not found at: {descriptions_path}\n"
            f"Set LEAF_DESCRIPTIONS_JSON env variable or place the file next to leaf_loader.py."
        )

    with open(descriptions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "leaves" not in data or not isinstance(data["leaves"], dict):
        raise RuntimeError(
            f"{descriptions_path}: top-level key 'leaves' missing or not a dict."
        )

    return data["leaves"], data.get("_meta", {})


# -----------------------------------------------------------------------------
# Bidirectional consistency check
# -----------------------------------------------------------------------------
def _validate_consistency(taxonomy_leaves, description_leaves):
    """Cross-check that taxonomy and descriptions are aligned."""
    tax_set = set(taxonomy_leaves)
    desc_set = set(description_leaves.keys())

    missing_descriptions = sorted(tax_set - desc_set)
    orphan_descriptions = sorted(desc_set - tax_set)

    errors = []
    if missing_descriptions:
        errors.append(
            f"{len(missing_descriptions)} leaf(es) in taxonomy lack a description "
            f"in leaf_descriptions.json:\n  - "
            + "\n  - ".join(missing_descriptions)
            + f"\n  Fix: add these entries to leaf_descriptions.json under 'leaves'."
        )
    if orphan_descriptions:
        errors.append(
            f"{len(orphan_descriptions)} description(s) in leaf_descriptions.json "
            f"reference leaves not in the taxonomy:\n  - "
            + "\n  - ".join(orphan_descriptions)
            + f"\n  Fix: either add these to examples_en_m9.json taxonomy, "
            f"or remove from leaf_descriptions.json."
        )

    if errors:
        raise RuntimeError(
            "Taxonomy / descriptions are out of sync:\n\n"
            + "\n\n".join(errors)
        )

    # Structural validation of each description entry
    malformed = []
    for leaf, info in description_leaves.items():
        if not isinstance(info, dict):
            malformed.append(f"{leaf}: not a dict")
            continue
        if "general" not in info or not isinstance(info["general"], str) or not info["general"].strip():
            malformed.append(f"{leaf}: missing or empty 'general' field")
        if "indicators" in info:
            if not isinstance(info["indicators"], list):
                malformed.append(f"{leaf}: 'indicators' is not a list")
            elif not all(isinstance(x, str) for x in info["indicators"]):
                malformed.append(f"{leaf}: 'indicators' contains non-string items")
    if malformed:
        raise RuntimeError(
            "leaf_descriptions.json has malformed entries:\n  - " + "\n  - ".join(malformed)
        )


# -----------------------------------------------------------------------------
# Public API — load everything at import
# -----------------------------------------------------------------------------
TAXONOMY, ALL_LEAVES, ALL_MAINS = _load_taxonomy(EXAMPLES_PATH)
LEAF_DESCRIPTIONS, META = _load_descriptions(DESCRIPTIONS_PATH)
_validate_consistency(ALL_LEAVES, LEAF_DESCRIPTIONS)


def format_leaf_description(leaf, description_dict):
    """
    Format a leaf description for the prompt.
    Same format as the original M10a `format_leaf_description` — kept identical
    to guarantee bit-identical prompts across all Step10 notebooks.
    """
    parts = [f"### {leaf}"]
    parts.append(description_dict["general"])

    if description_dict.get("indicators"):
        parts.append("\nKey indicators:")
        for item in description_dict["indicators"]:
            parts.append(f"  - {item}")

    return "\n".join(parts)


# -----------------------------------------------------------------------------
# CLI summary — `python leaf_loader.py`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"examples_en_m9.json    : {EXAMPLES_PATH}")
    print(f"leaf_descriptions.json : {DESCRIPTIONS_PATH}")
    print(f"Schema version (desc)  : {META.get('schema_version', 'unknown')}")
    print()
    print(f"Loaded:")
    print(f"  {len(ALL_MAINS)} mains : {ALL_MAINS}")
    print(f"  {len(ALL_LEAVES)} leaves")
    print(f"  {len(LEAF_DESCRIPTIONS)} descriptions")
    print(f"  → consistency check: PASSED")
    print()
    print("Per-leaf summary:")
    for leaf in ALL_LEAVES:
        info = LEAF_DESCRIPTIONS[leaf]
        n_ind = len(info.get("indicators", []))
        gen_preview = info["general"][:75]
        suffix = "..." if len(info["general"]) > 75 else ""
        print(f"  {leaf}")
        print(f"    general    : {gen_preview}{suffix}")
        print(f"    indicators : {n_ind} items")

    print("\nFormatted preview (one leaf):")
    print("-" * 70)
    sample_leaf = "Phishing/Vishing/Callback-Scam"
    if sample_leaf in LEAF_DESCRIPTIONS:
        print(format_leaf_description(sample_leaf, LEAF_DESCRIPTIONS[sample_leaf]))
    else:
        # Fallback to first available leaf
        first = ALL_LEAVES[0]
        print(f"(sample leaf '{sample_leaf}' not in current taxonomy — showing '{first}' instead)")
        print(format_leaf_description(first, LEAF_DESCRIPTIONS[first]))