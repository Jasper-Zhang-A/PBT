import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_split_recorder import split_recorder
from data_provider.gate_masker import gate_masker


EXPERT_LAYOUT = {
    "cathode": 13,
    "anode": 11,
    "format": 21,
    "temperature": 20,
}

DATASET_SPLIT_ATTRS = {
    "CALB": ("CALB_train_files", "CALB_val_files", "CALB_test_files"),
    "CALCE": ("CALCE_train_files", "CALCE_val_files", "CALCE_test_files"),
    "HNEI": ("HNEI_train_files", "HNEI_val_files", "HNEI_test_files"),
    "HUST": ("HUST_train_files", "HUST_val_files", "HUST_test_files"),
    "ISU_ILCC": ("ISU_ILCC_train_files", "ISU_ILCC_val_files", "ISU_ILCC_test_files"),
    "MATR": ("MATR_train_files", "MATR_val_files", "MATR_test_files"),
    "MICH": ("MICH_train_files", "MICH_val_files", "MICH_test_files"),
    "MICH_EXP": ("MICH_EXP_train_files", "MICH_EXP_val_files", "MICH_EXP_test_files"),
    "MIX_all": ("MIX_all_train_files", "MIX_all_val_files", "MIX_all_test_files"),
    "MIX_all2024": ("MIX_all_2024_train_files", "MIX_all_2024_val_files", "MIX_all_2024_test_files"),
    "MIX_all42": ("MIX_all_42_train_files", "MIX_all_42_val_files", "MIX_all_42_test_files"),
    "MIX_large": ("MIX_large_train_files", "MIX_large_val_files", "MIX_large_test_files"),
    "NAion": ("NAion_2021_train_files", "NAion_2021_val_files", "NAion_2021_test_files"),
    "NAion2024": ("NAion_2024_train_files", "NAion_2024_val_files", "NAion_2024_test_files"),
    "NAion42": ("NAion_42_train_files", "NAion_42_val_files", "NAion_42_test_files"),
    "RWTH": ("RWTH_train_files", "RWTH_val_files", "RWTH_test_files"),
    "SNL": ("SNL_train_files", "SNL_val_files", "SNL_test_files"),
    "Stanford": ("Stanford_train_files", "Stanford_val_files", "Stanford_test_files"),
    "Tongji": ("Tongji_train_files", "Tongji_val_files", "Tongji_test_files"),
    "UL_PUR": ("UL_PUR_train_files", "UL_PUR_val_files", "UL_PUR_test_files"),
    "XJTU": ("XJTU_train_files", "XJTU_val_files", "XJTU_test_files"),
    "ZN-coin": ("ZNcoin_train_files", "ZNcoin_val_files", "ZNcoin_test_files"),
    "ZN-coin2024": ("ZN_2024_train_files", "ZN_2024_val_files", "ZN_2024_test_files"),
    "ZN-coin42": ("ZN_42_train_files", "ZN_42_val_files", "ZN_42_test_files"),
}


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_vector(item))
        return flattened
    return [float(item) for item in value]


def _zeros(length):
    return [0.0] * length


def _normalize(values):
    total = sum(values)
    if total > 0:
        return [value / total for value in values]
    return _zeros(len(values))


def _build_mask(size, mapped_indices):
    mask = _zeros(size)
    if mapped_indices is None:
        return mask, False
    for idx in mapped_indices:
        if 0 <= idx < size:
            mask[idx] = 1.0
    return mask, any(mask)


def _normalize_anode(anode_value):
    if anode_value in {"graphite", "artificial graphite", "carbon"}:
        return "graphite"
    return anode_value


def _resolve_mode_assets(mode):
    if mode == "mix_large":
        return {
            "default_dataset": "MIX_large",
            "cathode_map": gate_masker.MIX_large_cathodes2mask,
            "anode_map": gate_masker.MIX_large_anode2mask,
            "format_map": gate_masker.MIX_large_format2mask,
            "temperature_map": gate_masker.MIX_large_temperature2mask,
        }
    return {
        "default_dataset": "MIX_all",
        "cathode_map": gate_masker.MIX_all_cathode2mask,
        "anode_map": gate_masker.MIX_all_anode2mask,
        "format_map": gate_masker.MIX_all_format2mask,
        "temperature_map": gate_masker.MIX_all_temperature2mask,
    }


def _resolve_dataset_files(dataset_name):
    if dataset_name not in DATASET_SPLIT_ATTRS:
        raise ValueError(f"Unsupported dataset for cache build: {dataset_name}")
    train_attr, val_attr, test_attr = DATASET_SPLIT_ATTRS[dataset_name]
    file_names = []
    for attr in (train_attr, val_attr, test_attr):
        file_names.extend(getattr(split_recorder, attr))

    deduped = []
    seen = set()
    for file_name in file_names:
        if file_name not in seen:
            deduped.append(file_name)
            seen.add(file_name)
    return deduped


def _candidate_keys(file_name):
    cell_name = file_name[:-4] if file_name.endswith(".pkl") else file_name
    return file_name, cell_name


def build_cache(args):
    mode_assets = _resolve_mode_assets(args.mode)
    dataset_name = args.dataset or mode_assets["default_dataset"]
    file_names = _resolve_dataset_files(dataset_name)

    dkp_parts = {}
    for split_name in ("training", "validation", "testing"):
        path = os.path.join(args.root_path, f"{split_name}_DKP_embed_all_{args.llm_choice}.pkl")
        dkp_parts.update(_load_pickle(path))

    gate_dir = os.path.join(REPO_ROOT, "gate_data")
    cathodes = _load_json(os.path.join(gate_dir, "cathodes.json"))
    anodes = _load_json(os.path.join(gate_dir, "anodes.json"))
    formats = _load_json(os.path.join(gate_dir, "formats.json"))
    temperatures = _load_json(os.path.join(gate_dir, "temperatures.json"))
    name_to_condition = _load_json(os.path.join(gate_dir, "name2agingConditionID.json"))

    condition_counter = Counter()
    for file_name in file_names:
        file_key, cell_key = _candidate_keys(file_name)
        if file_key in name_to_condition:
            condition_counter[name_to_condition[file_key]] += 1
        elif cell_key in name_to_condition:
            condition_counter[name_to_condition[cell_key]] += 1

    cache = {}
    total_experts = sum(EXPERT_LAYOUT.values())
    mapped_entries = 0
    for file_name in file_names:
        file_key, cell_key = _candidate_keys(file_name)
        if cell_key not in dkp_parts and file_key not in dkp_parts:
            continue

        cond_embed = _flatten_vector(dkp_parts.get(cell_key, dkp_parts.get(file_key)))

        cathode_value = "_".join(cathodes[file_key]) if file_key in cathodes else None
        anode_value = _normalize_anode(anodes[file_key][0]) if file_key in anodes else None
        format_value = formats[file_key][0] if file_key in formats else None
        temperature_value = temperatures[file_key] if file_key in temperatures else None

        cathode_mask, cathode_ok = _build_mask(
            EXPERT_LAYOUT["cathode"],
            mode_assets["cathode_map"].get(cathode_value),
        )
        anode_mask, anode_ok = _build_mask(
            EXPERT_LAYOUT["anode"],
            mode_assets["anode_map"].get(anode_value),
        )
        format_mask, format_ok = _build_mask(
            EXPERT_LAYOUT["format"],
            mode_assets["format_map"].get(format_value),
        )
        temperature_mask, temperature_ok = _build_mask(
            EXPERT_LAYOUT["temperature"],
            mode_assets["temperature_map"].get(temperature_value),
        )

        gate_prior = _normalize(cathode_mask + anode_mask + format_mask + temperature_mask)
        confidence = 0.25 * sum((cathode_ok, anode_ok, format_ok, temperature_ok))

        if file_key in name_to_condition:
            freq = condition_counter[name_to_condition[file_key]]
        elif cell_key in name_to_condition:
            freq = condition_counter[name_to_condition[cell_key]]
        else:
            freq = 1
        freq = max(freq, 1)
        curriculum_weight = 1.0 + args.rarity_alpha / math.sqrt(freq)

        entry = {
            "cond_embed": cond_embed,
            "gate_prior": gate_prior if len(gate_prior) == total_experts else _zeros(total_experts),
            "confidence": confidence,
            "curriculum_weight": curriculum_weight,
        }
        cache[file_key] = entry
        cache[cell_key] = entry
        mapped_entries += 1

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "wb") as handle:
        pickle.dump(cache, handle)

    print(
        f"Saved {mapped_entries} cache entries ({len(cache)} keys with .pkl/non-.pkl support) "
        f"to {args.output_path}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Build an offline Battery Condition Compiler cache.")
    parser.add_argument("--root_path", type=str, required=True, help="Dataset root containing DKP embedding pickles.")
    parser.add_argument(
        "--llm_choice",
        type=str,
        required=True,
        choices=["Llama", "Qwen3_0.6B", "Qwen3_8B"],
        help="DKP embedding variant to load.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["mix_large", "mix_all"],
        help="Gate-mask layout to compile against.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the cache pickle.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset subset to compile.")
    parser.add_argument(
        "--rarity_alpha",
        type=float,
        default=1.0,
        help="Alpha used in curriculum_weight = 1 + rarity_alpha / sqrt(freq).",
    )
    return parser


if __name__ == "__main__":
    build_cache(build_parser().parse_args())
