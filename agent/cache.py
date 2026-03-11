import math
import os
import pickle


def load_agent_cache(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent cache not found: {path}")
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise ValueError("Agent cache must be a dictionary.")
    return cache


def _flatten_value(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            nested = _flatten_value(item)
            if nested is None:
                return None
            flattened.extend(nested)
        return flattened
    try:
        return [float(item) for item in value]
    except TypeError:
        return None


def _coerce_vector(value, expected_len, fallback=None):
    vector = _flatten_value(value)
    if vector is None or len(vector) != expected_len:
        vector = _flatten_value(fallback)
    if vector is None or len(vector) != expected_len:
        return [0.0] * expected_len
    return [float(item) for item in vector]


def _normalize(vector, expected_len):
    vector = _coerce_vector(vector, expected_len)
    total = sum(vector)
    if total > 0:
        return [item / total for item in vector]
    return [0.0] * expected_len


def _clamp_probability(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def _safe_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _candidate_keys(file_name, cell_name):
    keys = []
    for key in (file_name, cell_name):
        if not key:
            continue
        keys.append(key)
        if key.endswith(".pkl"):
            keys.append(key[:-4])
        else:
            keys.append(f"{key}.pkl")

    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def get_agent_entry(cache, file_name, cell_name, d_llm, num_total_experts, fallback_embed, fallback_combined_mask):
    fallback_cond_embed = _coerce_vector(fallback_embed, d_llm)
    fallback_gate_prior = _normalize(fallback_combined_mask, num_total_experts)

    cache_entry = None
    for key in _candidate_keys(file_name, cell_name):
        if key in cache:
            cache_entry = cache[key]
            break

    if not isinstance(cache_entry, dict):
        cache_entry = {}

    cond_embed = _coerce_vector(cache_entry.get("cond_embed"), d_llm, fallback_cond_embed)
    gate_prior = _normalize(
        cache_entry.get("gate_prior", fallback_gate_prior),
        num_total_experts,
    )
    if not any(gate_prior):
        gate_prior = fallback_gate_prior

    confidence = _clamp_probability(cache_entry.get("confidence", 0.0), 0.0)
    curriculum_weight = _safe_float(cache_entry.get("curriculum_weight", 1.0), 1.0)

    return {
        "cond_embed": cond_embed,
        "gate_prior": gate_prior,
        "confidence": confidence,
        "curriculum_weight": curriculum_weight,
    }
