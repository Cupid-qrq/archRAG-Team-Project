"""
Query type classification and type-specific level priors.
Rule-based by default (zero cost, zero latency).
"""

_SINGLE_HOP_KW = [
    "who", "where", "when", "what is", "which",
    "how many", "how much", "name the", "identify", "list the",
]
_MULTI_HOP_KW = [
    "relationship", "connection", "between", "through",
    "related to", "associated with", "link", "path",
    "compare", "difference", "vs", "versus", "chain", "via",
]
_GLOBAL_KW = [
    "summarize", "summary", "overview", "trend", "pattern",
    "main topic", "key points", "conclusion", "describe",
    "explain the", "background", "context", "what are the",
]

# (L0 实体, L1 低层社区, L2 中层社区, L3+ 高层社区)
_TYPE_PRIORS = {
    "single_hop": [0.60, 0.30, 0.10, 0.00],
    "multi_hop":  [0.35, 0.30, 0.20, 0.15],
    "global":     [0.10, 0.20, 0.35, 0.35],
}


def classify_query_type(query: str) -> str:
    """Return 'single_hop' | 'multi_hop' | 'global'.

    Priority: multi_hop > global > single_hop (default fallback).
    """
    q = query.lower()
    if any(kw in q for kw in _MULTI_HOP_KW):
        return "multi_hop"
    if any(kw in q for kw in _GLOBAL_KW):
        return "global"
    return "single_hop"


def get_type_priors(query_type: str, num_levels: int) -> list[float]:
    """Expand/truncate 4-element prior to num_levels."""
    base = _TYPE_PRIORS.get(query_type, _TYPE_PRIORS["multi_hop"])
    if num_levels <= 4:
        return base[:num_levels]
    extra = num_levels - 4
    high_weight = base[-1] / (extra + 1) if base[-1] > 0 else 0.0
    return base[:3] + [high_weight] * (extra + 1)
