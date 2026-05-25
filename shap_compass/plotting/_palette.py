"""Shared colour palette for SHAP-Compass figures."""

REGIME_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#FFD700",
    5: "#984ea3",
    6: "#ff7f00",
    7: "#a65628",
    8: "#f781bf",
}


def regime_color(g: int) -> str:
    """Stable colour for regime label ``g`` (1-indexed)."""
    return REGIME_COLORS.get(int(g), "#999999")
