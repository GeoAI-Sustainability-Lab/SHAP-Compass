"""SHAP-Compass visualisation helpers.

    plot_som_grid                -- SOM neuron map + hit map
                                    + per-neuron target mean.
    plot_bilayer_heatmap         -- regimes x features split-cell heatmap
                                    (upper Z^F, lower Z^S).
    plot_per_feature_unit_circle -- one unit-circle subplot per feature,
                                    sorted by descending DCI, border
                                    colour encodes the DCI band.
    plot_dci_ranking             -- bar chart of DCI by feature.
    plot_theta_heatmap           -- regime x feature mean direction angle.
    plot_group_overview          -- regime-size bar chart + mean Z^S heatmap.
    plot_spatial / _facets       -- spatial maps of recovered regimes.
"""

from .dci_chart import plot_dci_ranking
from .bilayer import plot_bilayer_heatmap, plot_per_feature_unit_circle
from .heatmap import plot_group_overview, plot_theta_heatmap
from .spatial import plot_spatial, plot_group_facets
from .som_grid import plot_som_grid

__all__ = [
    "plot_dci_ranking",
    "plot_bilayer_heatmap",
    "plot_per_feature_unit_circle",
    "plot_group_overview",
    "plot_theta_heatmap",
    "plot_spatial",
    "plot_group_facets",
    "plot_som_grid",
]
