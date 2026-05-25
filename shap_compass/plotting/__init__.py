"""SHAP-Compass visualization helpers.

The headline figures of the paper are:

    plot_som_grid               -- Fig.6 / Fig.10 (SOM neuron map + hit map
                                  + per-neuron target mean)
    plot_ward_dendrogram        -- companion dendrogram for the Ward cut.
    plot_bilayer_heatmap        -- Fig.7 / Fig.11 (regimes x features,
                                  each cell split into upper Z^F and
                                  lower Z^S half-cells).
    plot_per_feature_unit_circle -- Fig.9 / Fig.13 (one unit-circle subplot
                                  per feature, sorted by descending DCI,
                                  border colour encodes the DCI band).
    plot_dci_ranking            -- bar chart of DCI by feature.
    plot_theta_heatmap          -- regime x feature mean direction angle.
    plot_group_overview         -- regime-size bar chart + mean Z^S heatmap.
    plot_spatial / _facets      -- Fig.8 / Fig.12 spatial maps.
"""

from .dci_chart import plot_dci_ranking
from .bilayer import plot_bilayer_heatmap, plot_per_feature_unit_circle
from .heatmap import plot_group_overview, plot_theta_heatmap
from .spatial import plot_spatial, plot_group_facets
from .som_grid import plot_som_grid, plot_ward_dendrogram

__all__ = [
    "plot_dci_ranking",
    "plot_bilayer_heatmap",
    "plot_per_feature_unit_circle",
    "plot_group_overview",
    "plot_theta_heatmap",
    "plot_spatial",
    "plot_group_facets",
    "plot_som_grid",
    "plot_ward_dendrogram",
]
