# SHAP-Compass examples

Three self-contained scripts. All three use **synthetic data only** —
no real groundwater monitoring data is bundled with the package.

| Script | Samples | Features | SOM | k | What it shows |
|---|---|---|---|---|---|
| `01_quickstart.py` | 500 | 8 | 7 x 7 | 3 | Smallest end-to-end demo. Runs in seconds. |
| `02_taiwan_synthetic.py` | 2,375 | 17 (in 7 functional dimensions) | 9 x 9 | 6 | Mirrors the Taiwan case study (Section 3.1) of the paper. |
| `03_conus_synthetic.py` | 6,000 | 35 (in 9 functional dimensions) | 20 x 20 | 7 | Mirrors the CONUS case study (Section 3.2), at a smaller feature count for runtime. |

## Why no real data?

The paper's Taiwan case relies on Taiwan EPA monitoring data whose terms
of use restrict redistribution; the CONUS case extends the dataset
curated by Ransom et al. (2022, *STOTEN*). Both are publicly retrievable
by their respective owners, but the SHAP-Compass package focuses on the
*method* and does not bundle source data. The synthetic generators in
these examples reproduce the **directional structure** that the paper
describes (multiple attribution regimes with distinct Z^F / Z^S
signatures, functional feature groupings, an ordered target range), so
that the full pipeline (transform → SOM → Ward → DCI → bilayer heatmap)
can be exercised without any external downloads.

## Running

```bash
pip install -e .
python examples/01_quickstart.py
python examples/02_taiwan_synthetic.py
python examples/03_conus_synthetic.py
```

Output figures and CSVs land in `examples/<name>/output/`.
