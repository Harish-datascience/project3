# Machine Learning Based Buyer Segmentation and Investment Profiling

Streamlit dashboard that segments real-estate buyers with K-Means (Hierarchical clustering as comparison)
using `clients.csv` and `properties.csv`.

## Folder structure
```
buyer_segmentation/
├── app.py              # Streamlit interface (6 pages + sidebar filters)
├── utils.py            # cleaning, encoding, scaling, clustering, naming, insights
├── requirements.txt
├── README.md
├── .streamlit/config.toml   # dark navy theme
└── data/
    ├── clients.csv
    └── properties.csv
```

## Run in VS Code (Windows / Mac / Linux)
1. Open the `buyer_segmentation` folder in VS Code (File > Open Folder), then open a terminal (Ctrl + `).
2. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py` (opens at http://localhost:8501)

The two maps on the Geographic page load map shapes from the internet; every other chart works offline.

## How the data is connected
`properties.client_ref` is matched to `clients.client_id`. Only **Sold** properties have a client reference
(7,305 records, all matched). **Available** properties (2,695) have no client, so they stay un-linked
and no client information is invented for them. The pipeline also counts sold records with an unknown
reference, so a different dataset would be handled safely.

## Methodology (what to explain in your review)
1. **Cleaning:** remove duplicates, normalise text labels, fix mixed date formats, `age` from `date_of_birth`,
   `sale_price` `"$300,385.62"` -> `300385.62`.
2. **Encoding:** binary fields (client type, purpose, loan) = label encoding (0/1); referral channel and country = one-hot.
3. **Scaling:** `StandardScaler` on age, satisfaction, units bought and average sale price.
4. **Purchase features** (from the real property data): units bought and average sale price per client.
5. **Choosing k:** Elbow Method (inertia) + Silhouette Score for k = 2 to 8. Default k = best silhouette between 3 and 6.
6. **Models:** K-Means (main) and Agglomerative/Hierarchical (Ward) for comparison, plus PCA only for the 2D plot.
7. **Naming:** rule-based names from each cluster's real averages (e.g. "Multi-Unit Buyers"). No income or "luxury" labels.

## Assumptions and limitations
- Slash dates (`MM/DD/YYYY`) are clearly month-first. Dash dates (`05-11-1968`) never exceed 12 in any position,
  so they are also treated as month-first. Worst case an age is off by under one year.
- Ages are calculated at a fixed date (31 Dec 2025) so results are reproducible.
- Region (57 values) and gender are not used to train the model; they are used to describe the clusters.
- Silhouette scores are low (about 0.14): the clusters overlap. Income, net worth and investment amount are not in the data,
  so PRD segments "Global Investors" and "Luxury Investors" cannot be identified.
- Filters change what is displayed. The model is trained once on all clients so segment names stay stable.
