"""utils.py - data cleaning, feature engineering, clustering and insight helpers.

All logic is kept here so app.py only deals with the user interface.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
REFERENCE_DATE = pd.Timestamp("2025-12-31")  # fixed date so ages (and clusters) never change
RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "satisfaction_score", "units_bought", "avg_sale_price"]
BINARY_FEATURES = {  # label encoding: column -> value that becomes 1
    "client_type": "Company",
    "acquisition_purpose": "Investment",
    "loan_applied": "Yes",
}
ONEHOT_FEATURES = ["referral_channel", "country"]  # one-hot encoding

COUNTRY_ISO = {
    "USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA",
    "Belgium": "BEL", "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK",
}
US_STATES = {
    "California": "CA", "Oregon": "OR", "Arizona": "AZ", "Nevada": "NV", "Colorado": "CO",
    "Utah": "UT", "Virginia": "VA", "Wyoming": "WY", "Kansas": "KS", "Ohio": "OH",
    "Washington": "WA", "Texas": "TX", "Florida": "FL", "New York": "NY", "Georgia": "GA",
}

# Columns used to describe / name clusters: flag columns (shown as %) and numeric columns
FLAG_LABELS = {
    "is_company": "Company clients", "is_investment": "Investment purchases",
    "loan_flag": "Loan applications", "is_usa": "USA-based clients",
    "via_website": "Website referrals", "via_agency": "Agency referrals",
    "via_client": "Client referrals",
}
NUM_LABELS = {
    "age": "Average age", "satisfaction_score": "Satisfaction score",
    "units_bought": "Units bought", "avg_sale_price": "Avg sale price ($)",
}


# ----------------------------------------------------------------------------
# 1. Cleaning
# ----------------------------------------------------------------------------
def parse_mixed_dates(series):
    """The files mix 'MM/DD/YYYY' and 'MM-DD-YYYY'. Unify the separator, then parse.

    Slash dates are clearly month-first (day goes above 12). Dash dates never exceed
    12 in either position, so we treat them as month-first too (consistent format).
    """
    cleaned = series.astype(str).str.strip().str.replace("-", "/", regex=False)
    return pd.to_datetime(cleaned, format="%m/%d/%Y", errors="coerce")


def clean_clients(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    report = {"clients_raw_rows": len(df), "clients_missing_before": int(df.isna().sum().sum())}

    # remove duplicate client entries
    df = df.drop_duplicates().drop_duplicates(subset="client_id", keep="first").copy()
    report["clients_duplicates_removed"] = report["clients_raw_rows"] - len(df)

    # normalise categorical labels
    cat_cols = ["client_type", "gender", "country", "region", "acquisition_purpose",
                "loan_applied", "referral_channel"]
    for col in cat_cols:
        df[col] = (df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
                   .fillna("Unknown").astype(object))
    for col in ["client_type", "acquisition_purpose", "loan_applied", "referral_channel"]:
        df[col] = df[col].str.title()
    df["gender"] = df["gender"].str.upper().str[0]
    df["country"] = df["country"].replace(
        {"Usa": "USA", "usa": "USA", "United States": "USA", "Uk": "UK", "uk": "UK",
         "United Kingdom": "UK"})

    # satisfaction score (1-5)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce").clip(1, 5)
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    # age from date_of_birth
    dob = parse_mixed_dates(df["date_of_birth"])
    age = np.floor((REFERENCE_DATE - dob).dt.days / 365.25)
    age = age.where(age.between(18, 100))
    report["clients_invalid_dob"] = int(age.isna().sum())
    df["age"] = age.fillna(age.median())
    df["age_group"] = pd.cut(df["age"], [0, 34, 49, 64, 200],
                             labels=["Under 35", "35-49", "50-64", "65+"]).astype(str)

    # simple 0/1 flags used for analysis and cluster naming
    df["is_company"] = (df["client_type"] == "Company").astype(int)
    df["is_investment"] = (df["acquisition_purpose"] == "Investment").astype(int)
    df["loan_flag"] = (df["loan_applied"] == "Yes").astype(int)
    df["is_usa"] = (df["country"] == "USA").astype(int)
    df["via_website"] = (df["referral_channel"] == "Website").astype(int)
    df["via_agency"] = (df["referral_channel"] == "Agency").astype(int)
    df["via_client"] = (df["referral_channel"] == "Client").astype(int)

    df = df.drop(columns=["first_name", "last_name", "date_of_birth"], errors="ignore")
    return df.reset_index(drop=True), report


def clean_properties(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    report = {"properties_raw_rows": len(df)}
    df = df.drop_duplicates(subset="listing_id").copy()
    report["properties_duplicates_removed"] = report["properties_raw_rows"] - len(df)

    # "$300,385.62" -> 300385.62
    df["sale_price"] = pd.to_numeric(
        df["sale_price"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"], errors="coerce")
    df["transaction_date"] = parse_mixed_dates(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    df["unit_category"] = df["unit_category"].astype(str).str.strip().str.title()
    df["listing_status"] = df["listing_status"].astype(str).str.strip().str.title()
    df["price_per_sqft"] = df["sale_price"] / df["floor_area_sqft"]

    # client_ref: blank / missing stays missing - we never invent a client
    ref = df["client_ref"].astype(str).str.strip().str.upper()
    df["client_ref"] = ref.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan}).astype(object)
    return df.reset_index(drop=True), report


def load_and_clean(data_dir=DATA_DIR):
    clients, r1 = clean_clients(pd.read_csv(Path(data_dir) / "clients.csv"))
    props, r2 = clean_properties(pd.read_csv(Path(data_dir) / "properties.csv"))

    has_ref = props["client_ref"].notna()
    matched = props["client_ref"].isin(clients["client_id"])
    report = {**r1, **r2,
              "records_matched": int(matched.sum()),
              "records_no_reference": int((~has_ref).sum()),
              "records_unmatched_reference": int((has_ref & ~matched).sum()),
              "clients_without_purchase": int((~clients["client_id"].isin(props["client_ref"])).sum())}
    return clients, props, report


def add_purchase_features(clients, props):
    """Add per-client purchase statistics from SOLD properties linked by client_ref."""
    sold = props[(props["listing_status"] == "Sold") & props["client_ref"].notna()]
    agg = sold.groupby("client_ref").agg(
        units_bought=("listing_id", "count"),
        total_spent=("sale_price", "sum"),
        avg_sale_price=("sale_price", "mean"),
        avg_floor_area=("floor_area_sqft", "mean"),
        office_share=("unit_category", lambda s: (s == "Office").mean()),
    )
    out = clients.merge(agg, left_on="client_id", right_index=True, how="left")
    out["units_bought"] = out["units_bought"].fillna(0)  # no purchase = 0 units (a fact, not a guess)
    return out


# ----------------------------------------------------------------------------
# 2. Encoding, scaling and clustering
# ----------------------------------------------------------------------------
def build_feature_matrix(clients):
    """Scale numeric features, label-encode binary ones, one-hot encode the rest."""
    num = clients[NUMERIC_FEATURES].copy()
    # clients with no purchase have no average price: median is used ONLY inside the model matrix
    num = num.fillna(num.median())
    scaled = pd.DataFrame(StandardScaler().fit_transform(num), columns=NUMERIC_FEATURES,
                          index=clients.index)
    binary = pd.DataFrame({f"{col}_is_{val.lower()}": (clients[col] == val).astype(float)
                           for col, val in BINARY_FEATURES.items()}, index=clients.index)
    onehot = pd.get_dummies(clients[ONEHOT_FEATURES], dtype=float)
    return pd.concat([scaled, binary, onehot], axis=1)


def evaluate_k(X, k_min=2, k_max=8):
    """Inertia (Elbow) and Silhouette score for K-Means and Hierarchical clustering."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette_kmeans": silhouette_score(X, km.labels_),
                     "silhouette_hierarchical": silhouette_score(X, hc)})
    return pd.DataFrame(rows)


def recommend_k(scores, lo=3, hi=6):
    """Best silhouette K-Means k between 3 and 6 (2 segments are too few to be useful)."""
    sub = scores[(scores["k"] >= lo) & (scores["k"] <= hi)]
    return int(sub.loc[sub["silhouette_kmeans"].idxmax(), "k"])


def elbow_k(scores):
    """Point furthest from the straight line joining first and last inertia values."""
    x = (scores["k"] - scores["k"].min()) / (scores["k"].max() - scores["k"].min())
    y = (scores["inertia"] - scores["inertia"].min()) / (scores["inertia"].max() - scores["inertia"].min())
    dist = np.abs(y - (1 - x)) / np.sqrt(2)
    return int(scores.loc[dist.idxmax(), "k"])


def fit_segments(X, k):
    """Final K-Means (labels sorted so cluster 1 is the largest) + Hierarchical + PCA."""
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    order = pd.Series(km.labels_).value_counts().index.tolist()
    labels = pd.Series(km.labels_).map({old: new for new, old in enumerate(order, 1)}).to_numpy()
    hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X) + 1
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return {"labels": labels, "hier_labels": hier,
            "sil_km": silhouette_score(X, labels), "sil_hc": silhouette_score(X, hier),
            "ari": adjusted_rand_score(labels, hier),
            "coords": pca.fit_transform(X), "explained": pca.explained_variance_ratio_}


# ----------------------------------------------------------------------------
# 3. Cluster profiling and naming (based only on actual data)
# ----------------------------------------------------------------------------
def profile_clusters(df, group="cluster"):
    g = df.groupby(group)
    prof = g.agg(clients=("client_id", "count"), avg_age=("age", "mean"),
                 avg_satisfaction=("satisfaction_score", "mean"),
                 pct_investment=("is_investment", "mean"), pct_company=("is_company", "mean"),
                 pct_loan=("loan_flag", "mean"), pct_usa=("is_usa", "mean"),
                 avg_units=("units_bought", "mean"), avg_sale_price=("avg_sale_price", "mean"))
    for c in ["pct_investment", "pct_company", "pct_loan", "pct_usa"]:
        prof[c] = prof[c] * 100
    prof["top_referral"] = g["referral_channel"].agg(lambda s: s.mode().iat[0])
    prof["top_country"] = g["country"].agg(lambda s: s.mode().iat[0])
    return prof.round(2).reset_index()


def name_clusters(df):
    """Rule-based names built from each cluster's real averages (no invented labels)."""
    cols = ["is_company", "is_investment", "loan_flag", "is_usa", "age",
            "satisfaction_score", "units_bought", "avg_sale_price"]
    overall = df[cols].mean()
    names = {}
    for cid, grp in df.groupby("cluster"):
        m = grp[cols].mean()
        purpose = "Investment Buyers" if m["is_investment"] >= 0.5 else "Home Buyers"
        if m["is_company"] >= 0.5:
            name = "Corporate Buyers"
        elif m["units_bought"] >= 1.5 * overall["units_bought"]:
            name = "Multi-Unit Buyers"
        elif m["is_investment"] >= overall["is_investment"] + 0.15:
            name = "Investment-Oriented Buyers"
        elif m["loan_flag"] >= overall["loan_flag"] + 0.15:
            name = "Loan-Financed " + purpose
        elif m["is_usa"] <= overall["is_usa"] - 0.25:
            name = "International " + purpose
        elif m["avg_sale_price"] >= 1.15 * overall["avg_sale_price"]:
            name = "Higher-Price " + purpose
        else:
            parts = []
            if m["age"] <= overall["age"] - 5:
                parts.append("Younger")
            elif m["age"] >= overall["age"] + 5:
                parts.append("Senior")
            if m["satisfaction_score"] >= overall["satisfaction_score"] + 0.8:
                parts.append("High-Satisfaction")
            elif m["satisfaction_score"] <= overall["satisfaction_score"] - 0.8:
                parts.append("Low-Satisfaction")
            name = " ".join(parts + [purpose]) if parts else "Mainstream " + purpose
        names[cid] = name

    counts = pd.Series(names).value_counts()  # keep names unique
    for cid, name in names.items():
        if counts[name] > 1:
            names[cid] = f"{name} (Cluster {cid})"
    return names


def cluster_traits(df, cid, top_n=3):
    """The traits that make a cluster most different from the overall client base."""
    cols = list(FLAG_LABELS) + list(NUM_LABELS)
    mu, sd = df[cols].mean(), df[cols].std().replace(0, np.nan)
    grp = df[df["cluster"] == cid][cols].mean()
    z = ((grp - mu) / sd).dropna()
    lines = []
    for col in z.abs().sort_values(ascending=False).head(top_n).index:
        arrow = "higher" if z[col] > 0 else "lower"
        if col in FLAG_LABELS:
            lines.append(f"{FLAG_LABELS[col]}: {grp[col]*100:.0f}% ({arrow}; overall {mu[col]*100:.0f}%)")
        else:
            fmt = ",.0f" if col == "avg_sale_price" else ",.1f"
            lines.append(f"{NUM_LABELS[col]}: {grp[col]:{fmt}} ({arrow}; overall {mu[col]:{fmt}})")
    return lines


# ----------------------------------------------------------------------------
# 4. Business insights (all numbers come from the currently filtered data)
# ----------------------------------------------------------------------------
def generate_insights(cl, sold, sil):
    """Plain-language insights. Wording adapts: small differences are reported as 'similar'."""
    out = []
    if cl.empty:
        return out
    inv = cl["is_investment"].mean() * 100

    seg = cl.groupby("segment").agg(n=("client_id", "count"), inv=("is_investment", "mean"),
                                    loan=("loan_flag", "mean"), sat=("satisfaction_score", "mean"))
    seg = seg[seg["n"] >= 10]
    if len(seg) > 1:
        s = seg["inv"].idxmax()
        spread = (seg["inv"].max() - seg["inv"].min()) * 100
        if spread < 8:
            out.append(("Investment intent is spread evenly across segments",
                        f"Investment share only ranges from {seg['inv'].min()*100:.0f}% to {seg['inv'].max()*100:.0f}% "
                        f"across segments ({inv:.0f}% overall). Segment alone is a weak predictor of investors, so "
                        "target investors using region and referral channel instead."))
        else:
            out.append(("Target the most investment-oriented segment",
                        f"'{s}' has the highest investment share ({seg.loc[s, 'inv']*100:.0f}% vs {inv:.0f}% overall). "
                        "Use investor-focused messaging (rental yield, resale value) for this group."))
        s = seg["loan"].idxmax()
        if (seg["loan"].max() - seg["loan"].min()) * 100 < 8:
            out.append(("Loan demand is similar in every segment",
                        f"Loan applications range from {seg['loan'].min()*100:.0f}% to {seg['loan'].max()*100:.0f}%. "
                        "Offer mortgage pre-approval and instalment plans to all buyers rather than one segment."))
        else:
            out.append(("Offer financing where loan demand is highest",
                        f"'{s}' applies for loans most often ({seg.loc[s, 'loan']*100:.0f}% vs "
                        f"{cl['loan_flag'].mean()*100:.0f}% overall). Partner with banks and promote financing here."))
        lo, hi = seg["sat"].idxmin(), seg["sat"].idxmax()
        if seg["sat"].max() - seg["sat"].min() < 0.3:
            out.append(("Satisfaction is similar across segments",
                        f"Average satisfaction ranges from only {seg['sat'].min():.2f} to {seg['sat'].max():.2f}. "
                        "Improve service quality for all clients."))
        else:
            out.append(("Fix the weakest customer experience",
                        f"'{lo}' has the lowest satisfaction ({seg.loc[lo, 'sat']:.2f}) and '{hi}' the highest "
                        f"({seg.loc[hi, 'sat']:.2f}). Review the after-sales process for '{lo}' and copy what works for '{hi}'."))

    reg = cl.groupby(["country", "region"]).agg(n=("client_id", "count"), inv=("is_investment", "mean")).reset_index()
    reg = reg[reg["n"] >= 15]
    if not reg.empty:
        r = reg.loc[reg["inv"].idxmax()]
        note = " This is a small sample, so validate before spending heavily." if r["n"] < 40 else ""
        out.append(("Focus regional campaigns",
                    f"{r['region']} ({r['country']}) has the highest investment share among regions with at least "
                    f"15 clients: {r['inv']*100:.0f}% ({int(r['n'])} clients).{note}"))

    ch = cl.groupby("referral_channel").agg(n=("client_id", "count"), inv=("is_investment", "mean"),
                                            sat=("satisfaction_score", "mean"))
    ch = ch[ch["n"] >= 10]
    if len(ch) > 1:
        if (ch["inv"].max() - ch["inv"].min()) * 100 < 5 and ch["sat"].max() - ch["sat"].min() < 0.3:
            out.append(("Referral channels perform similarly",
                        "Investor share and satisfaction are almost the same across channels, so choose channels "
                        "mainly by cost per client rather than by buyer quality."))
        else:
            bi, bs = ch["inv"].idxmax(), ch["sat"].idxmax()
            out.append(("Spend marketing budget on the right channel",
                        f"'{bi}' referrals bring the highest investor share ({ch.loc[bi, 'inv']*100:.0f}%) and "
                        f"'{bs}' referrals the best satisfaction ({ch.loc[bs, 'sat']:.2f}). Compare with each "
                        "channel's cost before shifting budget."))

    if not sold.empty and sold["acquisition_purpose"].nunique() > 1:
        pp = sold.groupby("acquisition_purpose")["price_per_sqft"].mean()
        off = sold.groupby("acquisition_purpose")["unit_category"].apply(lambda s: (s == "Office").mean() * 100)
        gap = abs(pp["Investment"] - pp["Home"]) / pp["Home"] * 100
        txt = (f"Investors pay ${pp['Investment']:,.0f}/sqft vs ${pp['Home']:,.0f}/sqft for home buyers; "
               f"{off['Investment']:.0f}% of investor purchases are offices (home buyers: {off['Home']:.0f}%). ")
        txt += ("Prices are almost identical, so investors are not buying pricier units - differentiate them by "
                "intent, not by price." if gap < 3 else "Price positioning should differ for the two groups.")
        out.append(("Investment vs personal-use purchases", txt))
    if not sold.empty and sold["month"].notna().any():
        peak = sold.groupby("month").size().idxmax()
        out.append(("Time campaigns before peak sales months",
                    f"The busiest sales month in the selected data is {peak:%B %Y}. Launch campaigns 1-2 months "
                    "earlier to capture buyers before the peak."))

    out.append(("Limitation - use segments as a guide, not strict rules",
                f"The K-Means silhouette score is {sil:.2f}, which is low: segments overlap and the dataset has no "
                "income, net-worth or investment-amount fields, so 'high-income' or 'luxury' segments cannot be "
                "claimed. Collecting budget, income and enquiry history would make future segments sharper."))
    return out
