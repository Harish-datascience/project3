"""Machine Learning Based Buyer Segmentation and Investment Profiling - Streamlit app."""
import pandas as pd
import plotly.express as px
import streamlit as st

import utils as U

st.set_page_config(page_title="Buyer Segmentation Dashboard", page_icon="🏢", layout="wide")

# ------------------------------------------------------------------ theme
COLORS = ["#38bdf8", "#6366f1", "#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#94a3b8"]
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = COLORS
px.defaults.color_continuous_scale = px.colors.sequential.Blues

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#0b1426 0%,#091020 100%);}
[data-testid="stSidebar"] {background:#0d1b33; border-right:1px solid #1e3358;}
.block-container {padding-top: 2rem;}
.kpi {background:linear-gradient(135deg,#12264a,#0f3a6b); border:1px solid #23518f; border-radius:14px;
      padding:16px 18px; box-shadow:0 4px 14px rgba(0,0,0,.35); margin-bottom:10px;}
.kpi .label {font-size:.75rem; color:#9db8e0; text-transform:uppercase; letter-spacing:.07em;}
.kpi .value {font-size:2rem; font-weight:700; color:#ffffff; line-height:1.2;}
.kpi .sub {font-size:.72rem; color:#7fa6d8;}
.card {background:#10213f; border-left:4px solid #3b82f6; padding:12px 16px; border-radius:8px; margin-bottom:10px;}
.card b {color:#7dd3fc;}
.warn {background:#2a2110; border-left:4px solid #fbbf24; padding:12px 16px; border-radius:8px; margin:8px 0;}
</style>
""", unsafe_allow_html=True)

PAGES = ["1. Buyer Segmentation Overview", "2. Buyer Behaviour Analysis", "3. Geographic Buyer Analysis",
         "4. Property & Investment Analysis", "5. ML & Segment Insights", "6. Business Insights"]


def kpi_row(items):
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def show(fig, height=380):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(gridcolor="#1e3358")
    fig.update_yaxes(gridcolor="#1e3358")
    st.plotly_chart(fig)


def stacked_pct(df, x, color, title, cmap=None):
    """100% stacked bar: share of each `color` value inside every `x` group."""
    ct = pd.crosstab(df[x], df[color], normalize="index").mul(100).round(1)
    long = ct.reset_index().melt(id_vars=x, var_name=color, value_name="percent")
    fig = px.bar(long, x=x, y="percent", color=color, title=title, text_auto=".0f",
                 color_discrete_map=cmap, labels={"percent": "% of clients"})
    return fig


# ------------------------------------------------------------------ data + models (cached)
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data():
    clients, props, report = U.load_and_clean()
    return U.add_purchase_features(clients, props), props, report


@st.cache_data(show_spinner="Evaluating cluster counts (Elbow + Silhouette)...")
def get_scores(clients):
    return U.evaluate_k(U.build_feature_matrix(clients))


@st.cache_data(show_spinner="Training clustering models...")
def get_segments(clients, k):
    return U.fit_segments(U.build_feature_matrix(clients), k)


clients, props, report = load_data()
scores = get_scores(clients)
rec_k = U.recommend_k(scores)

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 🏢 Buyer Intelligence")
st.sidebar.caption("ML-based buyer segmentation & investment profiling")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
st.sidebar.caption("Leave a filter empty to include everything.")
country_sel = st.sidebar.multiselect("Country", sorted(clients["country"].unique()), placeholder="All countries")
region_opts = sorted(clients[clients["country"].isin(country_sel)]["region"].unique() if country_sel
                     else clients["region"].unique())
region_sel = st.sidebar.multiselect("Region", region_opts, placeholder="All regions")
purpose_sel = st.sidebar.multiselect("Acquisition purpose", sorted(clients["acquisition_purpose"].unique()),
                                     placeholder="All purposes")
type_sel = st.sidebar.multiselect("Client type", sorted(clients["client_type"].unique()),
                                  placeholder="All client types")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model setting")
k = st.sidebar.slider("Number of clusters (k)", 2, 8, rec_k,
                      help=f"Default {rec_k} = best silhouette score between k=3 and k=6.")
st.sidebar.caption("Clusters are trained on all clients; filters only change what is displayed.")

# ------------------------------------------------------------------ segments + filtering
seg_result = get_segments(clients, k)
clients = clients.copy()
clients["cluster"] = seg_result["labels"]
names = U.name_clusters(clients)
clients["segment"] = clients["cluster"].map(names)
cmap = {n: COLORS[i % len(COLORS)] for i, n in enumerate(sorted(names.values()))}

mask = pd.Series(True, index=clients.index)
if country_sel:
    mask &= clients["country"].isin(country_sel)
if region_sel:
    mask &= clients["region"].isin([r for r in region_sel if r in region_opts])
if purpose_sel:
    mask &= clients["acquisition_purpose"].isin(purpose_sel)
if type_sel:
    mask &= clients["client_type"].isin(type_sel)
cl = clients[mask]
is_filtered = bool(country_sel or region_sel or purpose_sel or type_sel)

# SOLD properties joined to client info through client_ref (left join: unmatched rows keep NaN, nothing invented)
client_cols = ["client_id", "segment", "client_type", "acquisition_purpose", "country", "region", "loan_applied"]
sold_all = props[props["listing_status"] == "Sold"].merge(clients[client_cols], left_on="client_ref",
                                                          right_on="client_id", how="left")
sold = sold_all[sold_all["client_ref"].isin(cl["client_id"])] if is_filtered else sold_all
available = props[props["listing_status"] == "Available"]

if cl.empty:
    st.title(page[3:])
    st.warning("No clients match the selected filters. Please widen or clear some filters in the sidebar.")
    st.stop()

seg_sold = sold.dropna(subset=["segment"])
st.title(page[3:])
if is_filtered:
    st.caption(f"Filters active - showing {len(cl):,} of {len(clients):,} clients and {len(sold):,} linked sales.")


# ================================================================== PAGE 1
def page_overview():
    kpi_row([("Total Clients", f"{len(cl):,}", "after cleaning & de-duplication"),
             ("Property Records", f"{len(props):,}", "all records (listings)"),
             ("Sold Properties", f"{len(sold):,}", "linked to the selected clients"),
             ("Available Properties", f"{len(available):,}", "no client yet - not affected by filters"),
             ("Buyer Segments", f"{cl['segment'].nunique()}", f"K-Means, k = {k}")])
    c1, c2 = st.columns(2)
    dist = cl["segment"].value_counts().rename_axis("segment").reset_index(name="clients")
    with c1:
        show(px.pie(dist, names="segment", values="clients", hole=0.55, title="Buyer segment distribution",
                    color="segment", color_discrete_map=cmap))
    with c2:
        show(px.bar(dist.sort_values("clients"), x="clients", y="segment", orientation="h", text="clients",
                    title="Clients per segment", color="segment", color_discrete_map=cmap)
             .update_layout(showlegend=False))
    c1, c2, c3 = st.columns(3)
    with c1:
        show(px.histogram(cl, x="age", color="gender", nbins=20, barmode="overlay", opacity=0.75,
                          title="Age distribution by gender"), 320)
    with c2:
        show(px.pie(cl, names="acquisition_purpose", hole=0.5, title="Acquisition purpose"), 320)
    with c3:
        top = cl["country"].value_counts().head(8).rename_axis("country").reset_index(name="clients")
        show(px.bar(top, x="country", y="clients", title="Top countries", text="clients"), 320)

    with st.expander("Dataset overview and data-cleaning report"):
        a, b = st.columns(2)
        a.markdown(f"""**Clients file:** {report['clients_raw_rows']:,} rows  
- Duplicates removed: {report['clients_duplicates_removed']}  
- Missing values before cleaning: {report['clients_missing_before']}  
- Invalid birth dates: {report['clients_invalid_dob']}  
- Age calculated from `date_of_birth` (mixed `MM/DD/YYYY` and `MM-DD-YYYY` formats unified)""")
        b.markdown(f"""**Properties file:** {report['properties_raw_rows']:,} rows  
- `sale_price` converted from text like `$300,385.62` to numbers  
- Sold records matched to a client: **{report['records_matched']:,}**  
- Available records (no `client_ref`, left empty): **{report['records_no_reference']:,}**  
- Sold records with unknown `client_ref`: **{report['records_unmatched_reference']}**  
- Clients with no purchase: {report['clients_without_purchase']}""")
        t1, t2 = st.tabs(["Clients (cleaned)", "Properties (cleaned)"])
        t1.dataframe(clients.drop(columns=["is_company", "is_investment", "loan_flag", "is_usa", "via_website",
                                           "via_agency", "via_client"]).head(100), hide_index=True)
        t2.dataframe(props.head(100), hide_index=True)


# ================================================================== PAGE 2
def page_behaviour():
    kpi_row([("Investment Buyers", f"{cl['is_investment'].mean()*100:.1f}%", "of selected clients"),
             ("Company Clients", f"{cl['is_company'].mean()*100:.1f}%", "rest are individuals"),
             ("Applied for Loan", f"{cl['loan_flag'].mean()*100:.1f}%", "financing indicator"),
             ("Avg Satisfaction", f"{cl['satisfaction_score'].mean():.2f} / 5", "customer rating")])
    t1, t2, t3, t4, t5 = st.tabs(["Purpose", "Client type", "Loans", "Satisfaction", "Referral channel"])
    with t1:
        a, b = st.columns(2)
        with a:
            show(px.pie(cl, names="acquisition_purpose", hole=0.5, title="Investment vs personal use (Home)"))
        with b:
            show(stacked_pct(cl, "segment", "acquisition_purpose", "Purpose by segment"))
    with t2:
        a, b = st.columns(2)
        with a:
            show(stacked_pct(cl, "client_type", "acquisition_purpose", "Individual vs Company: purpose"))
        with b:
            show(stacked_pct(cl, "segment", "client_type", "Client type by segment"))
    with t3:
        a, b = st.columns(2)
        with a:
            show(stacked_pct(cl, "segment", "loan_applied", "Loan application by segment"))
        with b:
            lp = (cl.groupby(["acquisition_purpose", "client_type"])["loan_flag"].mean().mul(100).round(1)
                  .reset_index())
            show(px.bar(lp, x="acquisition_purpose", y="loan_flag", color="client_type", barmode="group",
                        text="loan_flag", title="Loan rate (%) by purpose and client type"))
    with t4:
        a, b = st.columns(2)
        sat = cl.groupby("segment")["satisfaction_score"].mean().round(2).reset_index()
        with a:
            show(px.bar(sat, x="segment", y="satisfaction_score", text="satisfaction_score", color="segment",
                        color_discrete_map=cmap, title="Average satisfaction by segment")
                 .update_layout(showlegend=False, yaxis_range=[0, 5]))
        with b:
            d = cl.assign(score=cl["satisfaction_score"].astype(int).astype(str))
            show(stacked_pct(d, "segment", "score", "Satisfaction score distribution by segment"))
    with t5:
        a, b = st.columns(2)
        with a:
            show(stacked_pct(cl, "segment", "referral_channel", "Referral channel by segment"))
        with b:
            ch = cl.groupby("referral_channel").agg(investment=("is_investment", "mean"),
                                                    satisfaction=("satisfaction_score", "mean")).reset_index()
            ch["investment"] = (ch["investment"] * 100).round(1)
            show(px.bar(ch, x="referral_channel", y="investment", text="investment", color="referral_channel",
                        title="Investment share (%) by referral channel").update_layout(showlegend=False))


# ================================================================== PAGE 3
def page_geography():
    geo = cl.groupby("country").agg(clients=("client_id", "count"), investment_share=("is_investment", "mean"),
                                    avg_satisfaction=("satisfaction_score", "mean")).reset_index()
    geo["investment_share"] = (geo["investment_share"] * 100).round(1)
    geo["avg_satisfaction"] = geo["avg_satisfaction"].round(2)
    geo["iso"] = geo["country"].map(U.COUNTRY_ISO)

    metric = st.radio("Map metric", ["clients", "investment_share"], horizontal=True,
                      format_func=lambda m: "Number of clients" if m == "clients" else "Investment share (%)")
    fig = px.choropleth(geo.dropna(subset=["iso"]), locations="iso", color=metric, hover_name="country",
                        hover_data={"iso": False, "clients": True, "investment_share": True},
                        title="Buyers by country")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, landcolor="#13233f", showcountries=True,
                    countrycolor="#2a4470", projection_type="natural earth")
    show(fig, 430)
    st.caption("The map needs an internet connection to load country shapes. Small countries have few clients, "
               "so their investment share can be unstable - check the client count in the hover box.")

    a, b = st.columns(2)
    with a:
        cc = cl.groupby(["country", "acquisition_purpose"]).size().reset_index(name="clients")
        order = geo.sort_values("clients", ascending=False)["country"].tolist()
        show(px.bar(cc, x="country", y="clients", color="acquisition_purpose", title="Clients by country and purpose",
                    category_orders={"country": order}))
    with b:
        ct = pd.crosstab(cl["country"], cl["segment"], normalize="index").mul(100).round(0)
        show(px.imshow(ct, text_auto=True, aspect="auto", title="Segment mix by country (% of country clients)"))

    st.subheader("Region drill-down")
    countries = sorted(cl["country"].unique())
    country = st.selectbox("Choose a country", countries, index=countries.index("USA") if "USA" in countries else 0)
    sub = cl[cl["country"] == country]
    reg = sub.groupby("region").agg(clients=("client_id", "count"), investment_share=("is_investment", "mean")).reset_index()
    reg["investment_share"] = (reg["investment_share"] * 100).round(1)
    a, b = st.columns(2)
    with a:
        show(px.bar(reg.sort_values("clients"), x="clients", y="region", orientation="h", text="clients",
                    title=f"Clients by region - {country}"), max(320, 28 * len(reg)))
    with b:
        show(px.bar(reg.sort_values("investment_share"), x="investment_share", y="region", orientation="h",
                    text="investment_share", title=f"Investment share (%) by region - {country}"),
             max(320, 28 * len(reg)))
    if country == "USA":
        usa = reg.assign(state=reg["region"].map(U.US_STATES)).dropna(subset=["state"])
        fig = px.choropleth(usa, locations="state", locationmode="USA-states", scope="usa", color="investment_share",
                            hover_name="region", hover_data=["clients"], title="USA: investment share (%) by state")
        fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#13233f")
        show(fig, 420)
    ct = pd.crosstab(sub["region"], sub["segment"], normalize="index").mul(100).round(0)
    show(px.imshow(ct, text_auto=True, aspect="auto", title=f"Segment mix by region - {country} (% of region clients)"),
         max(320, 26 * len(ct)))


# ================================================================== PAGE 4
def page_property():
    if sold.empty:
        st.info("No sold properties are linked to the selected clients.")
        return
    kpi_row([("Total Sales Value", f"${sold['sale_price'].sum()/1e6:,.1f}M", "sold properties"),
             ("Avg Sale Price", f"${sold['sale_price'].mean():,.0f}", "per unit"),
             ("Avg Floor Area", f"{sold['floor_area_sqft'].mean():,.0f} sqft", "per unit"),
             ("Avg Price / sqft", f"${sold['price_per_sqft'].mean():,.0f}", "sold units")])
    t1, t2, t3, t4 = st.tabs(["Prices & floor areas", "Sold vs available", "Transaction trends", "By buyer segment"])
    with t1:
        a, b = st.columns(2)
        with a:
            show(px.histogram(sold, x="sale_price", color="unit_category", nbins=40, barmode="overlay", opacity=0.75,
                              title="Sale price distribution", labels={"sale_price": "Sale price ($)"}))
        with b:
            show(px.scatter(sold.sample(min(2000, len(sold)), random_state=1), x="floor_area_sqft", y="sale_price",
                            color="unit_category", opacity=0.6, title="Floor area vs sale price (2,000 sample)"))
    with t2:
        a, b = st.columns(2)
        status = pd.concat([sold[["unit_category", "listing_status"]], available[["unit_category", "listing_status"]]])
        st_ct = status.groupby(["unit_category", "listing_status"]).size().reset_index(name="units")
        with a:
            show(px.bar(st_ct, x="unit_category", y="units", color="listing_status", barmode="group", text="units",
                        title="Apartment vs Office: sold and available"))
        with b:
            av = pd.concat([sold.assign(status="Sold"), available.assign(status="Available")])
            show(px.box(av, x="unit_category", y="sale_price", color="status",
                        title="Price: sold vs available (asking) units"))
        st.caption("Available units have no client, so client filters do not apply to them.")
    with t3:
        m = sold.groupby(["month", "unit_category"]).agg(units=("listing_id", "count"),
                                                         value=("sale_price", "sum")).reset_index()
        a, b = st.columns(2)
        with a:
            show(px.line(m, x="month", y="units", color="unit_category", markers=True, title="Units sold per month"))
        with b:
            pm = sold.groupby("month")["price_per_sqft"].mean().reset_index()
            show(px.line(pm, x="month", y="price_per_sqft", markers=True, title="Average price per sqft per month"))
        show(px.area(m, x="month", y="value", color="unit_category", title="Monthly sales value ($)"), 320)
    with t4:
        if seg_sold.empty:
            st.info("No sold records are linked to a client segment.")
            return
        a, b = st.columns(2)
        sp = seg_sold.groupby("segment").agg(avg_price=("sale_price", "mean"), units=("listing_id", "count"),
                                             clients=("client_ref", "nunique")).reset_index()
        sp["units_per_client"] = (sp["units"] / sp["clients"]).round(2)
        with a:
            show(px.bar(sp, x="segment", y="avg_price", color="segment", color_discrete_map=cmap,
                        title="Average sale price by segment ($)").update_layout(showlegend=False))
        with b:
            show(px.bar(sp, x="segment", y="units_per_client", color="segment", color_discrete_map=cmap,
                        text="units_per_client", title="Units bought per client").update_layout(showlegend=False))
        a, b = st.columns(2)
        with a:
            show(stacked_pct(seg_sold, "segment", "unit_category", "Apartment vs Office by segment"))
        with b:
            show(px.box(seg_sold, x="acquisition_purpose", y="price_per_sqft", color="client_type",
                        title="Price per sqft: investment vs home, individual vs company"))


# ================================================================== PAGE 5
def page_ml():
    kpi_row([("Selected k", f"{k}", f"recommended k = {rec_k}"),
             ("K-Means Silhouette", f"{seg_result['sil_km']:.3f}", "higher is better (max 1)"),
             ("Hierarchical Silhouette", f"{seg_result['sil_hc']:.3f}", "Ward linkage, same k"),
             ("K-Means vs Hierarchical", f"{seg_result['ari']:.2f}", "agreement (ARI, 1 = identical)")])
    st.markdown(f"""<div class="card"><b>Method.</b> Numeric features (age, satisfaction, units bought, average sale price)
are scaled with <b>StandardScaler</b>. Binary fields (client type, purpose, loan) are <b>label-encoded</b> (0/1); referral
channel and country are <b>one-hot encoded</b>. Region has 57 values, so it is used to <i>describe</i> clusters rather than
to train them. Gender is also not used for training. Purchase features come from joining sold properties to clients with
<code>client_ref</code>.</div>""", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        fig = px.line(scores, x="k", y="inertia", markers=True, title=f"Elbow Method (visual hint: k = {U.elbow_k(scores)})",
                      labels={"inertia": "Inertia (within-cluster distance)"})
        fig.add_vline(x=k, line_dash="dash", line_color="#fbbf24")
        show(fig)
    with b:
        long = scores.melt(id_vars="k", value_vars=["silhouette_kmeans", "silhouette_hierarchical"],
                           var_name="algorithm", value_name="silhouette")
        long["algorithm"] = long["algorithm"].map({"silhouette_kmeans": "K-Means", "silhouette_hierarchical": "Hierarchical"})
        fig = px.line(long, x="k", y="silhouette", color="algorithm", markers=True, title="Silhouette score by k")
        fig.add_vline(x=k, line_dash="dash", line_color="#fbbf24")
        show(fig)

    st.markdown(f"""<div class="warn"><b>Honest reading of the results:</b> the silhouette score is only about
{seg_result['sil_km']:.2f} (values near 0.1-0.2 mean weak structure) and the Elbow curve bends gently rather than sharply.
The clusters are therefore <b>overlapping groups, not sharply separated types</b>. K-Means scores better than Hierarchical
clustering here. The dataset has no income, net-worth or investment-amount columns, so the PRD segments
<i>Global Investors</i> (C1) and <i>Luxury Investors</i> (C4) cannot be identified; only segments supported by the real data are named.</div>""",
                unsafe_allow_html=True)

    plot = pd.DataFrame(seg_result["coords"], columns=["PC1", "PC2"])
    plot["segment"] = clients["segment"].to_numpy()
    plot = plot[mask.to_numpy()]
    ev = seg_result["explained"]
    a, b = st.columns(2)
    with a:
        show(px.scatter(plot, x="PC1", y="PC2", color="segment", color_discrete_map=cmap, opacity=0.6,
                        title=f"Clusters in 2D (PCA, {ev.sum()*100:.0f}% of variance)"))
    with b:
        cross = pd.crosstab(pd.Series(seg_result["labels"], name="K-Means cluster"),
                            pd.Series(seg_result["hier_labels"], name="Hierarchical cluster"))
        show(px.imshow(cross, text_auto=True, aspect="auto", title="K-Means vs Hierarchical (client counts)"))

    st.subheader("Cluster profiles (selected clients)")
    prof = U.profile_clusters(cl)
    prof.insert(1, "segment", prof["cluster"].map(names))
    st.dataframe(prof.rename(columns={
        "clients": "Clients", "avg_age": "Avg age", "avg_satisfaction": "Avg satisfaction",
        "pct_investment": "% Investment", "pct_company": "% Company", "pct_loan": "% Loan",
        "pct_usa": "% USA", "avg_units": "Avg units", "avg_sale_price": "Avg sale price ($)",
        "top_referral": "Top referral", "top_country": "Top country"}), hide_index=True)

    st.subheader("What makes each cluster different")
    cols = st.columns(2)
    for i, cid in enumerate(sorted(names)):
        traits = "".join(f"<li>{t}</li>" for t in U.cluster_traits(clients, cid))
        size = int((clients["cluster"] == cid).sum())
        cols[i % 2].markdown(f'<div class="card"><b>Cluster {cid}: {names[cid]}</b> ({size:,} clients)<ul>{traits}</ul></div>',
                             unsafe_allow_html=True)


# ================================================================== PAGE 6
def page_insights():
    st.caption("Generated automatically from the currently filtered data - change the filters to update them.")
    items = U.generate_insights(cl, sold, seg_result["sil_km"])
    for title, body in items:
        st.markdown(f'<div class="card"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)
    st.subheader("Suggested actions for the real estate company")
    st.markdown("""
1. **Segment-based campaigns:** use a different message for each named segment instead of one generic advert.
2. **Financing partnerships:** promote loan and instalment offers where loan applications are highest.
3. **Regional budget:** shift part of the budget to regions with a higher investment share (after checking sample size).
4. **Channel review:** compare cost per client of Website, Agency and Client referrals, since buyer quality is similar.
5. **Improve data collection:** record income, budget, enquiries and repeat purchases to build stronger investor segments.
""")


{PAGES[0]: page_overview, PAGES[1]: page_behaviour, PAGES[2]: page_geography,
 PAGES[3]: page_property, PAGES[4]: page_ml, PAGES[5]: page_insights}[page]()
