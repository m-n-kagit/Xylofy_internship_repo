import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="SuperStore Sales Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

DATASET_NAME = "rohitsahoo/sales-forecasting"
DATA_FILENAME = "train.csv"
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}

MODEL_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)


def _dataset_path() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to load the dataset used in analysis2.ipynb."
        ) from exc

    dataset_root = Path(kagglehub.dataset_download(DATASET_NAME))
    return dataset_root / DATA_FILENAME

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = pd.read_csv(_dataset_path())
    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Ship Date"] = pd.to_datetime(data["Ship Date"], errors="coerce")
    data = data.dropna().copy()
    data = data.rename(columns={"Sales": "Sales($)"})
    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["Day"] = data["Order Date"].dt.day
    data["Day of Week"] = data["Order Date"].dt.dayofweek
    data["Quarter"] = data["Order Date"].dt.quarter
    data["Season"] = data["Month"].map(SEASON_MAP)
    data["Shipping Time"] = (data["Ship Date"] - data["Order Date"]).dt.days
    return data


@st.cache_data(show_spinner=False)
def yearly_sales(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Year")["Sales($)"].sum().sort_index()


@st.cache_data(show_spinner=False)
def monthly_sales(df: pd.DataFrame) -> pd.Series:
    return df.resample("MS", on="Order Date")["Sales($)"].sum()


@st.cache_data(show_spinner=False)
def region_category_table(df: pd.DataFrame, regions: tuple[str, ...], categories: tuple[str, ...]) -> pd.DataFrame:
    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)].copy()
    pivot = filtered.pivot_table(
        index="Region",
        columns="Category",
        values="Sales($)",
        aggfunc="sum",
        fill_value=0,
    )
    return pivot.reindex(index=list(regions), columns=list(categories), fill_value=0)


@st.cache_data(show_spinner=False)
def build_feature_frame(series: pd.Series) -> pd.DataFrame:
    frame = series.to_frame(name="y").copy()
    frame["Lag 1"] = frame["y"].shift(1)
    frame["Lag 2"] = frame["y"].shift(2)
    frame["Lag 3"] = frame["y"].shift(3)
    frame["Rolling Mean 3"] = frame["y"].shift(1).rolling(window=3).mean()
    frame["Month"] = frame.index.month
    frame["Quarter"] = frame.index.quarter
    frame["Season"] = frame["Month"].map(SEASON_MAP)
    frame = pd.get_dummies(frame, columns=["Season"], drop_first=False)
    return frame.dropna()


def build_next_row(history: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    row = pd.DataFrame(index=[next_date])
    row["y"] = np.nan
    row["Lag 1"] = history.iloc[-1]
    row["Lag 2"] = history.iloc[-2]
    row["Lag 3"] = history.iloc[-3]
    row["Rolling Mean 3"] = history.iloc[-3:].mean()
    row["Month"] = next_date.month
    row["Quarter"] = next_date.quarter
    row["Season"] = SEASON_MAP[next_date.month]
    return pd.get_dummies(row, columns=["Season"], drop_first=False)


def forecast_series(series: pd.Series, horizon: int) -> tuple[pd.DataFrame, float, float]:
    if len(series) < 8:
        raise ValueError("Not enough monthly observations for forecasting.")

    holdout = min(3, max(1, len(series) // 6))
    if len(series) <= holdout + 3:
        holdout = 1

    train_series = series.iloc[:-holdout]
    test_series = series.iloc[-holdout:]

    train_frame = build_feature_frame(train_series)
    feature_columns = [column for column in train_frame.columns if column != "y"]

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(train_frame[feature_columns], train_frame["y"])

    working_history = train_series.copy()
    holdout_predictions = []
    for actual_date in test_series.index:
        next_row = build_next_row(working_history, actual_date)
        next_row = next_row.reindex(columns=train_frame.columns, fill_value=0)
        prediction = float(model.predict(next_row[feature_columns])[0])
        holdout_predictions.append(prediction)
        working_history.loc[actual_date] = prediction

    mae = mean_absolute_error(test_series, holdout_predictions)
    rmse = math.sqrt(mean_squared_error(test_series, holdout_predictions))

    full_frame = build_feature_frame(series)
    full_feature_columns = [column for column in full_frame.columns if column != "y"]
    full_model = XGBRegressor(**MODEL_PARAMS)
    full_model.fit(full_frame[full_feature_columns], full_frame["y"])

    future_history = series.copy()
    future_rows = []
    last_date = future_history.index.max()
    for month_ahead in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(months=1)
        next_row = build_next_row(future_history, next_date)
        next_row = next_row.reindex(columns=full_frame.columns, fill_value=0)
        prediction = float(full_model.predict(next_row[full_feature_columns])[0])
        future_rows.append({
            "Month Ahead": month_ahead,
            "Forecast Date": next_date,
            "Predicted Sales($)": prediction,
        })
        future_history.loc[next_date] = prediction
        last_date = next_date

    return pd.DataFrame(future_rows), mae, rmse


@st.cache_data(show_spinner=False)
def weekly_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_sales = df.resample("W", on="Order Date")["Sales($)"].sum().reset_index()
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    anomaly_flags = iso_model.fit_predict(weekly_sales[["Sales($)"]])
    anomalies = weekly_sales[anomaly_flags == -1].reset_index(drop=True)
    return weekly_sales, anomalies


@st.cache_data(show_spinner=False)
def subcategory_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str], pd.DataFrame]:
    subcat_monthly = df.groupby(["Sub-Category", "Year", "Month"])["Sales($)"].sum().reset_index()
    subcat_monthly["Date"] = pd.to_datetime(subcat_monthly["Year"].astype(str) + "-" + subcat_monthly["Month"].astype(str))

    monthly_sales_pivot = subcat_monthly.pivot_table(
        index="Sub-Category",
        columns="Date",
        values="Sales($)",
        fill_value=0,
    )

    yearly_sales_frame = df.groupby(["Sub-Category", "Year"])["Sales($)"].sum().unstack(fill_value=0).sort_index(axis=1)

    subcat_features = pd.DataFrame(index=monthly_sales_pivot.index)
    subcat_features["Total Sales Volume"] = df.groupby("Sub-Category")["Sales($)"].sum()
    subcat_features["Sales Volatility"] = monthly_sales_pivot.std(axis=1)
    subcat_features["Average Order Value"] = df.groupby("Sub-Category")["Sales($)"].sum() / df.groupby("Sub-Category")["Order ID"].nunique()
    subcat_features["YoY Growth Rate (%)"] = (
        yearly_sales_frame.pct_change(axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .mean(axis=1)
        .fillna(0)
        * 100
    )
    subcat_features = subcat_features.sort_values("Total Sales Volume", ascending=False)

    feature_columns = [
        "Total Sales Volume",
        "YoY Growth Rate (%)",
        "Sales Volatility",
        "Average Order Value",
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(subcat_features[feature_columns])

    optimal_k = min(4, len(subcat_features))
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    subcat_features["Cluster"] = kmeans.fit_predict(scaled_features)

    cluster_summary = subcat_features.groupby("Cluster")[feature_columns].mean().round(2)

    cluster_label_map: dict[int, str] = {}
    assigned_clusters: set[int] = set()
    growing_cluster = cluster_summary["YoY Growth Rate (%)"].idxmax()
    cluster_label_map[growing_cluster] = "Growing Demand"
    assigned_clusters.add(growing_cluster)

    remaining_clusters = [cluster for cluster in cluster_summary.index if cluster not in assigned_clusters]
    if remaining_clusters:
        high_volume_stable_cluster = max(
            remaining_clusters,
            key=lambda cluster: (
                cluster_summary.loc[cluster, "Total Sales Volume"] - cluster_summary.loc[cluster, "Sales Volatility"]
            ),
        )
        cluster_label_map[high_volume_stable_cluster] = "High Volume, Stable Demand"
        assigned_clusters.add(high_volume_stable_cluster)
        remaining_clusters = [cluster for cluster in remaining_clusters if cluster != high_volume_stable_cluster]

    if remaining_clusters:
        low_volume_stable_cluster = min(
            remaining_clusters,
            key=lambda cluster: (
                cluster_summary.loc[cluster, "Total Sales Volume"] + cluster_summary.loc[cluster, "Sales Volatility"]
            ),
        )
        cluster_label_map[low_volume_stable_cluster] = "Low Volume, Stable Demand"
        assigned_clusters.add(low_volume_stable_cluster)
        remaining_clusters = [cluster for cluster in remaining_clusters if cluster != low_volume_stable_cluster]

    if remaining_clusters:
        cluster_label_map[remaining_clusters[0]] = "High Volume, High Volatility"

    subcat_features["Cluster Label"] = subcat_features["Cluster"].map(cluster_label_map)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    pca_frame = subcat_features.copy()
    pca_frame["PC1"] = pca_components[:, 0]
    pca_frame["PC2"] = pca_components[:, 1]

    cluster_table = subcat_features.reset_index().rename(columns={"index": "Sub-Category"})
    cluster_table = cluster_table[["Sub-Category", "Cluster", "Cluster Label", "Total Sales Volume", "Sales Volatility", "Average Order Value", "YoY Growth Rate (%)"]]
    cluster_table = cluster_table.sort_values(["Cluster", "Total Sales Volume"], ascending=[True, False])

    return subcat_features, cluster_summary, pca_frame, cluster_label_map, cluster_table


@st.cache_data(show_spinner=False)
def segment_options(df: pd.DataFrame, dimension: str) -> list[str]:
    return sorted(df[dimension].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def filtered_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Order Date", "Region", "Category", "Sub-Category", "Sales($)", "Profit"]].sort_values("Order Date", ascending=False)


def make_yearly_figure(year_sales: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_sales.index.astype(str), year_sales.values, color="#4C78A8")
    ax.set_title("Total Sales by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3, fontsize=9)
    plt.tight_layout()
    return fig


def make_monthly_figure(monthly_series: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series.index, monthly_series.values, marker="o", linewidth=2, color="#F58518")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def make_region_category_figure(pivot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Region and Category")
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def make_forecast_figure(history: pd.Series, forecast_table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history.index, history.values, color="#4C78A8", linewidth=2, label="Historical Sales")
    ax.plot(forecast_table["Forecast Date"], forecast_table["Predicted Sales($)"], marker="o", color="#E45756", linewidth=2, label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def make_anomaly_figure(weekly_sales: pd.DataFrame, anomalies: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weekly_sales["Order Date"], weekly_sales["Sales($)"], label="Weekly Sales", alpha=0.75, color="#4C78A8")
    ax.scatter(anomalies["Order Date"], anomalies["Sales($)"], color="#E45756", label="Anomalies", marker="x", s=90, linewidths=2, zorder=5)
    ax.set_title("Sales Anomaly Chart")
    ax.set_xlabel("Order Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def make_cluster_figure(pca_frame: pd.DataFrame, cluster_label_map: dict[int, str]):
    fig, ax = plt.subplots(figsize=(12, 7))
    for cluster_id, cluster_data in pca_frame.groupby("Cluster"):
        label = cluster_label_map.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            s=180,
            alpha=0.85,
            label=label,
            edgecolors="white",
            linewidths=1.0,
        )
        for _, row in cluster_data.iterrows():
            ax.annotate(row.name, (row["PC1"], row["PC2"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title("Product Sub-Category Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster Label", loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def main():
    st.title("SuperStore Sales Interactive Dashboard")
    st.caption("Built from the analysis notebook and optimized for quick exploration.")

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Could not load the notebook dataset ({DATASET_NAME}): {exc}")
        st.stop()

    year_sales = yearly_sales(df)
    month_sales = monthly_sales(df)
    weekly_sales, anomalies = weekly_anomalies(df)
    subcat_features, cluster_summary, pca_frame, cluster_label_map, cluster_table = subcategory_clusters(df)

    tab_overview, tab_forecast, tab_anomaly, tab_clusters = st.tabs([
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments",
    ])

    with tab_overview:
        st.subheader("Sales Overview Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(make_yearly_figure(year_sales), clear_figure=True)
        with col2:
            st.pyplot(make_monthly_figure(month_sales), clear_figure=True)

        st.markdown("### Sales by Region and Category")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_regions = st.multiselect("Region filter", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique()))
        with filter_col2:
            selected_categories = st.multiselect("Category filter", sorted(df["Category"].dropna().unique()), default=sorted(df["Category"].dropna().unique()))

        if not selected_regions or not selected_categories:
            st.info("Please select at least one region and one category to see the chart.")
        else:
            pivot = region_category_table(df, tuple(selected_regions), tuple(selected_categories))
            st.pyplot(make_region_category_figure(pivot), clear_figure=True)
            st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

    with tab_forecast:
        st.subheader("Forecast Explorer")
        dimension = st.selectbox("Select forecasting dimension", ["Category", "Region"])
        options = segment_options(df, dimension)
        selected_value = st.selectbox(f"Select {dimension}", options)
        horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3, step=1)

        segment_df = df[df[dimension] == selected_value].copy()
        segment_series = segment_df.set_index("Order Date")["Sales($)"].resample("MS").sum().asfreq("MS").fillna(0)

        try:
            forecast_table, mae, rmse = forecast_series(segment_series, horizon)
            st.pyplot(make_forecast_figure(segment_series, forecast_table, f"XGBoost Forecast for {dimension}: {selected_value}"), clear_figure=True)
            st.dataframe(forecast_table.style.format({"Predicted Sales($)": "{:.2f}"}), use_container_width=True)
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("MAE", f"{mae:,.2f}")
            metric_col2.metric("RMSE", f"{rmse:,.2f}")
        except Exception as exc:
            st.warning(f"Forecasting could not be completed for {selected_value}: {exc}")

    with tab_anomaly:
        st.subheader("Anomaly Report")
        st.pyplot(make_anomaly_figure(weekly_sales, anomalies), clear_figure=True)
        st.markdown("### Detected anomaly dates")
        anomaly_table = anomalies[["Order Date", "Sales($)"]].copy()
        anomaly_table["Order Date"] = anomaly_table["Order Date"].dt.date
        st.dataframe(anomaly_table.style.format({"Sales($)": "{:.2f}"}), use_container_width=True)

    with tab_clusters:
        st.subheader("Product Demand Segments")
        st.pyplot(make_cluster_figure(pca_frame, cluster_label_map), clear_figure=True)
        st.markdown("### Sub-category to cluster mapping")
        st.dataframe(cluster_table.style.format({
            "Total Sales Volume": "{:.2f}",
            "Sales Volatility": "{:.2f}",
            "Average Order Value": "{:.2f}",
            "YoY Growth Rate (%)": "{:.2f}",
        }), use_container_width=True)

        st.markdown("### Cluster summary")
        st.dataframe(cluster_summary.style.format("{:.2f}"), use_container_width=True)


if __name__ == "__main__":
    main()
    





