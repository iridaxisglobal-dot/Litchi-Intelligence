
# ==========================================================
# Streamlit Story Dashboard for Lichi Image Analysis Outputs
# ==========================================================
# How to run:
# 1. Save this file as app.py in the same folder where your notebook runs.
# 2. Make sure the folder "research_outputs" exists and contains the CSV outputs.
# 3. Run:
#       streamlit run app.py
#
# Recommended packages:
#       pip install streamlit pandas numpy plotly scipy
# ==========================================================

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="LITCHI ORCHARD INTELLIGENCE SYSTEM (LOIS)",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
CSV_REGISTRY = {
    "environment_info": "environment_info.json",
    "dataset_statistics": "dataset_statistics.csv",
    "class_distribution": "class_distribution.csv",
    "dataset_label_audit": "dataset_label_audit.csv",
    "epoch_metrics": "epoch_training_validation_metrics.csv",
    "validation_metrics": "validation_metrics_summary.csv",
    "per_class_validation": "per_class_validation_metrics.csv",
    "fruit_predictions": "fruit_level_predictions_and_features.csv",
    "image_predictions": "image_level_predictions_and_agro_features.csv",
    "tree_prediction_summary": "tree_level_prediction_summary.csv",
    "fruit_disease_pattern": "fruit_level_disease_pattern_estimation.csv",
    "fruit_disease_index": "fruit_level_composite_disease_index.csv",
    "image_disease_index": "image_level_composite_disease_index.csv",
    "production_estimation": "disease_adjusted_image_level_future_marketable_production_estimation.csv",
    "tree_production_summary": "tree_level_disease_adjusted_production_summary.csv",
    "production_summary": "disease_loss_production_summary_before_after.csv",
    "disease_by_maturity": "disease_pattern_summary_by_maturity_stage.csv",
    "manual_iou_rows": "manual_iou50_detection_validation_rows.csv",
    "manual_iou_summary": "manual_iou50_validation_summary_by_class.csv",
    "prediction_maturity_stats": "prediction_statistics_by_maturity_class.csv",
    "image_research_stats": "image_level_research_summary_statistics.csv",
    "observation_counts": "observation_count_summary.csv",
}


def fmt_num(value, decimals=2, suffix=""):
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        if abs(float(value)) >= 1000000:
            return f"{float(value)/1000000:.{decimals}f}M{suffix}"
        if abs(float(value)) >= 1000:
            return f"{float(value):,.{decimals}f}{suffix}"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return str(value)


def clean_column_name(col):
    return str(col).replace("_", " ").title()


def has_cols(df, cols):
    return df is not None and not df.empty and all(c in df.columns for c in cols)


@st.cache_data(show_spinner=False)
def read_csv_safe(base_dir_str, filename):
    base_dir = Path(base_dir_str)
    path = base_dir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json_safe(base_dir_str, filename):
    base_dir = Path(base_dir_str)
    path = base_dir / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_all_outputs(output_dir):
    data = {}
    for key, filename in CSV_REGISTRY.items():
        if filename.endswith(".csv"):
            data[key] = read_csv_safe(str(output_dir), filename)
        elif filename.endswith(".json"):
            data[key] = read_json_safe(str(output_dir), filename)
    return data


def show_missing_outputs(data):
    rows = []
    for key, filename in CSV_REGISTRY.items():
        item = data.get(key)
        exists = bool(item) if isinstance(item, dict) else (item is not None and not item.empty)
        rows.append({
            "output_key": key,
            "file_name": filename,
            "status": "Available" if exists else "Missing / not generated",
            "rows": "" if isinstance(item, dict) else (0 if item is None else len(item))
        })
    return pd.DataFrame(rows)


def kpi(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def numeric_cols(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()


def safe_mean(df, col):
    if has_cols(df, [col]):
        return df[col].mean()
    return np.nan


def safe_sum(df, col):
    if has_cols(df, [col]):
        return df[col].sum()
    return np.nan


def safe_max(df, col):
    if has_cols(df, [col]):
        return df[col].max()
    return np.nan


def safe_nunique(df, col):
    if has_cols(df, [col]):
        return df[col].nunique()
    return np.nan

# ==========================================================
# MODIFIED plot_bar()
# ==========================================================

def plot_bar(df, x, y, title="", color=None, text_auto=True):
    if not has_cols(df, [x, y]):
        st.info(f"Required columns missing for chart: {x}, {y}")
        return
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color if color in df.columns else None,
        text_auto=text_auto
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"bar_{x}_{y}_{title}",
        config={
            "displaylogo": False
        }
    )
# ==========================================================
# MODIFIED plot_line()
# ==========================================================
def plot_line(df, x, y_cols, title=""):
    if df is None or df.empty or x not in df.columns:
        st.info("No data available for this line chart.")
        return
    fig = go.Figure()
    for y in y_cols:
        if y in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="lines+markers",
                    name=clean_column_name(y)
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"line_{x}_{'_'.join(y_cols)}_{title}",
        config={
            "displaylogo": False
        }
    )
# ==========================================================
# MODIFIED plot_hist()
# ==========================================================
def plot_hist(df, col, title=""):
    if not has_cols(df, [col]):
        st.info(f"Required column missing: {col}")
        return
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        marginal="box"
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10)
    )
   
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"hist_{col}_{title}",
        config={
            "displaylogo": False
        }
    )
# ==========================================================
# MODIFIED plot_scatter()
# ==========================================================
def plot_scatter(df, x, y, title="", color=None, size=None):
    if not has_cols(df, [x, y]):
        st.info(f"Required columns missing for scatter plot: {x}, {y}")
        return
    plot_df = df.copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    if plot_df.empty:
        st.info(f"No valid rows available for scatter plot: {title}")
        return
    safe_size = None
    if size is not None and size in plot_df.columns:
        plot_df[size] = pd.to_numeric(plot_df[size], errors="coerce")
        valid_size = plot_df[size].replace([np.inf, -np.inf], np.nan)
        if valid_size.notna().any():
            fallback_size = valid_size[valid_size.notna()].median()
            if pd.isna(fallback_size) or fallback_size < 0:
                fallback_size = 1
            plot_df[size] = valid_size.fillna(fallback_size)
            plot_df[size] = plot_df[size].clip(lower=0)
            if plot_df[size].max() == 0:
                plot_df[size] = 1
            safe_size = size
    safe_color = color if color is not None and color in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=safe_color,
        size=safe_size
    )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"scatter_{x}_{y}_{title}",
        config={
            "displaylogo": False
        }
    )
def describe_numeric(df):
    if df is None or df.empty:
        return pd.DataFrame()

    cols = numeric_cols(df)
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        mean_val = non_null.mean()
        std_val = non_null.std()
        cv = std_val / mean_val if mean_val not in [0, np.nan] and pd.notna(mean_val) else np.nan

        rows.append({
            "metric": col,
            "count": int(non_null.count()),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "mean": mean_val,
            "median": non_null.median(),
            "std": std_val,
            "cv": cv,
            "min": non_null.min(),
            "q1": q1,
            "q3": q3,
            "max": non_null.max(),
            "iqr": q3 - q1,
            "skewness": non_null.skew(),
            "kurtosis": non_null.kurt()
        })
    return pd.DataFrame(rows)


def correlation_table(df, target_col):
    if df is None or df.empty or target_col not in df.columns:
        return pd.DataFrame()

    cols = numeric_cols(df)
    if target_col not in cols:
        return pd.DataFrame()

    corr = df[cols].corr(numeric_only=True)[target_col].dropna().sort_values(ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns = ["metric", "correlation_with_target"]
    corr_df = corr_df[corr_df["metric"] != target_col]
    return corr_df


def apply_common_filters(data):
    """Filter key dataframes using sidebar filters when matching columns exist."""
    filtered = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in data.items()}

    production = filtered.get("production_estimation", pd.DataFrame())
    fruit = filtered.get("fruit_predictions", pd.DataFrame())
    image = filtered.get("image_predictions", pd.DataFrame())
    tree_prod = filtered.get("tree_production_summary", pd.DataFrame())

    st.sidebar.markdown("### Filters")

    scenario_values = []
    if has_cols(production, ["scenario"]):
        scenario_values = sorted(production["scenario"].dropna().unique().tolist())
    selected_scenarios = st.sidebar.multiselect(
        "Production scenario",
        options=scenario_values,
        default=scenario_values
    ) if scenario_values else []

    maturity_values = []
    if has_cols(fruit, ["predicted_maturity_class"]):
        maturity_values = sorted(fruit["predicted_maturity_class"].dropna().unique().tolist())
    selected_maturity = st.sidebar.multiselect(
        "Maturity class",
        options=maturity_values,
        default=maturity_values
    ) if maturity_values else []

    orchard_values = []
    for df in [production, image, tree_prod]:
        if has_cols(df, ["orchard_id"]):
            orchard_values = sorted(df["orchard_id"].dropna().unique().tolist())
            break

    selected_orchards = st.sidebar.multiselect(
        "Orchard",
        options=orchard_values,
        default=orchard_values
    ) if orchard_values else []

    for key in ["production_estimation", "tree_production_summary"]:
        df = filtered.get(key, pd.DataFrame())
        if has_cols(df, ["scenario"]) and selected_scenarios:
            filtered[key] = df[df["scenario"].isin(selected_scenarios)]

    for key in ["fruit_predictions", "fruit_disease_pattern", "fruit_disease_index"]:
        df = filtered.get(key, pd.DataFrame())
        if has_cols(df, ["predicted_maturity_class"]) and selected_maturity:
            filtered[key] = df[df["predicted_maturity_class"].isin(selected_maturity)]

    for key in ["image_predictions", "production_estimation", "tree_prediction_summary", "tree_production_summary"]:
        df = filtered.get(key, pd.DataFrame())
        if has_cols(df, ["orchard_id"]) and selected_orchards:
            filtered[key] = df[df["orchard_id"].isin(selected_orchards)]

    return filtered
# ==========================================================
# COMMON CHART TITLE FUNCTION
# ==========================================================

def chart_title(title):
    st.markdown(
        f"""
        <div style="
            font-size:20px;
            font-weight:700;
            margin-top:8px;
            margin-bottom:-8px;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------------
# Sidebar configuration
# ----------------------------------------------------------
st.sidebar.title("🍒 Lichi AI Dashboard")

default_output_dir = Path.cwd() / "research_outputs"
output_dir_input = st.sidebar.text_input(
    "Research output folder",
    value=str(default_output_dir)
)

OUTPUT_DIR = Path(output_dir_input).expanduser()
data = load_all_outputs(OUTPUT_DIR)
data = apply_common_filters(data)

st.sidebar.markdown("---")
st.sidebar.caption("This dashboard reads CSV/JSON outputs generated by the YOLO lichi image-analysis notebook.")

if not OUTPUT_DIR.exists():
    st.error(f"Output folder not found: {OUTPUT_DIR}")
    st.stop()


# ----------------------------------------------------------
# Load frequently used dataframes
# ----------------------------------------------------------
env = data.get("environment_info", {})
dataset_stats = data.get("dataset_statistics", pd.DataFrame())
class_dist = data.get("class_distribution", pd.DataFrame())
epoch_metrics = data.get("epoch_metrics", pd.DataFrame())
validation_metrics = data.get("validation_metrics", pd.DataFrame())
per_class_validation = data.get("per_class_validation", pd.DataFrame())
fruit_pred = data.get("fruit_predictions", pd.DataFrame())
image_pred = data.get("image_predictions", pd.DataFrame())
tree_pred = data.get("tree_prediction_summary", pd.DataFrame())
fruit_disease = data.get("fruit_disease_index", pd.DataFrame())
image_disease = data.get("image_disease_index", pd.DataFrame())
production = data.get("production_estimation", pd.DataFrame())
tree_production = data.get("tree_production_summary", pd.DataFrame())
production_summary = data.get("production_summary", pd.DataFrame())
disease_by_maturity = data.get("disease_by_maturity", pd.DataFrame())
manual_iou_summary = data.get("manual_iou_summary", pd.DataFrame())
prediction_maturity_stats = data.get("prediction_maturity_stats", pd.DataFrame())
image_research_stats = data.get("image_research_stats", pd.DataFrame())
observation_counts = data.get("observation_counts", pd.DataFrame())


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px;
        font-weight: 800;
        color: #b22222;
        text-align: center;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 2px;
    }

    .tagline {
        font-size: 16px;
        color: #666666;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 0px;
        line-height: 1.3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="big-title">🍒 LITCHI ORCHARD INTELLIGENCE SYSTEM (LOIS)</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Story & Statistical Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="tagline">AI Powered Orchard Intelligence, Disease Analytics & Production Forecasting Platform</p>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tabs = st.tabs([
    "1. Executive Story",
    "2. Maturity & Fruit Detection",
    "3. Disease & Quality",
    "4. Production Estimate",
    "5. Tree / Orchard View",
    "6. Dataset & Model",
    "7. Statistical Analysis",
    "8. Data Tables"
])


# ==========================================================
# Tab 1 — Executive Story
# ==========================================================
with tabs[0]:
    st.subheader("Executive Story")

    total_images = safe_nunique(image_pred, "image_name")
    total_fruits = safe_sum(image_pred, "detected_fruit_count")
    avg_fruits = safe_mean(image_pred, "detected_fruit_count")
    avg_conf = safe_mean(image_pred, "avg_detection_confidence")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi("Images scored", fmt_num(total_images, 0))
    with col2:
        kpi("Detected fruits", fmt_num(total_fruits, 0))
    with col3:
        kpi("Avg fruits / image", fmt_num(avg_fruits, 2))
    with col4:
        kpi("Avg detection confidence", fmt_num(avg_conf * 100 if pd.notna(avg_conf) and avg_conf <= 1 else avg_conf, 2, "%"))

    story_parts = []

    if has_cols(image_pred, ["red_count", "half_count", "green_count", "young_count"]):
        maturity_totals = {
            "Young": image_pred["young_count"].sum(),
            "Green": image_pred["green_count"].sum(),
            "Half": image_pred["half_count"].sum(),
            "Red": image_pred["red_count"].sum()
        }
        dominant_stage = max(maturity_totals, key=maturity_totals.get) if sum(maturity_totals.values()) > 0 else "NA"
        story_parts.append(f"The current crop image set is dominated by **{dominant_stage}** stage fruits based on detected counts.")

    if has_cols(image_disease, ["avg_composite_disease_index", "disease_loss_pct"]):
        disease_idx = image_disease["avg_composite_disease_index"].mean()
        disease_loss = image_disease["disease_loss_pct"].mean()
        story_parts.append(
            f"Average visible disease index is **{fmt_num(disease_idx, 2)} / 100**, with an estimated average disease-loss factor of **{fmt_num(disease_loss, 2)}%**."
        )

    if has_cols(production_summary, ["scenario", "total_future_production_after_disease_kg"]):
        base_row = production_summary[production_summary["scenario"].astype(str).str.lower().eq("base")]
        if not base_row.empty:
            base_prod = base_row["total_future_production_after_disease_kg"].iloc[0]
            story_parts.append(
                f"Under the **Base** scenario, disease-adjusted future marketable production is estimated at **{fmt_num(base_prod, 2)} kg**."
            )

    if has_cols(manual_iou_summary, ["f1_iou50"]):
        avg_f1 = manual_iou_summary["f1_iou50"].mean()
        story_parts.append(
            f"Manual IoU@0.50 validation shows an average class-level F1 of **{fmt_num(avg_f1 * 100, 2)}%**."
        )

    if story_parts:
        st.markdown('<div class="story-box">' + "<br>".join(story_parts) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Run the notebook first so the dashboard can generate the executive story from CSV outputs.")

    left, right = st.columns(2)

    with left:
        if has_cols(image_pred, ["young_count", "green_count", "half_count", "red_count"]):
            maturity_df = pd.DataFrame({
                "maturity_stage": ["Young", "Green", "Half", "Red"],
                "detected_count": [
                    image_pred["young_count"].sum(),
                    image_pred["green_count"].sum(),
                    image_pred["half_count"].sum(),
                    image_pred["red_count"].sum()
                ]
            })
            chart_title("Detected Fruits by Maturity Stage")
            plot_bar(
                maturity_df,
                "maturity_stage",
                "detected_count"
            )
        else:
            st.info("Maturity count columns are not available yet.")
            
    with right:
        if has_cols(production_summary, ["scenario", "total_future_production_after_disease_kg"]):
            chart_title("Disease-adjusted Future Production by Scenario")
            plot_bar(
                production_summary,
                "scenario",
                "total_future_production_after_disease_kg",
            )
        else:
            st.info("Production summary is not available yet.")

    st.markdown("#### Recommended decision reading")
    st.write(
        """
        Use this page to understand the crop condition in one flow: detection volume → maturity mix → disease pressure → production outlook.
        For research reporting, validate model quality in the Dataset & Model tab before using the production estimate commercially.
        """
    )


# ==========================================================
# Tab 2 — Dataset & Model
# ==========================================================
with tabs[1]:
    st.subheader("Dataset & Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    if not dataset_stats.empty:
        flat_stats = dataset_stats.copy()
        # Supports either key-value format or single-row summary format
        with c1:
            value = flat_stats.iloc[0, 1] if flat_stats.shape[1] >= 2 else len(flat_stats)
            kpi("Dataset stats rows", fmt_num(len(flat_stats), 0))
    else:
        with c1:
            kpi("Dataset stats", "NA")

    with c2:
        kpi("Classes", fmt_num(safe_nunique(class_dist, "class_name") if "class_name" in class_dist.columns else len(class_dist), 0))
    with c3:
        kpi("Manual validation classes", fmt_num(len(manual_iou_summary), 0))
    with c4:
        kpi("Epoch records", fmt_num(len(epoch_metrics), 0))

    left, right = st.columns(2)

    with left:
        st.markdown("#### Class distribution")
        if not class_dist.empty:
            y_col = "object_count" if "object_count" in class_dist.columns else class_dist.select_dtypes(include=[np.number]).columns.tolist()
            if isinstance(y_col, list) and y_col:
                y_col = y_col[0]
            x_col = "class_name" if "class_name" in class_dist.columns else class_dist.columns[0]
            if isinstance(y_col, str):
                plot_bar(class_dist, x_col, y_col, "Training Label Class Distribution")
            else:
                st.dataframe(class_dist, use_container_width=True)
        else:
            st.info("class_distribution.csv not available.")

    with right:
        st.markdown("#### Manual IoU@0.50 validation")
        if has_cols(manual_iou_summary, ["class_name", "precision_iou50", "recall_iou50", "f1_iou50"]):
            fig = go.Figure()
            for metric in ["precision_iou50", "recall_iou50", "f1_iou50"]:
                fig.add_trace(go.Bar(
                    x=manual_iou_summary["class_name"],
                    y=manual_iou_summary[metric],
                    name=clean_column_name(metric)
                ))
            fig.update_layout(
                barmode="group",
                title="Manual Detection Metrics by Class",
                yaxis_title="Score",
                height=420,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif not manual_iou_summary.empty:
            st.dataframe(manual_iou_summary, use_container_width=True)
        else:
            st.info("manual_iou50_validation_summary_by_class.csv not available.")

    st.markdown("#### Epoch-wise training curves")
    if not epoch_metrics.empty:
        x_col = "epoch" if "epoch" in epoch_metrics.columns else epoch_metrics.columns[0]
        loss_cols = [c for c in epoch_metrics.columns if "loss" in c.lower()]
        metric_cols = [c for c in epoch_metrics.columns if any(k in c.lower() for k in ["map", "precision", "recall", "f1"])]

        if loss_cols:
            plot_line(epoch_metrics, x_col, loss_cols[:5], "Training Loss Curves")
        if metric_cols:
            plot_line(epoch_metrics, x_col, metric_cols[:5], "Validation / Model Metric Curves")
        with st.expander("Raw epoch metrics"):
            st.dataframe(epoch_metrics, use_container_width=True)
    else:
        st.info("epoch_training_validation_metrics.csv not available. If training used val=False, validation metrics will appear in the separate validation files.")


# ==========================================================
# Tab 3 — Maturity & Detection
# ==========================================================
with tabs[2]:
    st.subheader("Maturity & Fruit Detection Analysis")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Fruit-level rows", fmt_num(len(fruit_pred), 0))
    with c2:
        kpi("Image-level rows", fmt_num(len(image_pred), 0))
    with c3:
        kpi("Avg maturity index", fmt_num(safe_mean(image_pred, "avg_maturity_index_0_100"), 2))
    with c4:
        kpi("Max fruits in image", fmt_num(safe_max(image_pred, "detected_fruit_count"), 0))

    left, right = st.columns(2)

    with left:
        if has_cols(fruit_pred, ["predicted_maturity_class"]):
            maturity_counts = (
                fruit_pred["predicted_maturity_class"]
                .value_counts()
                .reset_index()
            )
            maturity_counts.columns = ["predicted_maturity_class", "fruit_count"]
            chart_title("Fruit Count by Predicted Maturity")
            plot_bar(
                maturity_counts,
                "predicted_maturity_class",
                "fruit_count"
            )
        else:
            st.info("Fruit maturity predictions not available.")

    with right:
        if has_cols(prediction_maturity_stats, ["predicted_maturity_class", "avg_confidence"]):
            plot_bar(
                prediction_maturity_stats,
                "predicted_maturity_class",
                "avg_confidence",
                "Average Detection Confidence by Maturity Class"
            )
        elif has_cols(fruit_pred, ["predicted_maturity_class", "confidence"]):
            conf_by_class = fruit_pred.groupby("predicted_maturity_class", dropna=False)["confidence"].mean().reset_index()
            plot_bar(conf_by_class, "predicted_maturity_class", "confidence", "Average Detection Confidence by Maturity Class")
        else:
            st.info("Confidence by maturity class not available.")

    left2, right2 = st.columns(2)

    with left2:
        chart_title("Distribution of Detected Fruits per Image")
        plot_hist(
            image_pred,
            "detected_fruit_count"
        )
    with right2:
        chart_title("Distribution of Fruit Detection Confidence")
        plot_hist(
            fruit_pred,
            "confidence"
        )
    if has_cols(image_pred, ["detected_fruit_count", "avg_maturity_index_0_100"]):
        chart_title("Detected Fruit Count vs Average Maturity Index")
        plot_scatter(
            image_pred,
            "detected_fruit_count",
            "avg_maturity_index_0_100",
            color="orchard_id",
            size="avg_detection_confidence"
        )

    with st.expander("Prediction statistics by maturity class"):
        if not prediction_maturity_stats.empty:
            st.dataframe(prediction_maturity_stats, use_container_width=True)
        else:
            st.info("prediction_statistics_by_maturity_class.csv not available.")


# ==========================================================
# Tab 4 — Disease & Quality
# ==========================================================
with tabs[3]:
    st.subheader("Disease, Quality & Visual Stress Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Avg disease index", fmt_num(safe_mean(image_disease, "avg_composite_disease_index"), 2))
    with c2:
        kpi("Avg disease loss %", fmt_num(safe_mean(image_disease, "disease_loss_pct"), 2, "%"))
    with c3:
        kpi("Avg stress score", fmt_num(safe_mean(image_pred, "visible_stress_score"), 2))
    with c4:
        kpi("Avg canopy gap %", fmt_num(safe_mean(image_pred, "canopy_gap_pct"), 2, "%"))

    left, right = st.columns(2)

    with left:
        if has_cols(fruit_disease, ["disease_severity_band"]):
            sev = fruit_disease["disease_severity_band"].value_counts().reset_index()
            sev.columns = ["disease_severity_band", "fruit_count"]
            plot_bar(sev, "disease_severity_band", "fruit_count", "Fruit Count by Disease Severity Band")
        else:
            st.info("Fruit-level disease severity not available.")

    with right:
        if has_cols(disease_by_maturity, ["predicted_maturity_class", "avg_disease_risk_score"]):
            plot_bar(
                disease_by_maturity,
                "predicted_maturity_class",
                "avg_disease_risk_score",
                "Average Disease Risk Score by Maturity Stage"
            )
        else:
            st.info("Disease-by-maturity summary not available.")

    left2, right2 = st.columns(2)

    with left2:
        plot_hist(image_disease, "avg_composite_disease_index", "Distribution of Image-level Composite Disease Index")

    with right2:
        if has_cols(image_disease, ["moderate_or_high_disease_pct"]):
            plot_hist(image_disease, "moderate_or_high_disease_pct", "Moderate-or-High Disease % by Image")
        else:
            plot_hist(image_pred, "visible_stress_score", "Visible Stress Score Distribution")

    if has_cols(image_disease, ["avg_composite_disease_index", "disease_loss_pct"]):
        plot_scatter(
            image_disease,
            "avg_composite_disease_index",
            "disease_loss_pct",
            "Composite Disease Index vs Estimated Disease Loss %",
            color="high_disease_pct"
        )

    st.markdown("#### Disease by maturity table")
    if not disease_by_maturity.empty:
        st.dataframe(disease_by_maturity, use_container_width=True)
    else:
        st.info("disease_pattern_summary_by_maturity_stage.csv not available.")


# ==========================================================
# Tab 5 — Production
# ==========================================================
with tabs[4]:
    
    st.subheader("Disease-adjusted Production Estimate")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Production rows", fmt_num(len(production), 0))
    with c2:
        kpi("Future after disease kg", fmt_num(safe_sum(production, "disease_adjusted_marketable_future_production_kg"), 2))
    with c3:
        kpi("Future disease loss kg", fmt_num(safe_sum(production, "future_production_disease_loss_kg"), 2))
    with c4:
        kpi("Current harvest-ready kg", fmt_num(safe_sum(production, "disease_adjusted_current_harvest_ready_kg"), 2))

    if has_cols(production_summary, ["scenario"]):
        st.markdown("#### Before vs after disease loss by scenario")
        cols_needed = [
            "total_future_production_before_disease_kg",
            "total_future_disease_loss_kg",
            "total_future_production_after_disease_kg"
        ]
        available_cols = [c for c in cols_needed if c in production_summary.columns]
        if available_cols:
            fig = go.Figure()
            for col in available_cols:
                fig.add_trace(go.Bar(
                    x=production_summary["scenario"],
                    y=production_summary[col],
                    name=clean_column_name(col)
                ))
            fig.update_layout(
                barmode="group",
                title="Future Production: Before Disease, Disease Loss, After Disease",
                yaxis_title="kg",
                height=460,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(production_summary, use_container_width=True)
    else:
        st.info("Production summary file is not available.")

    left, right = st.columns(2)

    with left:
        if has_cols(production, ["scenario", "disease_adjusted_marketable_future_production_kg"]):
            scenario_image = (
                production
                .groupby("scenario", dropna=False)
                .agg(avg_image_production_kg=("disease_adjusted_marketable_future_production_kg", "mean"))
                .reset_index()
            )
            plot_bar(scenario_image, "scenario", "avg_image_production_kg", "Average Disease-adjusted Future Production per Image")
        else:
            st.info("Scenario production columns not available.")

    with right:
        if has_cols(production, ["expected_harvestable_fruit_count", "disease_adjusted_marketable_future_production_kg"]):
            plot_scatter(
                production,
                "expected_harvestable_fruit_count",
                "disease_adjusted_marketable_future_production_kg",
                "Harvestable Fruit Count vs Future Production",
                color="scenario",
                size="disease_loss_pct"
            )
        else:
            st.info("Harvestable count and production columns not available.")

    if has_cols(production, ["avg_composite_disease_index", "disease_adjusted_marketable_future_production_kg"]):
        plot_scatter(
            production,
            "avg_composite_disease_index",
            "disease_adjusted_marketable_future_production_kg",
            "Disease Index vs Disease-adjusted Future Production",
            color="scenario",
            size="detected_fruit_count"
        )


# ==========================================================
# Tab 6 — Tree / Orchard
# ==========================================================
with tabs[5]:
    st.subheader("Tree, Orchard & Spatial View")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Trees in prediction summary", fmt_num(safe_nunique(tree_pred, "tree_global_id"), 0))
    with c2:
        kpi("Trees in production summary", fmt_num(safe_nunique(tree_production, "tree_global_id"), 0))
    with c3:
        kpi("Orchards", fmt_num(safe_nunique(tree_production, "orchard_id"), 0))
    with c4:
        kpi("Blocks", fmt_num(safe_nunique(tree_production, "block_id"), 0))

    if has_cols(tree_production, ["orchard_id", "disease_adjusted_marketable_future_production_kg"]):
        orchard_summary = (
            tree_production
            .groupby(["scenario", "orchard_id"], dropna=False)
            .agg(
                trees=("tree_global_id", "nunique") if "tree_global_id" in tree_production.columns else ("orchard_id", "count"),
                total_detected_fruits=("total_detected_fruits", "sum") if "total_detected_fruits" in tree_production.columns else ("orchard_id", "count"),
                future_production_kg=("disease_adjusted_marketable_future_production_kg", "sum"),
                avg_disease_loss_pct=("disease_loss_pct", "mean") if "disease_loss_pct" in tree_production.columns else ("orchard_id", "count")
            )
            .reset_index()
        )

        plot_bar(
            orchard_summary,
            "orchard_id",
            "future_production_kg",
            "Future Production by Orchard",
            color="scenario"
        )
        st.dataframe(orchard_summary, use_container_width=True)

    elif has_cols(tree_pred, ["orchard_id", "total_detected_fruits"]):
        orchard_summary = (
            tree_pred
            .groupby("orchard_id", dropna=False)
            .agg(total_detected_fruits=("total_detected_fruits", "sum"))
            .reset_index()
        )
        plot_bar(orchard_summary, "orchard_id", "total_detected_fruits", "Detected Fruits by Orchard")
    else:
        st.info("Tree/orchard columns not available.")

    if has_cols(tree_production, ["tree_latitude", "tree_longitude", "disease_adjusted_marketable_future_production_kg"]):
        st.markdown("#### Tree map")
        map_df = tree_production.rename(columns={
            "tree_latitude": "lat",
            "tree_longitude": "lon"
        }).dropna(subset=["lat", "lon"])
        if not map_df.empty:
            st.map(map_df[["lat", "lon"]])
        else:
            st.info("Latitude/longitude columns exist but no valid coordinates found.")

    st.markdown("#### Tree-level production detail")
    if not tree_production.empty:
        sort_col = "disease_adjusted_marketable_future_production_kg"
        show_cols = [c for c in [
            "scenario", "orchard_id", "block_id", "row_id", "tree_id", "tree_global_id",
            "total_images", "total_detected_fruits",
            "disease_adjusted_marketable_future_production_kg",
            "future_production_disease_loss_kg",
            "avg_composite_disease_index", "disease_loss_pct",
            "avg_detection_confidence"
        ] if c in tree_production.columns]
        table_df = tree_production[show_cols].copy()
        if sort_col in table_df.columns:
            table_df = table_df.sort_values(sort_col, ascending=False)
        st.dataframe(table_df, use_container_width=True, height=420)
    else:
        st.info("tree_level_disease_adjusted_production_summary.csv not available.")


# ==========================================================
# Tab 7 — Statistical Analysis
# ==========================================================
with tabs[6]:
    
    st.subheader("Statistical Analysis Parameter Dashboard")
    dataset_options = {
        "Image-level predictions": image_pred,
        "Fruit-level predictions": fruit_pred,
        "Image-level disease index": image_disease,
        "Fruit-level disease index": fruit_disease,
        "Production estimation": production,
        "Tree production summary": tree_production,
        "Manual IoU summary": manual_iou_summary,
    }

    available_options = {k: v for k, v in dataset_options.items() if v is not None and not v.empty}

    if not available_options:
        st.info("No statistical datasets are available yet. Run the notebook outputs first.")
    else:
        selected_dataset_name = st.selectbox("Select dataset for statistical analysis", list(available_options.keys()))
        selected_df = available_options[selected_dataset_name]

        st.markdown("#### Descriptive statistics")
        desc = describe_numeric(selected_df)
        if not desc.empty:
            st.dataframe(desc, use_container_width=True, height=420)
        else:
            st.info("No numeric columns available for descriptive statistics.")

        st.markdown("#### Correlation analysis")
        num_cols = numeric_cols(selected_df)
        if len(num_cols) >= 2:
            target_default = None
            preferred_targets = [
                "disease_adjusted_marketable_future_production_kg",
                "detected_fruit_count",
                "avg_composite_disease_index",
                "confidence",
                "f1_iou50"
            ]
            for t in preferred_targets:
                if t in num_cols:
                    target_default = t
                    break
            if target_default is None:
                target_default = num_cols[0]

            target_col = st.selectbox(
                "Select target metric",
                num_cols,
                index=num_cols.index(target_default)
            )

            corr_df = correlation_table(selected_df, target_col)
            left, right = st.columns(2)

            with left:
                if not corr_df.empty:
                    st.dataframe(corr_df, use_container_width=True, height=360)
                else:
                    st.info("Could not calculate target correlations.")

            with right:
                corr_matrix = selected_df[num_cols].corr(numeric_only=True)
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Matrix"
                )
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Target distribution")
            plot_hist(selected_df, target_col, f"Distribution of {clean_column_name(target_col)}")

            predictor_options = [c for c in num_cols if c != target_col]
            if predictor_options:
                x_col = st.selectbox("Scatter X metric", predictor_options)
                plot_scatter(
                    selected_df,
                    x_col,
                    target_col,
                    f"{clean_column_name(x_col)} vs {clean_column_name(target_col)}"
                )
        else:
            st.info("At least two numeric columns are required for correlation analysis.")

        st.markdown("#### Outlier analysis")
        if num_cols:
            outlier_col = st.selectbox("Select metric for IQR outlier check", num_cols, key="outlier_metric")
            s = pd.to_numeric(selected_df[outlier_col], errors="coerce").dropna()

            if len(s) > 0:
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (pd.to_numeric(selected_df[outlier_col], errors="coerce") < lower) | (
                    pd.to_numeric(selected_df[outlier_col], errors="coerce") > upper
                )
                outliers = selected_df[outlier_mask]

                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    kpi("Q1", fmt_num(q1, 3))
                with oc2:
                    kpi("Q3", fmt_num(q3, 3))
                with oc3:
                    kpi("Outlier rows", fmt_num(len(outliers), 0))

                st.dataframe(outliers.head(200), use_container_width=True)
        else:
            st.info("No numeric columns available for outlier analysis.")

        if SCIPY_AVAILABLE and len(num_cols) >= 2:
            st.markdown("#### Optional significance test")
            test_cols = st.multiselect("Select two numeric columns for Pearson correlation test", num_cols, default=num_cols[:2])
            if len(test_cols) == 2:
                pair = selected_df[test_cols].dropna()
                if len(pair) >= 3:
                    r, p_value = stats.pearsonr(pair[test_cols[0]], pair[test_cols[1]])
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        kpi("Pearson r", fmt_num(r, 4))
                    with tc2:
                        kpi("p-value", fmt_num(p_value, 6))
                    with tc3:
                        interpretation = "Statistically significant" if p_value < 0.05 else "Not significant at 5%"
                        kpi("Interpretation", interpretation)
                else:
                    st.info("Not enough paired rows for Pearson test.")


# ==========================================================
# Tab 8 — Data Tables
# ==========================================================
with tabs[7]:
    
    st.subheader("Raw Output Tables")
    table_options = {
        "Dataset statistics": dataset_stats,
        "Class distribution": class_dist,
        "Observation counts": observation_counts,
        "Validation metrics": validation_metrics,
        "Per-class validation": per_class_validation,
        "Manual IoU summary": manual_iou_summary,
        "Fruit predictions": fruit_pred,
        "Image predictions": image_pred,
        "Tree prediction summary": tree_pred,
        "Fruit disease index": fruit_disease,
        "Image disease index": image_disease,
        "Production estimation": production,
        "Tree production summary": tree_production,
        "Production summary": production_summary,
        "Disease by maturity": disease_by_maturity,
        "Prediction maturity stats": prediction_maturity_stats,
        "Image research stats": image_research_stats,
        "Epoch metrics": epoch_metrics,
    }

    available_tables = {k: v for k, v in table_options.items() if v is not None and not v.empty}

    if not available_tables:
        st.info("No CSV outputs found in the selected research_outputs folder.")
    else:
        selected_table_name = st.selectbox("Select output table", list(available_tables.keys()))
        selected_table = available_tables[selected_table_name]

        st.write(f"Rows: **{len(selected_table):,}**, Columns: **{selected_table.shape[1]:,}**")

        search_text = st.text_input("Search text inside table", value="")
        display_table = selected_table.copy()

        if search_text:
            mask = display_table.astype(str).apply(
                lambda col: col.str.contains(search_text, case=False, na=False)
            ).any(axis=1)
            display_table = display_table[mask]

        st.dataframe(display_table, use_container_width=True, height=520)

        csv_bytes = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download filtered {selected_table_name}",
            data=csv_bytes,
            file_name=f"{selected_table_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

    st.markdown("#### Environment information")
    if env:
        st.json(env)
    else:
        st.info("environment_info.json not available.")
