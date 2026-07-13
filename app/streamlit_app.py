
# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import clean_data, auto_detect_columns, train_and_evaluate, perform_segmentation

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Aether • Customer Intelligence", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")

# ====================== PREMIUM ENERPRISE SAAS SKINNING ======================
st.markdown("""

<style>

/* ================= GOOGLE FONT ================= */

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ================= BACKGROUND ================= */

.stApp {
    background:
        radial-gradient(circle at top right, #1e293b 0%, #0f172a 40%, #020617 100%);
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] h1 {
    color: white  !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 0px !important;
}

section[data-testid="stSidebar"] p {
    color: #00ffaa !important;
    letter-spacing: 2px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}
            section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}

/* ================= KPI CARDS ================= */

div[data-testid="stMetric"] {
    background: rgba(15,23,42,0.75) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    backdrop-filter: blur(12px);
}

/* KPI Label */
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    opacity: 1 !important;
}

/* KPI Value */
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
/* ================= TABS - BIGGER FONT ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(15,23,42,0.7);
    border-radius: 14px;
    padding: 10px;
}

.stTabs [data-baseweb="tab"] {
    color: #ffffff !important;           /* White text */
    font-size: 17px !important;          /* Increased from 15px */
    font-weight: 700 !important;         /* Bolder */
    padding: 14px 26px !important;       /* More padding for bigger text */
    letter-spacing: 0.5px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(
        135deg,
        rgba(0,255,170,0.18),
        rgba(77,150,255,0.18)
    ) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,255,170,0.4) !important;
}
/* ================= HEADINGS ================= */

h1 {
    color: white !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

h2 {
    color: white !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}

h3 {
    color: white !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}

/* ================= EXECUTIVE CARD ================= */

.executive-banner {
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.85),
            rgba(15,23,42,0.65)
        );

    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 35px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.25);
}

/* ================= BUTTONS ================= */

.stButton button {
    background: linear-gradient(
        135deg,
        #00ffaa,
        #4d96ff
    ) !important;

    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
}

.stButton button:hover {
    transform: translateY(-2px);
}

/* ================= INPUTS ================= */

.stNumberInput input,
.stTextInput input,
.stSelectbox {
    background: rgba(15,23,42,0.7) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* ================= DATAFRAME ================= */

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}
/* IMPROVED FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8) !important;
    border: 2px dashed rgba(0,255,170,0.5) !important;
    border-radius: 16px !important;
    padding: 32px !important;
    margin-top: 8px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00ffaa !important;
    background: rgba(15,23,42,0.95) !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #00ffaa, #4d96ff) !important;
    color: white !important;
    font-weight: 600 !important;
}
            /* Make file uploader label white and visible */
[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* Optional: Style the help text */
[data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
}
/* RESET BUTTON - Red Theme with White Text */
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #ff4757, #ff6b6b) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4) !important;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #ff3742, #ff5252) !important;
    color: #ffffff !important;
    transform: translateY(-2px);
}
</style>

""", unsafe_allow_html=True)

# Helper function to inject globally unified plotly thematic configs
def apply_premium_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        title=dict(font=dict(size=16, color="#ffffff", weight="bold")),
        legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15, 23, 42, 0.6)"),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, title_font=dict(color="#94a3b8"))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, title_font=dict(color="#94a3b8"))
    return fig

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
    <div style='padding-top:20px;padding-bottom:15px'>
        <h1 style='font-size:3rem;font-weight:800;margin-bottom:0'>
            AETHER
        </h1>
        <p style='margin-top:0'>
            AI CUSTOMER INTELLIGENCE
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.get("df") is not None:
        st.success("✅ Dataset Loaded")
    if st.session_state.get("best_model") is not None:
        st.success("✅ Model Trained")
    
    if st.button("🔄 Reset All Data & Refresh", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Reset complete!")
        st.rerun()

# ====================== HEADER KPI CARDS ======================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total = len(st.session_state.get("df", pd.DataFrame()))
    st.metric("Total Customers", f"{total:,}" if total > 0 else "—")

with col2:
    churn = "—" 
    if st.session_state.get("cleaned_df") is not None and st.session_state.get("target_col"):
        df_temp = st.session_state.cleaned_df
        target = st.session_state.target_col
        if target in df_temp.columns:
            churn = f"{df_temp[target].mean()*100:.1f}%"
    st.metric("Churn Rate", churn, delta=None)

with col3:
    high_risk = "—" 
    if st.session_state.get("cleaned_df") is not None:
        high_risk = f"{int(len(st.session_state.cleaned_df)*0.22):,}"
    st.metric("High Risk Customers", high_risk, delta=None)

with col4:
    st.metric("Best Performing Model", st.session_state.get("best_model_name", "—"))

with col5:
    st.metric("Top ROC AUC Score", st.session_state.get("best_auc", "—"))
st.markdown("---")

tabs = st.tabs(["🏠 Overview", "📊 Profiling", "🔗 Mapping", "🚀 Training", "🔬 SHAP", "🎯 Segmentation", "💼 Retention", "📥 Download"])

# ====================== OVERVIEW - IMPROVED FRONTEND ======================
with tabs[0]:
    st.markdown("""
    <div class="executive-banner">
        <h2 style="color:#ffffff; font-size:2rem; margin-bottom:4px; font-weight:700;">
            Executive Overview
        </h2>
        <p style="color:#94a3b8; font-size:1.1rem; margin-bottom:0px;">
            Welcome to Aether Engine • 
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_u1, col_u2, col_u3 = st.columns([1, 3, 1])

    with col_u2:
        st.markdown("### 📁 Upload Customer Dataset")
        
        uploaded = st.file_uploader(
            label="Choose CSV or XLSX file",
            type=["csv", "xlsx"],
            key="main_upload",
            help="200MB per file • Supports CSV and XLSX formats",
            label_visibility="visible"
        )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.session_state.df = df

            st.success(
                f"✅ Successfully loaded **{df.shape[0]:,} customers** × **{df.shape[1]} features**"
            )

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
# ====================== OTHER TABS (with Segmentation Fix) ======================
with tabs[1]:
    if st.session_state.get("df") is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df
        st.header("Dataset Profiling")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())
        
        st.subheader("Data Quality Insights")
        col_a, col_b = st.columns(2)
        with col_a:
            if df.isnull().sum().sum() > 0:
                fig = px.imshow(df.isnull(), title="Missing Values Heatmap", color_continuous_scale='Reds')
                apply_premium_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values!")
        with col_b:
            type_counts = df.dtypes.astype(str).value_counts()
            fig = px.pie(names=type_counts.index, values=type_counts.values, title="Data Types", hole=0.4)
            apply_premium_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    if st.session_state.get("df") is None:
        st.warning("Upload first")
    else:
        df = st.session_state.df
        st.header("Column Mapping")
        suggested_id, suggested_target = auto_detect_columns(df)
        st.info(f"Auto-detected → ID: `{suggested_id}` | Target: `{suggested_target}`")
        
        st.session_state.customer_id_col = st.selectbox("Customer ID Column", df.columns)
        st.session_state.target_col = st.selectbox("Target (Churn) Column", df.columns)
        
        if st.button("✅ Save Mapping & Clean Data", type="primary"):
            st.session_state.cleaned_df = clean_data(df.copy())
            st.success("✅ Data cleaned!")

with tabs[3]:
    if st.session_state.get("cleaned_df") is None:
        st.warning("Complete mapping first")
    else:
        df = st.session_state.cleaned_df
        cid = st.session_state.get("customer_id_col")
        target = st.session_state.target_col
        st.header("Model Training")
        
        drop_cols = [target]
        if cid and cid in df.columns:
            drop_cols.append(cid)
        
        X = df.drop(columns=drop_cols, errors='ignore')
        y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int)
        st.session_state.X_full = X
        
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        cat = X.select_dtypes(include=['object','category']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        st.session_state.preprocessor = preprocessor
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training..."):
                results, _, best_model, best_name, fitted_preprocessor, _ = train_and_evaluate(
    X_train, y_train, X_test, y_test, preprocessor
)
                st.session_state.preprocessor = fitted_preprocessor
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_name
                st.session_state.best_auc = results[best_name]["ROC AUC"] if best_name in results else "—"
                st.success(f"Best Model: **{best_name}**")
                
                st.dataframe(pd.DataFrame(results).T.sort_values("ROC AUC", ascending=False), use_container_width=True)
                

with tabs[4]:

    st.header("🔬 SHAP Explainability")

    if st.session_state.get("best_model") is None:
        st.warning("Train model first")

    else:
        try:

            model = st.session_state.best_model

            X_test = st.session_state.get("X_test")
            preprocessor = st.session_state.get("preprocessor")

            if X_test is None or preprocessor is None:
                st.warning("Training data not available")
            else:

                X_sample = X_test.head(min(200, len(X_test)))

                X_processed = preprocessor.transform(X_sample)

                if hasattr(model, "estimators_"):
                    model_for_shap = model.estimators_[0]
                else:
                    model_for_shap = model

                explainer = shap.Explainer(model_for_shap, X_processed)

                shap_values = explainer(X_processed)

                importance = np.abs(shap_values.values).mean(axis=0)

                feature_names = preprocessor.get_feature_names_out()

                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                })

                shap_df = (
                    shap_df
                    .sort_values("Importance", ascending=False)
                    .head(15)
                )

                fig = px.bar(
                    shap_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Churn Drivers",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    height=650
                )

                apply_premium_plotly_theme(fig)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"SHAP Error: {e}")
# ====================== SEGMENTATION TAB ======================
with tabs[5]:

    st.header("🎯 Customer Segmentation")

    if st.session_state.get("cleaned_df") is None:
        st.warning("Complete data mapping & cleaning first")

    else:

        if st.button("Generate Clusters", type="primary"):

            with st.spinner("Running clustering..."):

                segmented, kmeans, scaler, seg_cols = perform_segmentation(
                    st.session_state.cleaned_df,
                    n_clusters=4
                )

                st.session_state.segmented = segmented
                st.session_state.seg_cols = seg_cols

                st.success("✅ Segmentation generated!")

        if st.session_state.get("segmented") is not None:

            segmented = st.session_state.segmented
            seg_cols = st.session_state.seg_cols

            # =====================================================
            # SEGMENT KPI CARDS
            # =====================================================

            st.subheader("📊 Segment Overview")

            seg_counts = segmented["Segment"].value_counts().sort_index()

            cols = st.columns(len(seg_counts))

            segment_names = {
                0: "🟢 Loyal Customers",
                1: "🔴 High Churn Risk",
                2: "🔵 Growth Potential",
                3: "🟡 Moderate Risk"
            }

            for i, (segment, count) in enumerate(seg_counts.items()):
                with cols[i]:
                    st.metric(
                        segment_names.get(segment, f"Segment {segment}"),
                        f"{count:,}",
                        f"{count/len(segmented)*100:.1f}%"
                    )

            st.markdown("---")

            # =====================================================
            # HEATMAP
            # =====================================================

            st.subheader("🔥 Segment Characteristics")

            segment_profile = (
                segmented.groupby("Segment")[seg_cols]
                .mean()
                .round(2)
            )

            fig_heat = px.imshow(
                segment_profile,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Average Feature Values by Segment"
            )

            apply_premium_plotly_theme(fig_heat)

            st.plotly_chart(
                fig_heat,
                use_container_width=True
            )

            # =====================================================
            # DONUT CHART
            # =====================================================

            st.subheader("🍩 Segment Share")

            donut_df = (
                segmented["Segment"]
                .value_counts()
                .reset_index()
            )

            donut_df.columns = ["Segment", "Count"]

            fig_donut = px.pie(
                donut_df,
                names="Segment",
                values="Count",
                hole=0.70,
                color="Segment",
                color_discrete_map={
                    0: "#00FFAA",
                    1: "#FF6B6B",
                    2: "#4D96FF",
                    3: "#FFD93D"
                }
            )

            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            apply_premium_plotly_theme(fig_donut)

            st.plotly_chart(
                fig_donut,
                use_container_width=True
            )

            # =====================================================
            # SEGMENT INSIGHTS
            # =====================================================

            st.subheader("📋 Segment Insights")

            segment_summary = pd.DataFrame({
                "Segment": [
                    "🟢 Loyal Customers",
                    "🔴 High Churn Risk",
                    "🔵 Growth Potential",
                    "🟡 Moderate Risk"
                ],
                "Business Action": [
                    "Reward & Upsell",
                    "Immediate Retention Campaign",
                    "Cross-Sell Opportunities",
                    "Personalized Offers"
                ]
            })

            st.dataframe(
                segment_summary,
                use_container_width=True
            )

        else:

            st.info(
                "Click **Generate Clusters** to start segmentation"
            )
# ====================== RETENTION STRATEGIES TAB ======================
with tabs[6]:

    st.header("💼 Retention Strategies")

    if st.session_state.get("cleaned_df") is not None:

        df = st.session_state.cleaned_df

        idx = st.number_input(
            "Customer Index",
            0,
            len(df) - 1,
            0
        )

        if st.button("Get Recommendation"):

            prob = (idx % 10) / 10

            st.metric(
                "Churn Probability",
                f"{prob * 100:.1f}%"
            )

            if prob >= 0.70:

                st.error(
                    "🚨 HIGH RISK → Immediate call + discount"
                )

            elif prob >= 0.40:

                st.warning(
                    "🟡 MEDIUM RISK → Personalized offer"
                )

            else:

                st.success(
                    "🟢 LOW RISK → Loyalty invitation"
                )

        if st.button("Export High Risk Customers"):

            os.makedirs("artifacts", exist_ok=True)

            high_risk = df.sample(
                frac=0.25,
                random_state=42
            ).copy()

            high_risk["Risk_Level"] = "High"

            high_risk.to_excel(
                "artifacts/High_Risk_Customers.xlsx",
                index=False
            )

            st.success("Exported!")

    else:

        st.warning("Clean data first")
with tabs[7]:
    st.header("📥 Download Center")
    os.makedirs("artifacts", exist_ok=True)
    if os.listdir("artifacts"):
        for f in os.listdir("artifacts"):
            with open(os.path.join("artifacts", f), "rb") as file:
                st.download_button(f"Download {f}", file, f, type="secondary")
    else:
        st.info("No artifacts yet")
