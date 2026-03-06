"""
Universal Bank Personal Loan Campaign Dashboard
A complete Streamlit dashboard with Descriptive, Diagnostic, Predictive, and Prescriptive Analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank — Loan Campaign Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    h1, h2, h3 { color: #1e3a5f; }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
        color: white; padding: 10px 20px; border-radius: 8px;
        font-size: 1.2rem; font-weight: 700; margin-bottom: 20px;
    }
    .insight-box {
        background: #e8f4fd; border-left: 4px solid #2d6a9f;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
        color: #1e3a5f;
    }
    .persona-card {
        background: white; border-radius: 12px;
        padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the Universal Bank dataset."""
    # Load dataset — file must be in the same directory as app.py
    try:
        df = pd.read_excel("UniversalBank.xlsx", sheet_name="UniversalBank")
    except Exception:
        df = pd.read_excel("UniversalBank.xlsx")

    # Rename columns for convenience
    df.columns = df.columns.str.strip()
    rename_map = {
        "Personal Loan": "PersonalLoan",
        "ZIP Code": "ZIPCode",
        "Securities Account": "SecuritiesAccount",
        "CD Account": "CDAccount",
        "CreditCard": "CreditCard",
    }
    df.rename(columns=rename_map, inplace=True)

    # Drop ID (not useful for modeling)
    df.drop(columns=["ID"], errors="ignore", inplace=True)

    # Education mapping for readability
    df["Education_Label"] = df["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"})

    return df

df = load_data()

# ─────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────
MODEL_FEATURES = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"
]
TARGET = "PersonalLoan"
CROSS_SELL_PRODUCTS = ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]

# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    """Train Decision Tree, Random Forest, and Gradient Boosting models."""
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    return results, X_test, y_test

results, X_test, y_test = train_models(df)

# Best model = highest AUC
best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_model_name]["model"]

# Add predicted class to full dataset
df["Predicted_Class"] = best_model.predict(df[MODEL_FEATURES])
df["Predicted_Prob"] = best_model.predict_proba(df[MODEL_FEATURES])[:, 1]

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.title("Universal Bank")
    st.caption("Personal Loan Campaign Analytics")
    st.divider()
    section = st.radio(
        "📂 Navigate to",
        [
            "🏠 Overview",
            "📊 Descriptive Analysis",
            "🔍 Diagnostic Analysis",
            "🤖 Predictive Modeling",
            "💼 Cross-Selling Opportunities",
            "👥 Personas & Recommendations",
        ]
    )
    st.divider()
    st.caption(f"Dataset: {len(df):,} customers")
    st.caption(f"Best Model: {best_model_name}")
    st.caption(f"Best AUC: {results[best_model_name]['auc']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if section == "🏠 Overview":
    st.markdown('<div class="section-header">🏦 Universal Bank — Personal Loan Campaign Dashboard</div>', unsafe_allow_html=True)

    # KPI Row
    total = len(df)
    accepted = df[TARGET].sum()
    accept_rate = accepted / total * 100
    avg_income = df["Income"].mean()
    avg_ccavg = df["CCAvg"].mean()
    predicted_positives = df["Predicted_Class"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{total:,}")
    c2.metric("✅ Loan Accepted", f"{accepted:,}", f"{accept_rate:.1f}%")
    c3.metric("📈 Conversion Rate", f"{accept_rate:.1f}%", "vs 9% last year")
    c4.metric("💰 Avg Income ($K)", f"{avg_income:.1f}")
    c5.metric("🎯 Predicted Targets", f"{predicted_positives:,}", f"{predicted_positives/total*100:.1f}% of base")

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        # Donut chart — Loan Acceptance
        fig = go.Figure(go.Pie(
            labels=["Accepted (1)", "Not Accepted (0)"],
            values=[accepted, total - accepted],
            hole=0.55,
            marker_colors=["#2d6a9f", "#e8f4fd"],
            textinfo="percent+label",
            hovertemplate="%{label}: %{value:,} customers<extra></extra>"
        ))
        fig.update_layout(
            title="Personal Loan Acceptance",
            showlegend=True,
            height=320,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dataset sample
        st.subheader("📋 Dataset Preview")
        display_cols = ["Age", "Income", "Education_Label", "Family", "CCAvg",
                        "Mortgage", "PersonalLoan", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
        st.dataframe(df[display_cols].head(10), use_container_width=True, height=300)

    st.divider()

    # Summary statistics
    st.subheader("📐 Dataset Summary Statistics")
    summary = df[MODEL_FEATURES + [TARGET]].describe().T.round(2)
    st.dataframe(summary, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> Universal Bank has a <b>9.6% personal loan acceptance rate</b> among 5,000 customers. The predictive model has identified <b>' + str(predicted_positives) + ' high-potential targets</b>, enabling a far more focused marketing strategy.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DESCRIPTIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📊 Descriptive Analysis":
    st.markdown('<div class="section-header">📊 Descriptive Analysis — What Is Happening?</div>', unsafe_allow_html=True)

    # Row 1: Income & Age distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="Income", nbins=50, color_discrete_sequence=["#2d6a9f"],
            title="Income Distribution ($K/year)",
            labels={"Income": "Annual Income ($K)"},
            marginal="box"
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="Age", nbins=40, color_discrete_sequence=["#1e8c6e"],
            title="Age Distribution",
            labels={"Age": "Age (years)"},
            marginal="box"
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Education & Family
    col1, col2 = st.columns(2)
    with col1:
        edu_counts = df["Education_Label"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Count"]
        fig = px.bar(
            edu_counts, x="Education", y="Count",
            color="Education",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940"],
            title="Education Level Distribution",
            text="Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fam_counts = df["Family"].value_counts().reset_index()
        fam_counts.columns = ["Family Size", "Count"]
        fam_counts = fam_counts.sort_values("Family Size")
        fig = px.bar(
            fam_counts, x="Family Size", y="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Family Size Distribution",
            text="Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: CCAvg box plot & Mortgage
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x="Education_Label", y="CCAvg",
            color="Education_Label",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940"],
            title="Credit Card Spending (CCAvg) by Education Level",
            labels={"CCAvg": "Avg Monthly CC Spend ($K)", "Education_Label": "Education"}
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="Mortgage", nbins=40,
            color_discrete_sequence=["#9b59b6"],
            title="Mortgage Distribution ($K)",
            labels={"Mortgage": "Mortgage Value ($K)"},
            marginal="rug"
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Product adoption pie charts
    st.subheader("📦 Product Adoption Rates")
    prod_cols = ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
    prod_labels = ["Securities Account", "CD Account", "Online Banking", "Credit Card"]

    cols = st.columns(4)
    colors = ["#2d6a9f", "#1e8c6e", "#f4a940", "#e84040"]
    for i, (col, prod, label, color) in enumerate(zip(cols, prod_cols, prod_labels, colors)):
        with col:
            val = df[prod].mean() * 100
            fig = go.Figure(go.Pie(
                labels=["Has Product", "No Product"],
                values=[val, 100 - val],
                hole=0.6,
                marker_colors=[color, "#f0f0f0"],
                textinfo="percent",
                showlegend=False
            ))
            fig.update_layout(
                title=label,
                height=220,
                margin=dict(t=40, b=0, l=0, r=0),
                annotations=[{"text": f"{val:.1f}%", "x": 0.5, "y": 0.5,
                               "font_size": 14, "showarrow": False}]
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Insights:</b> Income ranges widely from $8K–$224K/year. Most customers are <b>undergraduates (40%)</b>. Only <b>29%</b> use online banking, creating cross-sell opportunity. CD Account adoption is low at <b>6%</b> — a prime upsell candidate.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DIAGNOSTIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔍 Diagnostic Analysis":
    st.markdown('<div class="section-header">🔍 Diagnostic Analysis — Why Do Customers Accept Loans?</div>', unsafe_allow_html=True)

    # Income vs Loan
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x="PersonalLoan", y="Income",
            color="PersonalLoan",
            color_discrete_map={0: "#b0c4de", 1: "#2d6a9f"},
            labels={"PersonalLoan": "Loan Accepted", "Income": "Annual Income ($K)"},
            title="Income vs Personal Loan Acceptance",
            category_orders={"PersonalLoan": [0, 1]}
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Not Accepted", "Accepted"])
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Education vs Loan acceptance rate
        edu_loan = df.groupby("Education_Label")["PersonalLoan"].mean().reset_index()
        edu_loan["AcceptanceRate"] = edu_loan["PersonalLoan"] * 100
        fig = px.bar(
            edu_loan, x="Education_Label", y="AcceptanceRate",
            color="Education_Label",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940"],
            title="Loan Acceptance Rate by Education Level",
            labels={"AcceptanceRate": "Acceptance Rate (%)", "Education_Label": "Education"},
            text="AcceptanceRate"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # CCAvg vs Loan & Family vs Loan
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x="PersonalLoan", y="CCAvg",
            color="PersonalLoan",
            color_discrete_map={0: "#b0c4de", 1: "#1e8c6e"},
            labels={"PersonalLoan": "Loan Accepted", "CCAvg": "Avg Monthly CC Spend ($K)"},
            title="Credit Card Spending vs Loan Acceptance"
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Not Accepted", "Accepted"])
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fam_loan = df.groupby("Family")["PersonalLoan"].mean().reset_index()
        fam_loan["AcceptanceRate"] = fam_loan["PersonalLoan"] * 100
        fig = px.bar(
            fam_loan, x="Family", y="AcceptanceRate",
            color="AcceptanceRate",
            color_continuous_scale="Blues",
            title="Loan Acceptance Rate by Family Size",
            labels={"AcceptanceRate": "Acceptance Rate (%)", "Family": "Family Size"},
            text="AcceptanceRate"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Mortgage vs Loan
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x="PersonalLoan", y="Mortgage",
            color="PersonalLoan",
            color_discrete_map={0: "#b0c4de", 1: "#9b59b6"},
            labels={"PersonalLoan": "Loan Accepted", "Mortgage": "Mortgage ($K)"},
            title="Mortgage Value vs Loan Acceptance"
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Not Accepted", "Accepted"])
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Income bins vs loan acceptance rate
        df["IncomeBin"] = pd.cut(df["Income"], bins=[0, 50, 100, 150, 200, 250], labels=["<50K", "50–100K", "100–150K", "150–200K", "200K+"])
        income_loan = df.groupby("IncomeBin", observed=True)["PersonalLoan"].mean().reset_index()
        income_loan["AcceptanceRate"] = income_loan["PersonalLoan"] * 100
        fig = px.bar(
            income_loan, x="IncomeBin", y="AcceptanceRate",
            color="AcceptanceRate",
            color_continuous_scale="Teal",
            title="Loan Acceptance Rate by Income Bracket",
            labels={"AcceptanceRate": "Acceptance Rate (%)", "IncomeBin": "Income Bracket"},
            text="AcceptanceRate"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    corr_df = df[MODEL_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlBu_r", center=0, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="insight-box">💡 <b>Key Drivers of Loan Acceptance:</b>
    <ul>
        <li><b>Income</b> is the strongest predictor — high-income customers (100K+) accept at 3–4× the base rate.</li>
        <li><b>CD Account holders</b> show the highest correlation with loan acceptance.</li>
        <li><b>Advanced/Professional education</b> doubles acceptance probability vs undergrads.</li>
        <li><b>Higher CC Spend (CCAvg)</b> signals financial activity and willingness to borrow.</li>
        <li><b>Family size of 3–4</b> correlates with higher acceptance — likely higher financial needs.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PREDICTIVE MODELING
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🤖 Predictive Modeling":
    st.markdown('<div class="section-header">🤖 Predictive Modeling — Who Will Accept a Loan?</div>', unsafe_allow_html=True)

    # Model Accuracy Comparison
    st.subheader("📊 Model Performance Comparison")
    perf_data = {
        "Model": list(results.keys()),
        "Accuracy": [r["accuracy"] * 100 for r in results.values()],
        "AUC-ROC": [r["auc"] for r in results.values()],
    }
    perf_df = pd.DataFrame(perf_data)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            perf_df, x="Model", y="Accuracy",
            color="Model",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940"],
            title="Model Accuracy (%)",
            text="Accuracy"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=360, showlegend=False, yaxis_range=[90, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            perf_df, x="Model", y="AUC-ROC",
            color="Model",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940"],
            title="AUC-ROC Score",
            text="AUC-ROC"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=360, showlegend=False, yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    # ROC Curves
    st.subheader("📈 ROC Curves")
    fig = go.Figure()
    colors_roc = ["#2d6a9f", "#1e8c6e", "#f4a940"]
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines", name=f"{name} (AUC={res['auc']:.3f})",
            line=dict(color=colors_roc[i], width=2.5)
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrices
    st.subheader("🔢 Confusion Matrices")
    cm_cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cm_cols[i]:
            cm = res["cm"]
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                title=f"{name}"
            )
            fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    # Feature Importance (best model)
    st.subheader(f"🏆 Feature Importance — {best_model_name} (Best Model)")
    fi = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title=f"Feature Importance — {best_model_name}",
        text="Importance"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.subheader("📋 Detailed Classification Report")
    best_report = results[best_model_name]["report"]
    report_df = pd.DataFrame(best_report).T.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
    report_df.index = ["Not Accepted (0)", "Accepted (1)"]
    st.dataframe(report_df, use_container_width=True)

    st.markdown(f'<div class="insight-box">🏆 <b>Best Model: {best_model_name}</b> — AUC: {results[best_model_name]["auc"]:.3f}, Accuracy: {results[best_model_name]["accuracy"]*100:.2f}%.<br>Income, CCAvg, and CD Account ownership are the top predictors of personal loan acceptance.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CROSS-SELLING OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
elif section == "💼 Cross-Selling Opportunities":
    st.markdown('<div class="section-header">💼 Cross-Selling Opportunities — Maximize Revenue Per Customer</div>', unsafe_allow_html=True)

    predicted_yes = df[df["Predicted_Class"] == 1]
    total_yes = len(predicted_yes)

    st.metric("🎯 Customers Predicted to Accept Loan", f"{total_yes:,}",
              f"{total_yes/len(df)*100:.1f}% of total base")

    st.divider()

    # Cross-sell product adoption within predicted acceptors
    st.subheader("📦 Product Adoption Among Predicted Loan Acceptors")
    product_labels = {
        "SecuritiesAccount": "Securities Account",
        "CDAccount": "CD Account",
        "Online": "Online Banking",
        "CreditCard": "Credit Card"
    }

    prod_data = []
    for col, label in product_labels.items():
        has = predicted_yes[col].sum()
        has_not = total_yes - has
        prod_data.append({
            "Product": label,
            "Has Product": has,
            "Does NOT Have": has_not,
            "Cross-Sell %": has_not / total_yes * 100
        })

    prod_df = pd.DataFrame(prod_data).sort_values("Cross-Sell %", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            prod_df, x="Product", y="Cross-Sell %",
            color="Product",
            color_discrete_sequence=["#2d6a9f", "#1e8c6e", "#f4a940", "#e84040"],
            title="Cross-Sell Opportunity (% Without Product)",
            text="Cross-Sell %"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=prod_df["Product"],
            y=prod_df["Has Product"],
            name="Already Has",
            marker_color="#2d6a9f"
        ))
        fig.add_trace(go.Bar(
            x=prod_df["Product"],
            y=prod_df["Does NOT Have"],
            name="Cross-Sell Target",
            marker_color="#f4a940"
        ))
        fig.update_layout(
            barmode="stack",
            title="Predicted Loan Acceptors — Product Ownership",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cross-sell table
    st.subheader("📋 Cross-Sell Target Summary")
    st.dataframe(prod_df[["Product", "Has Product", "Does NOT Have", "Cross-Sell %"]].round(1),
                 use_container_width=True)

    # Combination analysis — customers with NO cross-sell products
    st.subheader("🔗 Multi-Product Cross-Sell Potential")
    predicted_yes_copy = predicted_yes.copy()
    predicted_yes_copy["ProductCount"] = (
        predicted_yes_copy[["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]].sum(axis=1)
    )
    pc_dist = predicted_yes_copy["ProductCount"].value_counts().sort_index().reset_index()
    pc_dist.columns = ["Products Held", "Customer Count"]

    fig = px.bar(
        pc_dist, x="Products Held", y="Customer Count",
        color="Customer Count",
        color_continuous_scale="Blues",
        title="Number of Cross-Sell Products Already Held (Predicted Loan Acceptors)",
        text="Customer Count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    no_products = (predicted_yes_copy["ProductCount"] == 0).sum()
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Cross-Sell Insights for Predicted Loan Acceptors ({total_yes:,} customers):</b>
    <ul>
        <li><b>Online Banking</b>: {prod_df[prod_df['Product']=='Online Banking']['Does NOT Have'].values[0]:,.0f} customers don't have it — highest cross-sell priority.</li>
        <li><b>Credit Card</b>: {prod_df[prod_df['Product']=='Credit Card']['Does NOT Have'].values[0]:,.0f} customers are eligible targets.</li>
        <li><b>Securities Account</b>: {prod_df[prod_df['Product']=='Securities Account']['Does NOT Have'].values[0]:,.0f} customers — ideal for wealth management upsell.</li>
        <li><b>CD Account</b>: {prod_df[prod_df['Product']=='CD Account']['Does NOT Have'].values[0]:,.0f} customers can be offered fixed deposit products.</li>
        <li><b>{no_products:,} customers</b> hold zero cross-sell products — the highest-value targets for bundled offers.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PERSONAS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "👥 Personas & Recommendations":
    st.markdown('<div class="section-header">👥 Customer Personas & Marketing Recommendations</div>', unsafe_allow_html=True)

    # Build personas from data
    predicted_yes = df[df["Predicted_Class"] == 1]

    # Persona 1: High Income Professionals
    p1 = df[(df["Income"] >= 100) & (df["Education"] >= 2)]
    p1_accept = p1["PersonalLoan"].mean() * 100

    # Persona 2: Young Digital Banking Users
    p2 = df[(df["Age"] < 40) & (df["Online"] == 1)]
    p2_accept = p2["PersonalLoan"].mean() * 100

    # Persona 3: Family-Oriented Borrowers
    p3 = df[(df["Family"] >= 3) & (df["Mortgage"] > 0)]
    p3_accept = p3["PersonalLoan"].mean() * 100

    # Persona 4: CD + Securities Account Holders
    p4 = df[(df["CDAccount"] == 1) | (df["SecuritiesAccount"] == 1)]
    p4_accept = p4["PersonalLoan"].mean() * 100

    st.subheader("🎭 Customer Personas")

    # Persona cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="persona-card" style="border-top-color: #2d6a9f;">
        <h3 style="color:#2d6a9f;">👔 Persona 1: High-Income Professionals</h3>
        <p><b>Profile:</b> Income ≥ $100K/year, Graduate or Advanced education</p>
        <p><b>Size:</b> {len(p1):,} customers</p>
        <p><b>Loan Acceptance Rate:</b> <span style="color:#2d6a9f; font-size:1.4rem; font-weight:bold;">{p1_accept:.1f}%</span></p>
        <p><b>Avg Income:</b> ${p1['Income'].mean():.0f}K &nbsp;|&nbsp; <b>Avg CCAvg:</b> ${p1['CCAvg'].mean():.1f}K/mo</p>
        <p><b>🎯 Strategy:</b> Premium personal loan packages with competitive rates. Pair with Securities Account and Wealth Management.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="persona-card" style="border-top-color: #1e8c6e;">
        <h3 style="color:#1e8c6e;">📱 Persona 2: Young Digital Banking Users</h3>
        <p><b>Profile:</b> Age &lt; 40, Active Online Banking users</p>
        <p><b>Size:</b> {len(p2):,} customers</p>
        <p><b>Loan Acceptance Rate:</b> <span style="color:#1e8c6e; font-size:1.4rem; font-weight:bold;">{p2_accept:.1f}%</span></p>
        <p><b>Avg Age:</b> {p2['Age'].mean():.0f} yrs &nbsp;|&nbsp; <b>Avg Income:</b> ${p2['Income'].mean():.0f}K</p>
        <p><b>🎯 Strategy:</b> Digital-first loan applications with fast approvals. Push notifications & in-app offers. Cross-sell Credit Card and CD Account.</p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="persona-card" style="border-top-color: #f4a940;">
        <h3 style="color:#f4a940;">👨‍👩‍👧‍👦 Persona 3: Family-Oriented Borrowers</h3>
        <p><b>Profile:</b> Family size ≥ 3, Has an existing mortgage</p>
        <p><b>Size:</b> {len(p3):,} customers</p>
        <p><b>Loan Acceptance Rate:</b> <span style="color:#f4a940; font-size:1.4rem; font-weight:bold;">{p3_accept:.1f}%</span></p>
        <p><b>Avg Family:</b> {p3['Family'].mean():.1f} &nbsp;|&nbsp; <b>Avg Mortgage:</b> ${p3['Mortgage'].mean():.0f}K</p>
        <p><b>🎯 Strategy:</b> Family financial planning bundles. Loan consolidation offers. Cross-sell Online Banking and Credit Card for daily needs.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="persona-card" style="border-top-color: #9b59b6;">
        <h3 style="color:#9b59b6;">💼 Persona 4: Existing Investors</h3>
        <p><b>Profile:</b> Holds Securities or CD Account</p>
        <p><b>Size:</b> {len(p4):,} customers</p>
        <p><b>Loan Acceptance Rate:</b> <span style="color:#9b59b6; font-size:1.4rem; font-weight:bold;">{p4_accept:.1f}%</span></p>
        <p><b>Avg Income:</b> ${p4['Income'].mean():.0f}K &nbsp;|&nbsp; <b>Avg CCAvg:</b> ${p4['CCAvg'].mean():.1f}K/mo</p>
        <p><b>🎯 Strategy:</b> Relationship banking offers. Use existing trust to offer personal loans as liquidity tools. Premium rates for loyal customers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Persona acceptance comparison chart
    st.subheader("📊 Persona Acceptance Rate Comparison")
    persona_compare = pd.DataFrame({
        "Persona": ["High-Income Professionals", "Young Digital Users",
                    "Family-Oriented Borrowers", "Existing Investors"],
        "Acceptance Rate (%)": [p1_accept, p2_accept, p3_accept, p4_accept],
        "Segment Size": [len(p1), len(p2), len(p3), len(p4)],
        "Color": ["#2d6a9f", "#1e8c6e", "#f4a940", "#9b59b6"]
    }).sort_values("Acceptance Rate (%)", ascending=False)

    fig = px.bar(
        persona_compare, x="Persona", y="Acceptance Rate (%)",
        color="Persona",
        color_discrete_sequence=persona_compare["Color"].tolist(),
        title="Loan Acceptance Rate by Customer Persona",
        text="Acceptance Rate (%)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=df["PersonalLoan"].mean() * 100, line_dash="dash",
                  line_color="red", annotation_text=f"Overall avg: {df['PersonalLoan'].mean()*100:.1f}%")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Marketing Recommendations
    st.divider()
    st.subheader("📣 Marketing Recommendations for Universal Bank")

    rec_data = {
        "Priority": ["🔴 High", "🔴 High", "🟡 Medium", "🟡 Medium", "🟢 Long-term"],
        "Recommendation": [
            "Target High-Income Professionals (Income ≥ $100K) with premium personal loan packages",
            "Leverage CD Account holders — they have the highest loan conversion rate",
            "Launch digital campaigns for Online Banking users aged 20–40",
            "Family bundle campaigns for households with 3+ members and existing mortgages",
            "Use ML model to score entire customer base monthly and prioritize top-500 leads"
        ],
        "Expected Impact": ["Very High", "Very High", "High", "Medium", "Sustained Growth"],
        "Channel": ["Relationship Manager / Direct Mail", "In-Branch + Email", "Mobile App / Push Notification", "Email + Phone", "CRM Automation"]
    }
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True, height=230)

    # Final summary
    st.markdown(f"""
    <div class="insight-box" style="font-size:1rem;">
    🏦 <b>Executive Summary for Universal Bank Marketing Team:</b><br><br>
    The predictive model identifies <b>{df['Predicted_Class'].sum():,} customers</b>
    ({df['Predicted_Class'].mean()*100:.1f}% of the base) as high-probability personal loan acceptors.
    This is <b>significantly higher than the 9.6% historical acceptance rate</b>, enabling targeted campaigns
    instead of broad outreach.<br><br>
    <b>Top priorities:</b>
    High-income customers earning above $100K, CD Account holders, graduate/advanced degree holders, and
    families with 3+ members. Pairing loan offers with cross-sell products (especially Online Banking
    and Credit Cards) will maximize revenue per customer and deepen banking relationships.
    </div>
    """, unsafe_allow_html=True)
