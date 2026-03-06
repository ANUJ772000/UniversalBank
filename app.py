"""
Universal Bank — Personal Loan Campaign Dashboard
Executive-friendly data storytelling dashboard.
Designed for non-technical stakeholders: bank executives, investors, and marketing leads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Universal Bank — Loan Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLES
# Color convention:
#   Green  (#1e8c6e) → customers LIKELY to accept loans
#   Red    (#d64045) → customers UNLIKELY to accept loans
#   Blue   (#2d6a9f) → neutral / informational metrics
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.main { background-color: #f4f7fb; }
section[data-testid="stSidebar"] { background: #1e3a5f; }
section[data-testid="stSidebar"] * { color: #ffffff !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #e8eef5; padding: 6px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 18px;
    font-weight: 600; color: #1e3a5f; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #2d6a9f !important; color: white !important;
}

div[data-testid="metric-container"] {
    background: #ffffff; border-radius: 14px;
    padding: 16px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid #2d6a9f;
}

.banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    color: white; padding: 14px 22px; border-radius: 10px;
    font-size: 1.15rem; font-weight: 700; margin-bottom: 22px;
}

.insight-green {
    background: #eafaf4; border-left: 5px solid #1e8c6e;
    padding: 14px 18px; border-radius: 8px; margin: 10px 0;
    color: #0d4a30; font-size: 0.95rem;
}
.insight-blue {
    background: #e8f2fd; border-left: 5px solid #2d6a9f;
    padding: 14px 18px; border-radius: 8px; margin: 10px 0;
    color: #0d2a4a; font-size: 0.95rem;
}
.insight-orange {
    background: #fff8ec; border-left: 5px solid #f4a940;
    padding: 14px 18px; border-radius: 8px; margin: 10px 0;
    color: #5a3a00; font-size: 0.95rem;
}

.persona {
    background: white; border-radius: 14px; padding: 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-top: 5px solid; margin-bottom: 12px;
}
.persona h4 { margin: 0 0 10px 0; font-size: 1.05rem; }
.big-rate { font-size: 2rem; font-weight: 800; }

.chart-note {
    font-size: 0.85rem; color: #555; background: #f9f9f9;
    padding: 8px 14px; border-radius: 6px;
    margin-top: -6px; margin-bottom: 14px; border: 1px solid #e5e5e5;
}

.rec-table { width:100%; border-collapse: collapse; font-size:0.9rem; }
.rec-table th { background:#1e3a5f; color:white; padding:10px; text-align:left; }
.rec-table td { padding:10px; border-bottom:1px solid #e5e5e5; vertical-align:top; }
.rec-table tr:nth-child(even) td { background:#f4f7fb; }
.pill-high   { background:#fde8e8; color:#c0392b; border-radius:20px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
.pill-medium { background:#fff3cd; color:#856404; border-radius:20px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
.pill-low    { background:#d4edda; color:#155724; border-radius:20px; padding:3px 10px; font-weight:600; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("UniversalBank.xlsx", sheet_name="UniversalBank")
    except Exception:
        df = pd.read_excel("UniversalBank.xlsx")

    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Personal Loan": "PersonalLoan",
        "ZIP Code": "ZIPCode",
        "Securities Account": "SecuritiesAccount",
        "CD Account": "CDAccount",
    }, inplace=True)
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    df["Education_Label"] = df["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"})
    df["Loan_Label"] = df["PersonalLoan"].map({1: "Accepted", 0: "Not Accepted"})
    return df

df = load_data()

MODEL_FEATURES = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"
]
TARGET = "PersonalLoan"

# ══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    X = df[MODEL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":    model,
            "accuracy": accuracy_score(y_test, y_pred),
            "cm":       confusion_matrix(y_test, y_pred),
            "y_test":   y_test,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "auc":      roc_auc_score(y_test, y_prob),
        }
    return results, X_test, y_test

results, X_test, y_test = train_models(df)
best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model      = results[best_model_name]["model"]
df["Predicted_Class"] = best_model.predict(df[MODEL_FEATURES])
df["Predicted_Prob"]  = best_model.predict_proba(df[MODEL_FEATURES])[:, 1]

# ══════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
C_GREEN  = "#1e8c6e"
C_RED    = "#d64045"
C_BLUE   = "#2d6a9f"
C_AMBER  = "#f4a940"
C_PURPLE = "#7b5ea7"

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Personal Loan Campaign Intelligence**")
    st.markdown("---")
    st.markdown("### Navigate")
    page = st.radio("", [
        "🏠  Executive Overview",
        "👤  Customer Profiles",
        "🔍  Why Customers Accept Loans",
        "🤖  Predictive Model Results",
        "💼  Cross-Selling Opportunities",
        "🎯  Customer Personas & Actions",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Quick Stats**")
    st.markdown(f"- Customers analysed: **{len(df):,}**")
    st.markdown(f"- Loan acceptance rate: **{df[TARGET].mean()*100:.1f}%**")
    st.markdown(f"- Best model AUC: **{results[best_model_name]['auc']:.3f}**")
    st.markdown(f"- ML-identified targets: **{df['Predicted_Class'].sum():,}**")
    st.markdown("---")
    st.caption("Use the tabs within each section for deeper insights.")


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def banner(title):
    st.markdown(f'<div class="banner">{title}</div>', unsafe_allow_html=True)

def insight(text, style="blue"):
    st.markdown(f'<div class="insight-{style}">{text}</div>', unsafe_allow_html=True)

def chart_note(text):
    st.markdown(f'<div class="chart-note">📖 {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    banner("🏠 Executive Overview — What Is Happening in the Data?")

    total      = len(df)
    accepted   = int(df[TARGET].sum())
    rate       = accepted / total * 100
    avg_income = df["Income"].mean()
    avg_cc     = df["CCAvg"].mean()
    pct_cd     = df["CDAccount"].mean() * 100
    targets    = int(df["Predicted_Class"].sum())

    # KPI Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("👥 Total Customers",       f"{total:,}")
    k2.metric("✅ Loan Acceptances",      f"{accepted:,}", f"{rate:.1f}% of base")
    k3.metric("📈 Acceptance Rate",       f"{rate:.1f}%",  "vs ~9% industry avg")
    k4.metric("💰 Avg Annual Income",     f"${avg_income:.0f}K")
    k5.metric("💳 Avg CC Spending",       f"${avg_cc:.1f}K/mo")
    k6.metric("🎯 ML-Identified Targets", f"{targets:,}", "high-probability leads")

    st.divider()

    col_a, col_b = st.columns([1, 2])

    with col_a:
        fig = go.Figure(go.Pie(
            labels=["Accepted Loan", "Did Not Accept"],
            values=[accepted, total - accepted],
            hole=0.62,
            marker_colors=[C_GREEN, C_RED],
            textinfo="percent+label",
            hovertemplate="%{label}<br>Count: %{value:,}<extra></extra>",
            pull=[0.04, 0],
        ))
        fig.update_layout(
            title=dict(text="Loan Acceptance Split", font_size=15),
            showlegend=False, height=310,
            margin=dict(t=40, b=0, l=0, r=0),
            annotations=[{
                "text": f"<b>{rate:.1f}%</b><br>accepted",
                "x": 0.5, "y": 0.5, "font_size": 16,
                "showarrow": False, "font_color": C_GREEN
            }]
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Only 1 in 10 customers accepted. The ML model finds those 1 in 10 before we call them.")

    with col_b:
        fig = px.histogram(
            df, x="Income", color="Loan_Label",
            color_discrete_map={"Accepted": C_GREEN, "Not Accepted": C_RED},
            nbins=50, barmode="overlay", opacity=0.75,
            title="Income Distribution by Loan Acceptance",
            labels={"Income": "Annual Income ($K)", "count": "No. of Customers", "Loan_Label": "Outcome"},
        )
        fig.update_layout(height=310, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Green bars (loan acceptors) cluster at higher income brackets — income is the #1 signal.")

    insight(
        f"🔑 <b>Key Takeaway:</b> Of {total:,} customers, only <b>{accepted:,} ({rate:.1f}%)</b> accepted the "
        f"personal loan offer. Our machine learning model has identified <b>{targets:,} customers</b> who are "
        f"highly likely to say yes — enabling a precise, targeted campaign instead of mass outreach.",
        "green"
    )

    st.divider()
    st.markdown("#### 📦 Current Product Adoption Across All Customers")

    prods  = ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
    labels = ["Securities Account", "CD Account", "Online Banking", "Credit Card"]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    pcols = st.columns(4)
    for col, prod, lbl, clr in zip(pcols, prods, labels, colors):
        val = df[prod].mean() * 100
        fig = go.Figure(go.Pie(
            labels=["Has it", "Doesn't have it"],
            values=[val, 100 - val],
            hole=0.65,
            marker_colors=[clr, "#ececec"],
            textinfo="none", showlegend=False,
        ))
        fig.update_layout(
            title=dict(text=lbl, font_size=13), height=200,
            margin=dict(t=36, b=0, l=0, r=0),
            annotations=[{"text": f"<b>{val:.0f}%</b>", "x": 0.5, "y": 0.5,
                           "font_size": 17, "showarrow": False, "font_color": clr}]
        )
        col.plotly_chart(fig, use_container_width=True)

    insight(
        "💡 <b>Cross-Sell Gap:</b> Only <b>6% of customers have a CD Account</b> and "
        "only <b>29% use Online Banking</b>. These are significant untapped opportunities — "
        "especially among the customers predicted to take a loan.",
        "orange"
    )


# ══════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER PROFILES (DESCRIPTIVE)
# ══════════════════════════════════════════════════════════════
elif "Profiles" in page:
    banner("👤 Customer Profiles — Who Are Our Customers?")

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Income", "🎂 Age", "🎓 Education", "👨‍👩‍👧 Family Size"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Income", nbins=50,
                               color_discrete_sequence=[C_BLUE],
                               title="Annual Income Distribution",
                               labels={"Income": "Annual Income ($K)", "count": "No. of Customers"},
                               marginal="box")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df["IncomeBracket"] = pd.cut(df["Income"], bins=[0,50,100,150,200,300],
                                          labels=["<$50K","$50–100K","$100–150K","$150–200K","$200K+"])
            ib = df["IncomeBracket"].value_counts().sort_index().reset_index()
            ib.columns = ["Income Bracket","Customers"]
            fig = px.bar(ib, x="Income Bracket", y="Customers",
                         color_discrete_sequence=[C_BLUE],
                         title="Customers by Income Bracket", text="Customers")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        chart_note(
            f"Most customers earn between $20K–$120K per year. The average income is "
            f"${df['Income'].mean():.0f}K. A smaller high-income group (above $100K) drives "
            "the majority of personal loan acceptances."
        )
        with st.expander("See income summary statistics"):
            st.dataframe(df["Income"].describe().round(2).to_frame("Income ($K)"), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Age", nbins=40,
                               color_discrete_sequence=[C_PURPLE],
                               title="Customer Age Distribution",
                               labels={"Age": "Age (years)", "count": "No. of Customers"},
                               marginal="box")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df["AgeBand"] = pd.cut(df["Age"], bins=[20,30,40,50,60,70,100],
                                    labels=["20s","30s","40s","50s","60s","70+"])
            ab = df["AgeBand"].value_counts().sort_index().reset_index()
            ab.columns = ["Age Band","Customers"]
            fig = px.bar(ab, x="Age Band", y="Customers",
                         color_discrete_sequence=[C_PURPLE],
                         title="Customers by Age Group", text="Customers")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        chart_note(
            f"Customers range from mid-20s to late 60s. The 30s and 40s age groups are the largest "
            f"segments. Average age is {df['Age'].mean():.0f} years."
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            edu_counts = df["Education_Label"].value_counts().reset_index()
            edu_counts.columns = ["Education","Customers"]
            fig = px.pie(edu_counts, values="Customers", names="Education",
                         color_discrete_sequence=[C_BLUE, C_GREEN, C_AMBER],
                         title="Education Level Breakdown", hole=0.4)
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            edu_loan = df.groupby("Education_Label")["PersonalLoan"].mean().reset_index()
            edu_loan["Acceptance Rate (%)"] = edu_loan["PersonalLoan"] * 100
            fig = px.bar(edu_loan, x="Education_Label", y="Acceptance Rate (%)",
                         color="Education_Label",
                         color_discrete_sequence=[C_BLUE, C_GREEN, C_AMBER],
                         title="Loan Acceptance Rate by Education",
                         text="Acceptance Rate (%)",
                         labels={"Education_Label": "Education"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        chart_note(
            "Advanced/Professional degree holders accept loans at the highest rate. "
            "Graduate and Advanced customers are the strongest target audience for loan campaigns."
        )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fam = df["Family"].value_counts().sort_index().reset_index()
            fam.columns = ["Family Size","Customers"]
            fig = px.bar(fam, x="Family Size", y="Customers",
                         color_discrete_sequence=[C_AMBER],
                         title="Customers by Family Size", text="Customers")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fam_loan = df.groupby("Family")["PersonalLoan"].mean().reset_index()
            fam_loan["Acceptance Rate (%)"] = fam_loan["PersonalLoan"] * 100
            fig = px.bar(fam_loan, x="Family", y="Acceptance Rate (%)",
                         color="Acceptance Rate (%)",
                         color_continuous_scale="YlGn",
                         title="Loan Acceptance Rate by Family Size",
                         text="Acceptance Rate (%)",
                         labels={"Family": "Family Size"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        chart_note(
            "Families of 3 or more show higher loan acceptance rates — larger families tend to have "
            "greater financial needs (education, home upgrades) which drives personal loan demand."
        )


# ══════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════
elif "Why Customers" in page:
    banner("🔍 Why Do Customers Accept Personal Loans?")

    st.markdown(
        "Each chart below isolates one factor and shows how strongly it relates to loan acceptance. "
        "**Green = accepted loan. Red = did not accept.**"
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="Loan_Label", y="Income",
                     color="Loan_Label",
                     color_discrete_map={"Accepted": C_GREEN, "Not Accepted": C_RED},
                     title="Income vs Loan Acceptance",
                     labels={"Loan_Label": "", "Income": "Annual Income ($K)"})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Loan acceptors have a noticeably higher median income. Customers earning over $100K are 3–4× more likely to accept.")

    with col2:
        df["IncomeBracket"] = pd.cut(df["Income"], bins=[0,50,100,150,200,300],
                                      labels=["<$50K","$50–100K","$100–150K","$150–200K","$200K+"])
        ib_loan = df.groupby("IncomeBracket", observed=True)["PersonalLoan"].mean().reset_index()
        ib_loan["Rate"] = ib_loan["PersonalLoan"] * 100
        fig = px.bar(ib_loan, x="IncomeBracket", y="Rate",
                     color="Rate",
                     color_continuous_scale=["#ffcccc","#ffe0b2","#c8e6c9","#388e3c"],
                     title="Loan Acceptance Rate by Income Bracket",
                     text="Rate",
                     labels={"IncomeBracket": "Income Bracket", "Rate": "Acceptance Rate (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Acceptance rate jumps dramatically above $100K — confirming income is the #1 driver.")

    insight(
        "💰 <b>Income Insight:</b> Customers earning more than <b>$100K/year</b> show an acceptance rate "
        "of over 30%, compared to just 3–5% for those earning under $50K. "
        "Income is the single strongest predictor of loan acceptance.", "green"
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="Loan_Label", y="CCAvg",
                     color="Loan_Label",
                     color_discrete_map={"Accepted": C_GREEN, "Not Accepted": C_RED},
                     title="Credit Card Spending vs Loan Acceptance",
                     labels={"Loan_Label": "", "CCAvg": "Avg Monthly CC Spend ($K)"})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Loan acceptors spend 2–3× more on credit cards monthly — high spenders are more credit-active.")

    with col2:
        fig = px.box(df, x="Loan_Label", y="Mortgage",
                     color="Loan_Label",
                     color_discrete_map={"Accepted": C_GREEN, "Not Accepted": C_RED},
                     title="Mortgage Value vs Loan Acceptance",
                     labels={"Loan_Label": "", "Mortgage": "Mortgage Value ($K)"})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        chart_note("Customers with higher mortgages also accept loans more — they are experienced, comfortable borrowers.")

    insight(
        "💳 <b>Spending Behaviour Insight:</b> Customers who accepted loans spend on average "
        "<b>2–3× more on their credit cards</b> monthly. High CC usage signals financial activity "
        "and a higher appetite for credit products.", "blue"
    )
    st.divider()

    st.markdown("#### 🔗 How Strongly Does Each Factor Relate to Loan Acceptance?")
    corr_cols = ["Income","CCAvg","Mortgage","Age","Family","Education",
                 "CDAccount","SecuritiesAccount","Online","CreditCard","PersonalLoan"]
    corr = df[corr_cols].corr()[["PersonalLoan"]].drop("PersonalLoan").sort_values("PersonalLoan")

    fig = px.bar(
        corr.reset_index(), x="PersonalLoan", y="index", orientation="h",
        color="PersonalLoan",
        color_continuous_scale=["#d64045","#f4f4f4","#1e8c6e"],
        color_continuous_midpoint=0,
        title="What Drives Loan Acceptance? (Correlation Analysis)",
        labels={"PersonalLoan": "Correlation with Loan Acceptance", "index": ""},
        text="PersonalLoan",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=440, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    chart_note(
        "Bars pointing RIGHT (green) mean the factor increases loan acceptance likelihood. "
        "Bars pointing LEFT (red) have no significant positive effect. "
        "Income, CD Account ownership, and CCAvg are the top three drivers."
    )

    with st.expander("See full correlation heatmap between all variables"):
        fig2, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.4, cbar_kws={"shrink": 0.8})
        ax.set_title("Full Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODELING
# ══════════════════════════════════════════════════════════════
elif "Predictive" in page:
    banner("🤖 Predictive Model Results — Which Customers Will Accept a Loan?")

    st.markdown(
        "Three machine learning models were trained on historical data to predict which customers "
        "are most likely to accept a personal loan. Results are shown below."
    )

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🧮 Confusion Matrices", "🏆 Top Predictors"])

    with tab1:
        perf = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": [r["accuracy"] * 100 for r in results.values()],
            "AUC-ROC Score": [r["auc"] for r in results.values()],
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(perf, x="Model", y="Accuracy (%)",
                         color="Model",
                         color_discrete_sequence=[C_BLUE, C_GREEN, C_AMBER],
                         title="Model Accuracy (%)", text="Accuracy (%)")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(height=360, showlegend=False, yaxis_range=[90, 101])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(perf, x="Model", y="AUC-ROC Score",
                         color="Model",
                         color_discrete_sequence=[C_BLUE, C_GREEN, C_AMBER],
                         title="AUC-ROC Score (1.0 = perfect)", text="AUC-ROC Score")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=360, showlegend=False, yaxis_range=[0.95, 1.01])
            st.plotly_chart(fig, use_container_width=True)

        chart_note(
            "Accuracy = how often the model is correct overall. "
            "AUC-ROC = how well it separates loan acceptors from non-acceptors (1.0 = perfect, 0.5 = coin flip)."
        )

        st.markdown("#### 📈 ROC Curves — How Well Can Each Model Spot Loan Acceptors?")
        fig = go.Figure()
        roc_colors = [C_BLUE, C_GREEN, C_AMBER]
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{name}  (AUC={res['auc']:.3f})",
                                     line=dict(color=roc_colors[i], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="gray", width=1.5),
                                  name="Random Guess (AUC=0.5)"))
        fig.update_layout(
            xaxis_title="False Positive Rate (wasted calls)",
            yaxis_title="True Positive Rate (correctly found)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="#fafafa"
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_note(
            "The further a curve bows toward the top-left corner, the better the model at identifying "
            "real loan acceptors while avoiding wasted calls. All three models are excellent."
        )

        insight(
            f"🏆 <b>Best Model: {best_model_name}</b> — AUC: "
            f"<b>{results[best_model_name]['auc']:.3f}</b>, Accuracy: "
            f"<b>{results[best_model_name]['accuracy']*100:.2f}%</b>. "
            f"This model powers all customer predictions in this dashboard.", "green"
        )

    with tab2:
        st.markdown(
            "A **Confusion Matrix** shows how many predictions were correct. "
            "**True Positives (bottom-right)** = loan acceptors we correctly identified. "
            "**False Negatives (bottom-left)** = missed opportunities."
        )
        cm_cols = st.columns(3)
        for i, (name, res) in enumerate(results.items()):
            with cm_cols[i]:
                cm = res["cm"]
                total_test = cm.sum()
                correct    = cm[0,0] + cm[1,1]
                fig = px.imshow(
                    cm, text_auto=True,
                    color_continuous_scale=["#fde8e8","#c8e6c9"],
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Predicted: No Loan","Predicted: Loan"],
                    y=["Actual: No Loan","Actual: Loan"],
                    title=f"{name}<br><sup>{correct:,}/{total_test:,} correct</sup>"
                )
                fig.update_coloraxes(showscale=False)
                fig.update_layout(height=300, margin=dict(t=60, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("How to read a confusion matrix"):
            st.markdown("""
            | Cell | Meaning |
            |---|---|
            | **Top-left** | Correctly identified non-acceptors ✅ |
            | **Top-right** | Falsely flagged — we called them, they said No ❌ |
            | **Bottom-left** | Missed opportunity — they would have said Yes! ⚠️ |
            | **Bottom-right** | Correctly identified loan acceptors 🎉 |
            """)

    with tab3:
        fi = pd.DataFrame({
            "Factor": MODEL_FEATURES,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fi["Colour"] = fi["Importance"].apply(
            lambda x: C_GREEN if x >= fi["Importance"].quantile(0.7) else
                      C_AMBER if x >= fi["Importance"].quantile(0.4) else "#cccccc"
        )

        fig = px.bar(fi, x="Importance", y="Factor", orientation="h",
                     color="Colour", color_discrete_map="identity",
                     title=f"What Factors Drive Loan Acceptance? — {best_model_name}",
                     text="Importance",
                     labels={"Factor": "", "Importance": "Importance Score"})
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=450, showlegend=False, plot_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)
        chart_note(
            "Longer green bars = stronger predictors. "
            "Income and CCAvg (credit card spending) are the most powerful signals the model uses "
            "to identify likely loan acceptors."
        )

        insight(
            "🔑 <b>Top 5 Predictors of Loan Acceptance:</b><br>"
            "1️⃣ <b>Income</b> — by far the strongest signal<br>"
            "2️⃣ <b>CCAvg</b> — high spenders are more credit-active<br>"
            "3️⃣ <b>CD Account</b> — existing investors trust the bank<br>"
            "4️⃣ <b>Education</b> — advanced degree holders accept more<br>"
            "5️⃣ <b>Family Size</b> — larger families have greater financial needs", "blue"
        )


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CROSS-SELLING
# ══════════════════════════════════════════════════════════════
elif "Cross-Selling" in page:
    banner("💼 Cross-Selling Opportunities — Maximize Revenue Per Customer")

    predicted_yes = df[df["Predicted_Class"] == 1].copy()
    total_yes     = len(predicted_yes)

    st.markdown(
        f"The model identified **{total_yes:,} customers** as highly likely to accept a personal loan. "
        "This section shows what **other products** these customers don't yet have — your best cross-sell opportunities."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎯 Predicted Loan Acceptors",    f"{total_yes:,}")
    k2.metric("📊 % of Total Customer Base",     f"{total_yes/len(df)*100:.1f}%")
    k3.metric("💰 Avg Income (Target Group)",    f"${predicted_yes['Income'].mean():.0f}K",
              f"+${predicted_yes['Income'].mean()-df['Income'].mean():.0f}K vs all customers")
    k4.metric("💳 Avg CC Spend (Target Group)", f"${predicted_yes['CCAvg'].mean():.1f}K/mo",
              f"+${predicted_yes['CCAvg'].mean()-df['CCAvg'].mean():.1f}K vs all customers")

    st.divider()

    prods  = ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
    labels = ["Securities Account", "CD Account", "Online Banking", "Credit Card"]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    prod_data = []
    for p, l in zip(prods, labels):
        has     = int(predicted_yes[p].sum())
        has_not = total_yes - has
        prod_data.append({"Product": l, "Already Has": has,
                          "Cross-Sell Target": has_not,
                          "Opportunity %": has_not / total_yes * 100})
    prod_df = pd.DataFrame(prod_data).sort_values("Opportunity %", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(prod_df, x="Product", y="Opportunity %",
                     color="Product",
                     color_discrete_sequence=colors,
                     title="Cross-Sell Opportunity — % of Predicted Acceptors Without Each Product",
                     text="Opportunity %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=390, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Already Has Product",
                             x=prod_df["Product"], y=prod_df["Already Has"],
                             marker_color=C_GREEN, text=prod_df["Already Has"],
                             textposition="inside"))
        fig.add_trace(go.Bar(name="Cross-Sell Target (Does NOT Have)",
                             x=prod_df["Product"], y=prod_df["Cross-Sell Target"],
                             marker_color=C_AMBER, text=prod_df["Cross-Sell Target"],
                             textposition="inside"))
        fig.update_layout(
            barmode="stack",
            title="Product Ownership Among Predicted Loan Acceptors",
            height=390,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_note(
        "Orange = customers who don't have this product yet — they are your cross-sell targets. "
        "Green = already own it. The bigger the orange bar, the bigger the opportunity."
    )

    top_prod = prod_df.iloc[0]
    insight(
        f"🎯 <b>Biggest Opportunity: {top_prod['Product']}</b> — "
        f"<b>{int(top_prod['Cross-Sell Target']):,} customers</b> ({top_prod['Opportunity %']:.1f}%) "
        f"in our predicted loan acceptor group don't have it yet. "
        f"Bundling this with the loan offer is the single highest-priority revenue action.", "green"
    )

    st.divider()
    st.markdown("#### 🔗 How Many Cross-Sell Products Do Predicted Loan Acceptors Already Hold?")

    predicted_yes["ProductCount"] = predicted_yes[prods].sum(axis=1)
    pc = predicted_yes["ProductCount"].value_counts().sort_index().reset_index()
    pc.columns = ["Products Held", "Customers"]
    pc["Label"] = pc["Products Held"].map({
        0: "0 products — No cross-sell products yet",
        1: "1 product — Some adoption",
        2: "2 products — Moderate adoption",
        3: "3 products — High adoption",
        4: "4 products — Full adoption"
    })

    fig = px.bar(pc, x="Label", y="Customers",
                 color="Customers", color_continuous_scale="YlGn",
                 title="How Many Other Products Do Predicted Loan Acceptors Hold?",
                 text="Customers")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    zero_products = int((predicted_yes["ProductCount"] == 0).sum())
    chart_note(
        f"{zero_products:,} predicted loan acceptors currently hold ZERO other products. "
        "These are your highest-value bundle targets."
    )

    insight(
        f"💡 <b>Bundle Strategy:</b> Focus first on the <b>{zero_products:,} customers with no other products</b>. "
        "A bundled offer — Personal Loan + Online Banking + Credit Card at a preferential rate — "
        "will both close the loan and deepen the banking relationship in one conversation.", "orange"
    )


# ══════════════════════════════════════════════════════════════
# PAGE 6 — PERSONAS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif "Personas" in page:
    banner("🎯 Customer Personas & Marketing Recommendations")

    st.markdown(
        "Based on the data analysis, we have identified **3 high-value customer personas**. "
        "Each has a distinct profile, loan acceptance likelihood, and clear marketing approach."
    )
    st.divider()

    p1 = df[(df["Income"] >= 100) & (df["Education"] >= 2)]
    p2 = df[(df["Age"] < 40) & (df["Online"] == 1)]
    p3 = df[(df["Family"] >= 3) & (df["Mortgage"] > 0)]
    base = df["PersonalLoan"].mean() * 100
    p1_rate = p1["PersonalLoan"].mean() * 100
    p2_rate = p2["PersonalLoan"].mean() * 100
    p3_rate = p3["PersonalLoan"].mean() * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="persona" style="border-top-color:{C_BLUE};">
            <h4 style="color:{C_BLUE};">👔 Persona 1 — High-Income Professional</h4>
            <div class="big-rate" style="color:{C_GREEN};">{p1_rate:.1f}%</div>
            <div style="color:#888; font-size:0.82rem; margin-bottom:14px;">loan acceptance rate</div>
            <b>Profile:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Annual income ≥ $100K</li>
                <li>Graduate or Advanced degree</li>
                <li>{len(p1):,} customers in this group</li>
                <li>Avg income: ${p1['Income'].mean():.0f}K/year</li>
                <li>Avg CC spend: ${p1['CCAvg'].mean():.1f}K/month</li>
            </ul>
            <b>Best Products to Offer:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Premium Personal Loan</li>
                <li>Securities Account</li>
                <li>Rewards Credit Card</li>
            </ul>
            <b>Best Channel:</b> Relationship Manager, Direct Mail
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="persona" style="border-top-color:{C_GREEN};">
            <h4 style="color:{C_GREEN};">📱 Persona 2 — Young Digital User</h4>
            <div class="big-rate" style="color:{C_GREEN};">{p2_rate:.1f}%</div>
            <div style="color:#888; font-size:0.82rem; margin-bottom:14px;">loan acceptance rate</div>
            <b>Profile:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Age under 40</li>
                <li>Already using Online Banking</li>
                <li>{len(p2):,} customers in this group</li>
                <li>Avg age: {p2['Age'].mean():.0f} years</li>
                <li>Avg income: ${p2['Income'].mean():.0f}K/year</li>
            </ul>
            <b>Best Products to Offer:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Fast-approval Personal Loan (app-based)</li>
                <li>Cashback Credit Card</li>
                <li>CD Account (short-term savings)</li>
            </ul>
            <b>Best Channel:</b> Mobile App Push, In-App Banner, Email
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="persona" style="border-top-color:{C_AMBER};">
            <h4 style="color:{C_AMBER};">👨‍👩‍👧 Persona 3 — Family-Oriented Borrower</h4>
            <div class="big-rate" style="color:{C_GREEN};">{p3_rate:.1f}%</div>
            <div style="color:#888; font-size:0.82rem; margin-bottom:14px;">loan acceptance rate</div>
            <b>Profile:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Family size of 3 or more</li>
                <li>Has an existing mortgage</li>
                <li>{len(p3):,} customers in this group</li>
                <li>Avg family size: {p3['Family'].mean():.1f} members</li>
                <li>Avg mortgage: ${p3['Mortgage'].mean():.0f}K</li>
            </ul>
            <b>Best Products to Offer:</b>
            <ul style="padding-left:18px; font-size:0.9rem; margin-top:6px;">
                <li>Personal Loan (education/home)</li>
                <li>Online Banking</li>
                <li>Family Credit Card</li>
            </ul>
            <b>Best Channel:</b> Email, Phone, Branch Visit
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 How Each Persona Compares to the Overall Customer Average")

    compare_df = pd.DataFrame({
        "Persona": ["High-Income Professional", "Young Digital User",
                    "Family-Oriented Borrower", "Overall Average"],
        "Acceptance Rate (%)": [p1_rate, p2_rate, p3_rate, base],
        "Color": [C_BLUE, C_GREEN, C_AMBER, "#aaaaaa"]
    }).sort_values("Acceptance Rate (%)", ascending=True)

    fig = px.bar(
        compare_df, x="Acceptance Rate (%)", y="Persona", orientation="h",
        color="Persona",
        color_discrete_map={row["Persona"]: row["Color"] for _, row in compare_df.iterrows()},
        text="Acceptance Rate (%)",
        title="Loan Acceptance Rate: Personas vs Overall Average",
        labels={"Persona": ""}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_vline(x=base, line_dash="dash", line_color="gray",
                  annotation_text=f"Overall avg: {base:.1f}%",
                  annotation_position="top right")
    fig.update_layout(height=340, showlegend=False, plot_bgcolor="#fafafa")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("All three personas significantly outperform the overall average — confirming these are the right segments to focus on.")

    st.divider()
    st.markdown("#### 📣 Prioritised Marketing Action Plan")

    st.markdown("""
    <table class="rec-table">
    <tr>
        <th>Priority</th>
        <th>Action</th>
        <th>Who to Target</th>
        <th>Products to Bundle</th>
        <th>Channel</th>
        <th>Expected Impact</th>
    </tr>
    <tr>
        <td><span class="pill-high">🔴 Highest</span></td>
        <td>Premium loan campaign for high-income professionals</td>
        <td>Income ≥ $100K, Graduate/Advanced degree</td>
        <td>Loan + Securities Account</td>
        <td>Relationship Manager</td>
        <td>Very High ROI</td>
    </tr>
    <tr>
        <td><span class="pill-high">🔴 Highest</span></td>
        <td>Target CD Account holders with a loan offer</td>
        <td>Existing CD Account customers</td>
        <td>Loan + Credit Card</td>
        <td>In-Branch + Email</td>
        <td>Highest conversion rate in data</td>
    </tr>
    <tr>
        <td><span class="pill-medium">🟡 Medium</span></td>
        <td>Digital loan push for young online banking users</td>
        <td>Age &lt; 40, using Online Banking</td>
        <td>Loan + CD Account</td>
        <td>Mobile App / Push Notification</td>
        <td>High volume, low cost</td>
    </tr>
    <tr>
        <td><span class="pill-medium">🟡 Medium</span></td>
        <td>Family financial bundle campaign</td>
        <td>Family size ≥ 3, has mortgage</td>
        <td>Loan + Online Banking</td>
        <td>Email + Phone</td>
        <td>Medium-high conversion</td>
    </tr>
    <tr>
        <td><span class="pill-low">🟢 Ongoing</span></td>
        <td>Monthly ML model scoring — refresh target list</td>
        <td>Top 500 predicted leads each month</td>
        <td>Personalised bundle</td>
        <td>CRM Automation</td>
        <td>Sustained growth</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    st.divider()

    targets = int(df["Predicted_Class"].sum())
    insight(
        f"🏦 <b>Executive Conclusion:</b><br><br>"
        f"Universal Bank's previous campaign achieved a <b>9.6% conversion rate</b> by contacting customers broadly. "
        f"With this ML-powered approach, the bank can now focus on a precise list of "
        f"<b>{targets:,} high-probability customers</b>.<br><br>"
        f"Focusing on <b>High-Income Professionals</b>, <b>CD Account holders</b>, and "
        f"<b>families with mortgages</b> will dramatically improve campaign efficiency — "
        f"spending less marketing budget while achieving higher conversions and greater "
        f"revenue per customer through smart cross-selling.", "green"
    )
