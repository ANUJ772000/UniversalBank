# 🏦 Universal Bank — Personal Loan Campaign Dashboard

A complete data analytics and machine learning dashboard built with **Streamlit**, analyzing Universal Bank's personal loan campaign data to identify high-conversion customers and cross-selling opportunities.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📋 Project Overview

Universal Bank wants to convert **liability customers** (depositors) into **asset customers** (personal loan holders). Last year's campaign achieved a **9.6% conversion rate**. This dashboard uses data analytics and ML to dramatically improve targeting precision.

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| 🏠 Overview | KPI metrics, acceptance rate, dataset preview |
| 📊 Descriptive Analysis | Income, age, education, product adoption charts |
| 🔍 Diagnostic Analysis | Drivers of loan acceptance, correlation heatmap |
| 🤖 Predictive Modeling | Decision Tree, Random Forest, Gradient Boosting models |
| 💼 Cross-Selling | Product gaps among predicted loan acceptors |
| 👥 Personas & Recommendations | Customer segments and marketing strategies |

---

## 🤖 Model Performance

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Decision Tree | 97.76% | 0.9846 |
| Random Forest | **98.96%** | **0.9987** |
| Gradient Boosting | 98.80% | 0.9986 |

---

## 📁 Repository Structure

```
universalbank/
├── app.py                  # Main Streamlit dashboard (single file)
├── UniversalBank.xlsx      # Dataset (5,000 customers, 14 features)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── .gitignore
└── README.md
```

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 📦 Dataset Features

| Feature | Description |
|---|---|
| Age | Customer age |
| Experience | Years of professional experience |
| Income | Annual income ($K) |
| Family | Family size |
| CCAvg | Avg monthly credit card spend ($K) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage value ($K) |
| **PersonalLoan** | **Target: 1=Accepted, 0=Not Accepted** |
| SecuritiesAccount | Has securities account |
| CDAccount | Has CD account |
| Online | Uses online banking |
| CreditCard | Has credit card |

---

## 🔑 Key Findings

- **Income** is the strongest predictor of loan acceptance
- Customers earning **$100K+** accept at 3–4× the base rate
- **CD Account holders** show the highest correlation with loan acceptance
- **Advanced/Professional degree** holders convert at 2× the rate of undergrads
- The ML model identifies **~500 high-probability targets** for focused campaigns

---

## 🏗️ Built With

- [Streamlit](https://streamlit.io/) — Dashboard framework
- [scikit-learn](https://scikit-learn.org/) — Machine learning models
- [Plotly](https://plotly.com/) — Interactive visualizations
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Data processing
- [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/) — Statistical plots
