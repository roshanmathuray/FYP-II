# Smart Recommendations - Streamlit App

This is a Streamlit app demonstrating a clustering + classification pipeline for your FYP project.

## Features
- Upload CSV/Excel or auto-detect `online_shoppers_intention.xlsx`
- Preprocessing (label encoding + optional scaling)
- Clustering: K-Prototypes (if available) or fallback to KMeans + OneHot
- Models: Logistic Regression, Random Forest, Gradient Boosting
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- Visuals: PCA cluster plot, accuracy bar chart, confusion matrices
- Export clustered data as CSV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub (public or private).
2. Go to https://streamlit.io/cloud and click **New app**.
3. Select your repo, branch, and set **App file = app.py**.
4. Click **Deploy** → get a public URL like `https://your-app.streamlit.app`.

---
For academic/demo purposes only.
