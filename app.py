
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Try K-Prototypes (mixed-type clustering) if available
try:
    from kmodes.kprototypes import KPrototypes
    HAS_KPROTOTYPES = True
except Exception:
    HAS_KPROTOTYPES = False

st.set_page_config(page_title="Smart Recommendations - Streamlit", layout="wide")
st.title("Smart Recommendations - Streamlit")
st.caption("Auto-loads local dataset and shows results immediately.")

# Show files for sanity
st.write("Files in current directory:", os.listdir("."))

# -------- Load dataset (Excel preferred, CSV fallback) --------
DATA_XLSX = "online_shoppers_intention.xlsx"
DATA_CSV  = "online_shoppers_intention.csv"

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_XLSX):
        # Explicit engine ensures Streamlit Cloud installs openpyxl matching our requirements
        df = pd.read_excel(DATA_XLSX, engine="openpyxl")
        return df, DATA_XLSX
    elif os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        return df, DATA_CSV
    else:
        return None, None

df, used_name = load_dataset()
if df is None:
    st.error("Dataset not found. Put 'online_shoppers_intention.xlsx' (Excel) or 'online_shoppers_intention.csv' (CSV) next to app.py in the repo.")
    st.stop()

st.success(f"Loaded '{used_name}' with shape: {df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df.head())

# -------- Parameters --------
target_col = "Revenue" if "Revenue" in df.columns else st.text_input("Target column name", value="Revenue")
test_size = 0.2
random_state = 42
scale_choice = "StandardScaler"
n_clusters = 4

# -------- Preprocess --------
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

y = df[target_col]
X = df.drop(columns=[target_col])

cat_cols = list(X.select_dtypes(include=["object","category","bool"]).columns)
num_cols = list(X.select_dtypes(include=["number"]).columns)
st.write(f"Categorical columns: {cat_cols if cat_cols else 'None'}")
st.write(f"Numeric columns: {num_cols if num_cols else 'None'}")

X_enc = X.copy()
label_maps = {}
for c in cat_cols:
    le = LabelEncoder()
    X_enc[c] = le.fit_transform(X_enc[c].astype(str))
    label_maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))

if scale_choice == "StandardScaler":
    scaler = StandardScaler()
    X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])
elif scale_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

# -------- Clustering --------
st.subheader("Clustering")
if HAS_KPROTOTYPES and len(cat_cols) > 0:
    st.caption("Using K-Prototypes for mixed-type clustering.")
    categorical_idx = [X.columns.get_loc(c) for c in cat_cols]
    X_for_kproto = X.copy()
    for c in cat_cols:
        X_for_kproto[c] = X_for_kproto[c].astype(str)
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', verbose=0, random_state=int(random_state))
    clusters = kproto.fit_predict(X_for_kproto.to_numpy(), categorical=categorical_idx)
else:
    st.caption("Using KMeans on One-Hot encoded features (fallback).")
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_ohe = ohe.fit_transform(X[cat_cols].astype(str))
        X_ohe_df = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(cat_cols), index=X.index)
        X_km = pd.concat([X_ohe_df, X[num_cols]], axis=1)
    else:
        X_km = X[num_cols].copy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(random_state))
    clusters = kmeans.fit_predict(X_km)

df_clusters = df.copy()
df_clusters["Cluster"] = clusters
st.dataframe(df_clusters.head())

# PCA plot
st.markdown("**Clusters (PCA 2D projection)**")
pca = PCA(n_components=2, random_state=int(random_state))
X_vis = pca.fit_transform(X_enc)
fig, ax = plt.subplots()
ax.scatter(X_vis[:,0], X_vis[:,1], c=clusters, alpha=0.7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clusters (PCA projection)")
st.pyplot(fig)

# -------- Models --------
st.subheader("Modeling & Evaluation")
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=float(test_size), random_state=int(random_state),
    stratify=y if y.nunique() <= 10 else None
)

results = {}
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "cm": confusion_matrix(y_test, y_pred),
}

rf = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "cm": confusion_matrix(y_test, y_pred),
}

gb = GradientBoostingClassifier(random_state=int(random_state))
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
results["Gradient Boosting"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
    "cm": confusion_matrix(y_test, y_pred),
}

# Summary
summary_df = pd.DataFrame([[k, v["accuracy"], v["precision"], v["recall"], v["f1"]] for k,v in results.items()],
                          columns=["Model","Accuracy","Precision","Recall","F1"])
st.dataframe(summary_df.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}))

fig2, ax2 = plt.subplots()
ax2.bar(summary_df["Model"], summary_df["Accuracy"])
ax2.set_ylim(0,1); ax2.set_ylabel("Accuracy"); ax2.set_title("Model Accuracy Comparison")
st.pyplot(fig2)

# CMs
st.markdown("**Confusion Matrices**")
cols = st.columns(len(results))
for (name, r), col in zip(results.items(), cols):
    with col:
        cm = r["cm"]
        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, interpolation="nearest")
        ax_cm.set_title(name)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax_cm.text(j, i, str(v), ha="center", va="center")
        st.pyplot(fig_cm)

# Export
csv = df_clusters.to_csv(index=False).encode("utf-8")
st.download_button("Download clustered data as CSV", csv, file_name="clustered_data.csv", mime="text/csv")
