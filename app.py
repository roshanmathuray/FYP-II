import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Try K-Prototypes (mixed-type clustering) if available
try:
    from kmodes.kprototypes import KPrototypes
    HAS_KPROTOTYPES = True
except Exception:
    HAS_KPROTOTYPES = False

st.set_page_config(page_title="Smart Recommendations - Streamlit (Random Forest)", layout="wide")
st.title("Smart Recommendations - Streamlit (Random Forest)")
st.caption("Auto-loads local dataset and shows results immediately.")

# Optional: list files to confirm dataset is present
st.write("Files in current directory:", os.listdir("."))

# -------- Load dataset (Excel preferred, CSV fallback) --------
DATA_XLSX = "online_shoppers_intention.xlsx"
DATA_CSV  = "online_shoppers_intention.csv"

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_XLSX):
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
scale_choice = "StandardScaler"   # or "MinMaxScaler" or "None"
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

# ---- Helper: One-Hot + numeric matrix for KMeans/Elbow (works even if K-Prototypes is used)
def build_kmeans_matrix(X_in, cat_cols_in, num_cols_in):
    if len(cat_cols_in) > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_ohe = ohe.fit_transform(X_in[cat_cols_in].astype(str))
        X_ohe_df = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(cat_cols_in), index=X_in.index)
        X_km_mat = pd.concat([X_ohe_df, X_in[num_cols_in]], axis=1)
    else:
        X_km_mat = X_in[num_cols_in].copy()
    return X_km_mat

# Label-encode categoricals for supervised model
X_enc = X.copy()
label_maps = {}
for c in cat_cols:
    le = LabelEncoder()
    X_enc[c] = le.fit_transform(X_enc[c].astype(str))
    label_maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))

# Scale numerics if selected
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
    X_km_mat_for_run = build_kmeans_matrix(X, cat_cols, num_cols)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(random_state))
    clusters = kmeans.fit_predict(X_km_mat_for_run)

df_clusters = df.copy()
df_clusters["Cluster"] = clusters
st.dataframe(df_clusters.head())

# PCA plot for clusters (on encoded/scaled features for visualization)
st.markdown("**Clusters (PCA 2D projection)**")
pca = PCA(n_components=2, random_state=int(random_state))
X_vis = pca.fit_transform(X_enc)
fig, ax = plt.subplots()
ax.scatter(X_vis[:,0], X_vis[:,1], c=clusters, alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Clusters (PCA projection)")
st.pyplot(fig)

# ---------------- Elbow Graph (KMeans WCSS vs K) ----------------
st.markdown("**Elbow Graph (WCSS vs K)**")
X_km_mat = build_kmeans_matrix(X, cat_cols, num_cols)

wcss = []
K_range = range(2, 11)  # K = 2..10
for k in K_range:
    km_tmp = KMeans(n_clusters=k, n_init=10, random_state=int(random_state))
    km_tmp.fit(X_km_mat)
    wcss.append(km_tmp.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(list(K_range), wcss, marker="o")
ax_elbow.set_xlabel("Number of clusters (K)")
ax_elbow.set_ylabel("WCSS (inertia)")
ax_elbow.set_title("Elbow Method for KMeans")
st.pyplot(fig_elbow)

# ---------------- Revenue Graphs ----------------
if "Revenue" in df.columns:
    st.subheader("Revenue Graphs")

    # (a) Overall Revenue distribution
    st.markdown("**Overall Revenue Distribution**")
    rev_counts = df["Revenue"].value_counts().sort_index()
    fig_rev, ax_rev = plt.subplots()
    ax_rev.bar(rev_counts.index.astype(str), rev_counts.values)
    ax_rev.set_xlabel("Revenue (0 = No, 1 = Yes)")
    ax_rev.set_ylabel("Count")
    ax_rev.set_title("Revenue Class Counts")
    st.pyplot(fig_rev)

    # (b) Revenue rate by cluster (mean of Revenue within each cluster)
    st.markdown("**Revenue Rate by Cluster**")
    rev_by_cluster = df_clusters.groupby("Cluster")["Revenue"].mean()
    fig_rc, ax_rc = plt.subplots()
    ax_rc.bar(rev_by_cluster.index.astype(str), rev_by_cluster.values)
    ax_rc.set_xlabel("Cluster")
    ax_rc.set_ylabel("Revenue Rate")
    ax_rc.set_title("Average Revenue (Purchase Intention) by Cluster")
    st.pyplot(fig_rc)
else:
    st.info("Column 'Revenue' not found, so revenue graphs are skipped. Set the correct target column name above if needed.")

# -------- Random Forest Only --------
st.subheader("Modeling & Evaluation (Random Forest)")
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=float(test_size), random_state=int(random_state),
    stratify=y if y.nunique() <= 10 else None
)

rf = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro")
rec = recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro")
f1 = f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro")
cm = confusion_matrix(y_test, y_pred)
report_text = classification_report(y_test, y_pred, zero_division=0)

# Metrics table
st.markdown("**Metrics**")
metrics_df = pd.DataFrame(
    [{"Model": "Random Forest", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}]
)
st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}))

# Confusion matrix
st.markdown("**Confusion Matrix (Random Forest)**")
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm, interpolation="nearest")
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    ax_cm.text(j, i, str(v), ha="center", va="center")
st.pyplot(fig_cm)

# Classification report text
st.markdown("**Classification Report (Random Forest)**")
st.text(report_text)

# Export clustered data
csv = df_clusters.to_csv(index=False).encode("utf-8")
st.download_button("Download clustered data as CSV", csv, file_name="clustered_data.csv", mime="text/csv")

