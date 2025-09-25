
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
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

# ---------- Header / Tabs ----------
st.title("Smart Recommendations - Streamlit")
st.caption("Interactive demo of your FYP: clustering + classification on mixed-type data (ready for Streamlit Cloud).")

about_tab, data_tab, cluster_tab, model_tab, export_tab = st.tabs(["About", "Data", "Clustering", "Modeling", "Export"])

with about_tab:
    st.markdown("""

### Project Overview
Problem - Traditional recommenders struggle with context, sparsity, and cold-starts.  
Objective - Build a pipeline that segments users (clustering) and predicts intent (classification).  
Methods - K-Prototypes / KMeans for clustering; Logistic Regression, Random Forest, and Gradient Boosting for prediction.

Workflow
1) Load dataset (CSV/Excel) or use synthetic demo  
2) Preprocess (encode categoricals, scale numerics)  
3) Cluster users into segments  
4) Train/test split for supervised models  
5) Train models and evaluate (Accuracy, Precision, Recall, F1, Confusion Matrix)

Tip: On Streamlit Cloud, keep large/private datasets out of the repo. Let users upload at runtime.
""")

with st.sidebar:
    st.header("Settings")
    data_src = st.radio("Data Source", ["Upload file", "Auto-detect local file", "Use synthetic demo"], index=0)
    target_col = st.text_input("Target column (classification)", value="Revenue")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    st.markdown("---")
    st.subheader("Scaling")
    scale_choice = st.selectbox("Numeric scaling", ["None", "StandardScaler", "MinMaxScaler"], index=1)
    st.markdown("---")
    st.subheader("Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 4, 1)
    st.caption("If K-Prototypes not installed, fallback is KMeans on One-Hot encoded features.")
    st.markdown("---")
    st.subheader("Models")
    model_choices = st.multiselect(
        "Choose models",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

@st.cache_data
def load_local_default():
    # Auto-detect a common dataset filename if present
    from pathlib import Path
    for name in ["online_shoppers_intention.xlsx", "OnlineShoppersIntention.xlsx", "online_shoppers_intention.csv"]:
        try:
            if Path(name).exists():
                if name.endswith(".csv"):
                    return pd.read_csv(name)
                else:
                    return pd.read_excel(name)
        except Exception:
            pass
    return None

@st.cache_data
def make_synthetic():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({
        "Administrative": rng.integers(0, 20, size=n),
        "Informational": rng.integers(0, 15, size=n),
        "ProductRelated": rng.integers(1, 300, size=n),
        "BounceRates": rng.random(n) * 0.2,
        "ExitRates": rng.random(n) * 0.4,
        "PageValues": rng.random(n) * 20,
        "SpecialDay": rng.random(n),
        "Month": rng.choice(["Jan","Feb","Mar","May","Nov","Dec"], size=n, p=[.1,.1,.2,.2,.25,.15]),
        "OperatingSystems": rng.integers(1, 4, size=n),
        "Browser": rng.integers(1, 6, size=n),
        "Region": rng.integers(1, 10, size=n),
        "TrafficType": rng.integers(1, 15, size=n),
        "VisitorType": rng.choice(["New_Visitor","Returning_Visitor"], size=n, p=[.3,.7]),
        "Weekend": rng.choice([False, True], size=n, p=[.7,.3]),
    })
    logits = (
        0.01*df["ProductRelated"]
        + 0.5*df["SpecialDay"]
        + 0.02*df["PageValues"]
        - 0.8*df["BounceRates"]
        - 0.2*df["ExitRates"]
        + (df["VisitorType"]=="Returning_Visitor").astype(int)*0.4
    )
    prob = 1/(1+np.exp(-logits/2.0))
    df["Revenue"] = (rng.random(n) < prob).astype(int)
    return df

with data_tab:
    st.subheader("Load Data")
    df = None
    if data_src == "Upload file":
        up = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
        if up is not None:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    elif data_src == "Auto-detect local file":
        df = load_local_default()
        if df is None:
            st.info("No local dataset found. Place 'online_shoppers_intention.xlsx' in the app folder or switch to upload/demo.")
    else:
        df = make_synthetic()

    if df is None:
        st.stop()

    st.success(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head())

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found. Please set it correctly in the sidebar.")
        st.stop()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols = list(X.select_dtypes(include=["object","category","bool"]).columns)
    num_cols = list(X.select_dtypes(include=["number"]).columns)

    st.write(f"Categorical: {cat_cols if cat_cols else 'None'}")
    st.write(f"Numeric: {num_cols if num_cols else 'None'}")

    # Encode categoricals for supervised models
    X_enc = X.copy()
    label_maps = {}
    for c in cat_cols:
        le = LabelEncoder()
        X_enc[c] = le.fit_transform(X_enc[c].astype(str))
        label_maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Scale numerics if chosen
    if scale_choice == "StandardScaler":
        scaler = StandardScaler()
        X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])
    elif scale_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

with cluster_tab:
    st.subheader("Clustering")
    if df is None:
        st.stop()

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

    st.markdown("Cluster Visualization (PCA 2D)")
    pca = PCA(n_components=2, random_state=int(random_state))
    X_vis = pca.fit_transform(X_enc)
    fig, ax = plt.subplots()
    sc = ax.scatter(X_vis[:,0], X_vis[:,1], c=clusters, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clusters (PCA projection)")
    st.pyplot(fig)

with model_tab:
    st.subheader("Modeling & Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=float(test_size), random_state=int(random_state),
        stratify=y if y.nunique() <= 10 else None
    )

    results = {}
    if "Logistic Regression" in model_choices:
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        results["Logistic Regression"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, zero_division=0, output_dict=False),
        }

    if "Random Forest" in model_choices:
        rf = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results["Random Forest"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, zero_division=0, output_dict=False),
        }

    if "Gradient Boosting" in model_choices:
        gb = GradientBoostingClassifier(random_state=int(random_state))
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        results["Gradient Boosting"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score[y_test, y_pred] if False else precision_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "recall": recall_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "f1": f1_score(y_test, y_pred, zero_division=0, average="binary" if y.nunique()==2 else "macro"),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, zero_division=0, output_dict=False),
        }

    if not results:
        st.warning("Select at least one model in the sidebar.")
    else:
        rows = [[name, r["accuracy"], r["precision"], r["recall"], r["f1"]] for name, r in results.items()]
        summary_df = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1"])
        st.dataframe(summary_df.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}))

        fig2, ax2 = plt.subplots()
        ax2.bar(summary_df["Model"], summary_df["Accuracy"])
        ax2.set_ylim(0,1); ax2.set_ylabel("Accuracy"); ax2.set_title("Model Accuracy Comparison")
        st.pyplot(fig2)

        st.markdown("Confusion Matrices")
        cols = st.columns(len(results))
        for (name, r), col in zip(results.items(), cols):
            with col:
                cm = r["cm"]
                fig_cm, ax_cm = plt.subplots()
                im = ax_cm.imshow(cm, interpolation="nearest")
                ax_cm.set_title(name)
                ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
                for (i, j), v in np.ndenumerate(cm):
                    ax_cm.text(j, i, str(v), ha="center", va="center")
                st.pyplot(fig_cm)
                st.text(r["report"])

with export_tab:
    st.subheader("Export")
    if 'df_clusters' in locals():
        csv = df_clusters.to_csv(index=False).encode("utf-8")
        st.download_button("Download clustered data as CSV", csv, file_name="clustered_data.csv", mime="text/csv")
    else:
        st.info("Run clustering first to enable export.")
