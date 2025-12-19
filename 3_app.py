import streamlit as st
import deepchem as dc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from stmol import showmol
import py3Dmol

# --- CONFIGURATION ---
st.set_page_config(page_title="AyurSafe AI Research Platform", page_icon="🧬", layout="wide")

# --- CUSTOM CSS (For Professional Look) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# --- LOAD MODELS ---
@st.cache_resource
def load_model():
    return joblib.load('toxicity_model.pkl')


@st.cache_resource
def get_featurizer():
    return dc.feat.CircularFingerprint(size=1024, radius=2)


try:
    model = load_model()
    featurizer = get_featurizer()
except:
    st.error("⚠️ Model files not found. Please ensure 'toxicity_model.pkl' is in the folder.")


# --- HELPER FUNCTIONS ---
def draw_radar_chart(features, title):
    # Normalize features to 0-1 scale for the chart
    labels = list(features.keys())
    values = list(features.values())

    # Close the polygon
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#4CAF50', alpha=0.25)
    ax.plot(angles, values, color='#4CAF50', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, y=1.1, fontsize=10, weight='bold')
    return fig


# --- SIDEBAR (THE CONTROL CENTER) ---
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-ayurveda-logo-template_23-2149405626.jpg", width=120)
st.sidebar.title("AyurSafe AI 🧬")
st.sidebar.markdown("**Research-Grade Screening Tool**")
st.sidebar.markdown("---")

# 1. Mode Selection
mode = st.sidebar.radio("Select Workflow:", ["Single Molecule Lab", "Batch Screening (CSV)"])

# 2. Analysis Modules (The "Swiss Army Knife" Options)
st.sidebar.markdown("### ⚙️ Analysis Modules")
run_tox = st.sidebar.checkbox("Toxicity Prediction (AI)", value=True)
run_adme = st.sidebar.checkbox("ADME & Lipinski Rules", value=True)
run_qed = st.sidebar.checkbox("QED Drug-Likeness Score", value=False)
run_radar = st.sidebar.checkbox("Generate Thesis Radar Plot", value=False)

st.sidebar.markdown("---")
st.sidebar.info("🎓 **For Academic Use Only**\n\n cite: AyurSafe AI v1.0 (2025)")

# --- MAIN APP LOGIC ---

# === MODE 1: SINGLE MOLECULE LAB ===
if mode == "Single Molecule Lab":
    st.title("🧪 In-Silico Drug Discovery Lab")
    st.markdown("Analyze molecular candidates for Safety, Bioavailability, and Drug-Likeness.")

    col_input, col_vis = st.columns([1, 1])

    with col_input:
        # Default: Curcumin (Turmeric)
        smiles = st.text_area("Enter SMILES String:", "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
                              height=100)
        analyze_btn = st.button("🚀 Run Analysis")

    if analyze_btn and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("Invalid SMILES string.")
            else:
                # --- VISUALIZATION ---
                with col_vis:
                    st.markdown("**3D Structure Viewer**")
                    mol_block = Chem.MolToMolBlock(mol)
                    view = py3Dmol.view(width=400, height=250)
                    view.addModel(mol_block, 'mol')
                    view.setStyle({'stick': {}})
                    view.zoomTo()
                    showmol(view, height=250, width=400)

                st.markdown("---")

                # --- DYNAMIC ANALYSIS MODULES ---

                # MODULE 1: TOXICITY (AI)
                if run_tox:
                    st.subheader("1️⃣ Toxicity Profile (AI Prediction)")
                    f = featurizer.featurize([smiles])
                    prob = model.predict_proba(f)[0][1]
                    risk_score = prob * 100

                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric("Toxicity Risk", f"{risk_score:.1f}%")
                    with c2:
                        if risk_score < 40:
                            st.success("✅ **Predicted SAFE**\n\nSuitable for further lab testing.")
                        elif risk_score < 70:
                            st.warning("⚠️ **Moderate Risk**\n\nCheck functional groups.")
                        else:
                            st.error("☠️ **High Toxicity Risk**\n\nNot recommended.")

                # MODULE 2: ADME & LIPINSKI
                if run_adme:
                    st.markdown("---")
                    st.subheader("2️⃣ ADME & Bioavailability (Lipinski's Rule)")
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mol. Weight", f"{mw:.1f}", "Target: <500")
                    c2.metric("LogP (Lipophilicity)", f"{logp:.1f}", "Target: <5")
                    c3.metric("H-Donors", hbd, "Target: <5")
                    c4.metric("H-Acceptors", hba, "Target: <10")

                    violations = 0
                    if mw > 500: violations += 1
                    if logp > 5: violations += 1
                    if hbd > 5: violations += 1
                    if hba > 10: violations += 1

                    if violations == 0:
                        st.success("✅ **Passes Lipinski's Rule of 5** (Good Oral Absorption)")
                    else:
                        st.warning(f"⚠️ **{violations} Violation(s) Detected** (Absorption issues possible)")

                # MODULE 3: QED SCORE
                if run_qed:
                    st.markdown("---")
                    st.subheader("3️⃣ QED Drug-Likeness Score")
                    qed_score = QED.qed(mol)
                    st.progress(qed_score)
                    st.caption(f"QED Score: **{qed_score:.2f}** (0 = Poor, 1 = Ideal Drug)")

                    if qed_score > 0.6:
                        st.success("🌟 Excellent Drug-Like Properties")
                    else:
                        st.info("ℹ️ Low Drug-Likeness (Common for Natural Products)")

                # MODULE 4: RADAR PLOT
                if run_radar:
                    st.markdown("---")
                    st.subheader("4️⃣ Bioactivity Radar Chart")
                    # Normalized approximate values for the chart
                    radar_data = {
                        "Size (MW)": min(mw / 500, 1.0),
                        "Polarity": min(Descriptors.TPSA(mol) / 140, 1.0),
                        "Insolubility": min(logp / 5, 1.0) if logp > 0 else 0,
                        "Flexibility": min(Descriptors.NumRotatableBonds(mol) / 10, 1.0),
                        "Saturation": Descriptors.FractionCSP3(mol)
                    }

                    fig = draw_radar_chart(radar_data, "Molecular Property Landscape")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.pyplot(fig)
                    with c2:
                        st.info(
                            "💡 **How to read:**\nA good drug candidate typically stays within the green shaded area (balanced properties).")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")

# === MODE 2: BATCH SCREENING ===
elif mode == "Batch Screening (CSV)":
    st.title("📂 Bulk Research Screening")
    st.write("Upload a CSV file with a column named `SMILES`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            st.write(f"Loaded {len(df)} molecules.")

            if st.button("🚀 Run Batch Analysis"):
                results = []
                progress_bar = st.progress(0)

                for i, row in df.iterrows():
                    try:
                        smiles = row['SMILES']
                        mol = Chem.MolFromSmiles(smiles)
                        res = {"SMILES": smiles}

                        # Apply Selected Modules
                        if run_tox:
                            f = featurizer.featurize([smiles])
                            res["Toxicity_Prob"] = round(model.predict_proba(f)[0][1], 3)

                        if run_adme:
                            res["MW"] = round(Descriptors.MolWt(mol), 2)
                            res["LogP"] = round(Descriptors.MolLogP(mol), 2)
                            res["Lipinski_Violations"] = 0
                            if res["MW"] > 500: res["Lipinski_Violations"] += 1
                            if res["LogP"] > 5: res["Lipinski_Violations"] += 1

                        if run_qed:
                            res["QED_Score"] = round(QED.qed(mol), 3)

                        results.append(res)
                    except:
                        results.append({"SMILES": row['SMILES'], "Error": "Invalid Structure"})

                    progress_bar.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)

                # Color code toxicity if present
                if run_tox:
                    st.subheader("🏆 Ranked Candidates (Safest First)")
                    results_df = results_df.sort_values(by="Toxicity_Prob")
                else:
                    st.subheader("🏆 Analysis Results")

                st.dataframe(results_df)
                st.download_button("📥 Download Research Data", results_df.to_csv(index=False), "AyurSafe_Results.csv")