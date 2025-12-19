import streamlit as st
import deepchem as dc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED, GraphDescriptors
from stmol import showmol
import py3Dmol

# --- CONFIGURATION ---
st.set_page_config(page_title="AyurSafe AI Research Platform", page_icon="🧬", layout="wide")

# --- CUSTOM CSS (Professional UI) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
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
    st.error("⚠️ Model files not found. Ensure 'toxicity_model.pkl' is in the repo.")


# --- HELPER FUNCTIONS ---

def check_pains(mol):
    """Checks for Pan-Assay Interference Compounds (Fake Hits)."""
    # Simplified PAINS SMARTS patterns (Quinones, Catechols, etc.)
    pains_smarts = [
        "O=C1C=CC(=O)C=C1",  # Quinone
        "c1ccc(O)c(O)c1",  # Catechol
        "C=C(C=O)C=O"  # Michael Acceptor
    ]
    for smarts in pains_smarts:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return True, "Contains PAINS Substructure (Possible False Positive)"
    return False, "Passes PAINS Filter"


def draw_radar_chart(features, title):
    labels = list(features.keys())
    values = list(features.values())
    values += values[:1]  # Close loop
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close loop

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#2E7D32', alpha=0.3)
    ax.plot(angles, values, color='#2E7D32', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, weight='bold')
    ax.set_title(title, y=1.1, fontsize=11, weight='bold', color="#2E7D32")
    return fig


# --- SIDEBAR ---
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-ayurveda-logo-template_23-2149405626.jpg", width=120)
st.sidebar.title("AyurSafe AI 🧬")
st.sidebar.markdown("**Research-Grade Discovery Platform**")
st.sidebar.caption("v2.0 | Conference Edition")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Select Workflow:", ["Single Molecule Lab", "Batch Screening (CSV)"])

st.sidebar.markdown("### ⚙️ Analysis Modules")
run_tox = st.sidebar.checkbox("Toxicity Prediction (AI)", value=True)
run_adme = st.sidebar.checkbox("ADME & Bioavailability", value=True)
run_advanced = st.sidebar.checkbox("Advanced (PAINS + Complexity)", value=False)
run_radar = st.sidebar.checkbox("Generate Radar Plot", value=False)

st.sidebar.markdown("---")
st.sidebar.info("🎓 **Citation:** \nAyurSafe AI: GNN-Enhanced Phytochemical Screening Tool (2025).")

# --- MAIN APP ---

if mode == "Single Molecule Lab":
    st.title("🧪 In-Silico Drug Discovery Lab")
    st.markdown("Analyze molecular candidates for Safety, Drug-Likeness, and Manufacturability.")

    col_input, col_vis = st.columns([1, 1])

    with col_input:
        smiles = st.text_area("Enter SMILES String:", "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
                              height=100)
        analyze_btn = st.button("🚀 Run Full Analysis")

    if analyze_btn and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("Invalid SMILES.")
            else:
                with col_vis:
                    st.markdown("**3D Structure**")
                    mol_block = Chem.MolToMolBlock(mol)
                    view = py3Dmol.view(width=400, height=250)
                    view.addModel(mol_block, 'mol')
                    view.setStyle({'stick': {}})
                    view.zoomTo()
                    showmol(view, height=250, width=400)

                st.markdown("---")

                # MODULE 1: TOXICITY
                if run_tox:
                    st.subheader("1️⃣ Toxicity Profile (AI)")
                    f = featurizer.featurize([smiles])
                    prob = model.predict_proba(f)[0][1]
                    risk = prob * 100

                    c1, c2 = st.columns([1, 3])
                    c1.metric("Risk Score", f"{risk:.1f}%")
                    with c2:
                        if risk < 40:
                            st.success("✅ **Predicted SAFE**")
                        elif risk < 70:
                            st.warning("⚠️ **Moderate Risk**")
                        else:
                            st.error("☠️ **High Toxicity**")

                # MODULE 2: ADME
                if run_adme:
                    st.markdown("---")
                    st.subheader("2️⃣ ADME & Lipinski Rules")
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mol. Weight", f"{mw:.1f}", "<500")
                    c2.metric("LogP (Solubility)", f"{logp:.1f}", "<5")
                    c3.metric("QED Score", f"{QED.qed(mol):.2f}", "0-1")

                    if mw < 500 and logp < 5:
                        st.success("✅ **Passes Lipinski Rules** (Good Oral Drug Candidate)")
                    else:
                        st.warning("⚠️ **Lipinski Violation** (Absorption Issues)")

                # MODULE 3: ADVANCED (PAINS + COMPLEXITY)
                if run_advanced:
                    st.markdown("---")
                    st.subheader("3️⃣ Advanced Filters (Bioinformatics/Chem)")

                    # PAINS Check
                    is_pains, pains_msg = check_pains(mol)
                    if is_pains:
                        st.error(f"🚫 {pains_msg}")
                    else:
                        st.success(f"✅ {pains_msg}")

                    # Complexity (BertzCT)
                    complexity = GraphDescriptors.BertzCT(mol)
                    st.write(f"**Structural Complexity (BertzCT):** {complexity:.1f}")
                    if complexity < 800:
                        st.info("🔹 Low Complexity (Easier to Synthesize)")
                    else:
                        st.info("🔶 High Complexity (Difficult Synthesis / Natural Product)")

                # MODULE 4: RADAR PLOT
                if run_radar:
                    st.markdown("---")
                    st.subheader("4️⃣ Bioactivity Radar")
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)

                    data = {
                        "Size": min(mw / 500, 1.0),
                        "Polarity": min(Descriptors.TPSA(mol) / 140, 1.0),
                        "Insolubility": min(logp / 5, 1.0) if logp > 0 else 0,
                        "Flexibility": min(Descriptors.NumRotatableBonds(mol) / 10, 1.0),
                        "Saturation": Descriptors.FractionCSP3(mol)
                    }
                    fig = draw_radar_chart(data, "Molecular Property Map")
                    c1, c2 = st.columns([1, 2])
                    c1.pyplot(fig)
                    c2.info(
                        "This chart maps the 5 key properties of drug-likeness. Ideal candidates stay within the center-green zone.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")

elif mode == "Batch Screening (CSV)":
    st.title("📂 Bulk Research Screening")
    st.write("Upload CSV with `SMILES` column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if st.button("🚀 Run Batch Analysis"):
            res_list = []
            bar = st.progress(0)

            for i, row in df.iterrows():
                s = row.get('SMILES', '')
                try:
                    m = Chem.MolFromSmiles(s)
                    d = {"SMILES": s}

                    if run_tox:
                        f = featurizer.featurize([s])
                        d["Toxicity_Prob"] = round(model.predict_proba(f)[0][1], 3)

                    if run_adme:
                        d["MW"] = round(Descriptors.MolWt(m), 2)
                        d["LogP"] = round(Descriptors.MolLogP(m), 2)
                        d["QED"] = round(QED.qed(m), 3)

                    if run_advanced:
                        is_pains, _ = check_pains(m)
                        d["PAINS_Alert"] = is_pains
                        d["Complexity"] = round(GraphDescriptors.BertzCT(m), 1)

                    res_list.append(d)
                except:
                    res_list.append({"SMILES": s, "Error": "Invalid"})

                bar.progress((i + 1) / len(df))

            final_df = pd.DataFrame(res_list)
            if run_tox: final_df = final_df.sort_values("Toxicity_Prob")
            st.dataframe(final_df)
            st.download_button("📥 Download Research Data", final_df.to_csv(index=False), "AyurSafe_Final_Report.csv")