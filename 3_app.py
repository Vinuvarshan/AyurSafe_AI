import streamlit as st
import deepchem as dc
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from stmol import showmol
import py3Dmol

# --- CONFIGURATION ---
st.set_page_config(page_title="AyurSafe AI", page_icon="🌿", layout="wide")


# --- LOAD THE BRAIN (Cached for speed) ---
@st.cache_resource
def load_model():
    return joblib.load('toxicity_model.pkl')


@st.cache_resource
def get_featurizer():
    return dc.feat.CircularFingerprint(size=1024, radius=2)


model = load_model()
featurizer = get_featurizer()

# --- SIDEBAR (The Control Panel) ---
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-ayurveda-logo-template_23-2149405626.jpg", width=150)
st.sidebar.title("AyurSafe 🌿")
st.sidebar.write("**AI-Powered Toxicity Screening for Phytochemicals**")
st.sidebar.markdown("---")

# MODE SELECTION
mode = st.sidebar.radio("Select Mode:", ["Single Molecule Scanner", "Batch Upload (CSV)"])

# COMMERCIAL CONTACT INFO (New)
st.sidebar.markdown("---")
st.sidebar.info(
    "💼 **Commercial License**\n\nThe public demo is limited to small batches. For private, unlimited server deployment for your lab:\n\n[📩 **Contact Founder**](mailto:your.email@gmail.com)")

# --- MAIN PAGE: SINGLE MOLECULE ---
if mode == "Single Molecule Scanner":
    st.title("🧪 In-Silico Toxicity Predictor")
    st.write("Enter a chemical structure (SMILES format) to predict its safety.")

    # Input: Default is Curcumin (Turmeric)
    default_smiles = "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O"
    smiles_input = st.text_area("Paste SMILES String here:", default_smiles, height=70)

    if st.button("🚀 Analyze Molecule"):
        if not smiles_input:
            st.warning("Please enter a SMILES string.")
        else:
            try:
                # 1. VISUALIZE (2D & 3D)
                mol = Chem.MolFromSmiles(smiles_input)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("2D Structure")
                    st.image(Draw.MolToImage(mol), width=300)

                with col2:
                    st.subheader("3D Interactive Model")
                    mol_block = Chem.MolToMolBlock(mol)
                    view = py3Dmol.view(width=500, height=300)
                    view.addModel(mol_block, 'mol')
                    view.setStyle({'stick': {}})
                    view.zoomTo()
                    showmol(view, height=300, width=500)

                # 2. PREDICT
                features = featurizer.featurize([smiles_input])
                probability = model.predict_proba(features)[0][1]

                # 3. SHOW RESULTS
                st.markdown("---")
                st.subheader("📊 AI Analysis Report")

                risk_score = probability * 100

                if risk_score < 40:
                    st.success(f"✅ **SAFE** (Toxicity Risk: {risk_score:.2f}%)")
                    st.info("Recommendation: Proceed to Wet Lab testing.")
                elif risk_score < 70:
                    st.warning(f"⚠️ **CAUTION** (Toxicity Risk: {risk_score:.2f}%)")
                    st.write("Recommendation: Modify functional groups.")
                else:
                    st.error(f"☠️ **TOXIC** (Toxicity Risk: {risk_score:.2f}%)")
                    st.write("Recommendation: DISCARD immediately.")

            except Exception as e:
                st.error(f"Could not process molecule. Check SMILES format. Error: {e}")

# --- MAIN PAGE: BATCH MODE (Freemium Locked) ---
elif mode == "Batch Upload (CSV)":
    st.title("📂 Bulk Screening Pipeline")

    # WARNING MESSAGES
    st.warning("⚠️ Public Demo Limit: Maximum 5 molecules per file.")
    st.info("🔒 For bulk processing (unlimited) and privacy protection, contact us for an Enterprise License.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:

            # --- THE LOCK (Business Logic) ---
            if len(df) > 5:
                st.error(f"❌ Upload failed! Your file has {len(df)} molecules.")
                st.error("The Free Public Version is limited to 5 molecules.")
                st.markdown(
                    "To analyze this file, please [Contact Founder](mailto:your.email@gmail.com) for a Pro License.")

            else:
                # UNLOCKED: Run analysis
                st.write(f"Loaded {len(df)} molecules.")
                if st.button("Run AI Screening"):
                    progress_bar = st.progress(0)
                    results = []

                    for i, row in df.iterrows():
                        try:
                            f = featurizer.featurize([row['SMILES']])
                            prob = model.predict_proba(f)[0][1]
                            results.append(prob)
                        except:
                            results.append(None)  # Handle errors

                        progress_bar.progress((i + 1) / len(df))

                    df['Toxicity_Risk'] = results
                    df = df.sort_values(by='Toxicity_Risk', ascending=True)

                    st.write("### 🏆 Top Safest Candidates")
                    st.dataframe(df)

                    st.download_button("Download Full Report", df.to_csv(), "ai_report.csv")
        else:
            st.error("CSV must have a column named 'SMILES'")