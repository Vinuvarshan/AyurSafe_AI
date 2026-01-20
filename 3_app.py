import streamlit as st
import deepchem as dc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, GraphDescriptors, Lipinski, Crippen, rdMolDescriptors
from stmol import showmol
import py3Dmol
from fpdf import FPDF

# --- CONFIGURATION ---
st.set_page_config(page_title="AyurSafe AI Research Platform", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "current_user" not in st.session_state:
    st.session_state.current_user = "Guest"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None  # To store results for PDF

# --- 🟢 UNIQUE VISITOR TRACKING ---
query_params = st.query_params
if "visited" not in query_params:
    st.markdown(
        '<img src="https://visitor-badge.laobi.icu/badge?page_id=ayursafe_ai_project_vinu" width="0" height="0" style="display:none">',
        unsafe_allow_html=True
    )
    st.query_params["visited"] = "true"


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
    st.error("⚠️ Model files not found.")


# --- HELPER FUNCTIONS ---

# 1. SAFE PAINS FILTER (Prevents Server Crash)
def check_pains(mol):
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)

        if catalog.HasMatch(mol):
            entry = catalog.GetFirstMatch(mol)
            return True, f"ALERT: PAINS Structure Detected ({entry.GetDescription()})"
    except Exception as e:
        # Fails safely if RDKit database has issues
        return False, "PAINS Filter Unavailable (System Limit)"

    return False, "Passes PAINS Filter (Clean)"


def draw_radar_chart(features, title):
    labels = list(features.keys())
    values = list(features.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#2E7D32', alpha=0.3)
    ax.plot(angles, values, color='#2E7D32', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, weight='bold')
    ax.set_title(title, y=1.1, fontsize=11, weight='bold', color="#2E7D32")
    return fig


# 2. ADVANCED ADME CALCULATOR (SwissADME Replica)
def calculate_adme_properties(mol):
    if not mol: return None

    # Basic Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    # Advanced Descriptors
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mr = Crippen.MolMR(mol)
    csp3 = Lipinski.FractionCSP3(mol)
    qed_score = QED.qed(mol)

    # Rule Checks
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if h_donors > 5: lipinski_violations += 1
    if h_acceptors > 10: lipinski_violations += 1
    lipinski_status = "PASSED" if lipinski_violations == 0 else f"FAILED ({lipinski_violations} violations)"

    veber_violations = 0
    if rotatable_bonds > 10: veber_violations += 1
    if tpsa > 140: veber_violations += 1
    veber_status = "PASSED" if veber_violations == 0 else "FAILED"

    # PAINS Check
    is_pains, pains_msg = check_pains(mol)

    # BOILED-Egg Logic
    egg_status = "Low Absorption"
    if (tpsa <= 142) and (-2.3 <= logp <= 6.8):
        egg_status = "High GI Absorption (Intestine)"
    if (tpsa <= 79) and (0.4 <= logp <= 6.0):
        egg_status = "BBB Permeant (Brain / Yolk)"

    return {
        "Molecular Formula": formula,
        "Molecular Weight": f"{mw:.2f} g/mol",
        "Molar Refractivity": f"{mr:.2f}",
        "Fraction Csp3": f"{csp3:.2f}",
        "LogP (Lipophilicity)": f"{logp:.2f}",
        "H-Bond Donors": str(h_donors),
        "H-Bond Acceptors": str(h_acceptors),
        "Rotatable Bonds": str(rotatable_bonds),
        "TPSA": f"{tpsa:.2f} A^2",
        "QED Drug-Likeness": f"{qed_score:.3f} (0-1)",
        "Lipinski Rule": lipinski_status,
        "Veber Rule": veber_status,
        "PAINS Filter Check": pains_msg,
        "Bioavailability Model": egg_status
    }


# 3. PDF GENERATOR (3-Color Logic)
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'AyurSafe AI - Comprehensive Analysis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Automated Toxicity & Pharmacokinetic Screening Report', 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Generated by AyurSafe AI | Powered by Graph Neural Networks (GNN)', 0, 0, 'C')


def create_pdf(smiles, risk_score, adme_data):
    pdf = PDFReport()
    pdf.add_page()

    # Section 1: AI Result
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, "1. AI Toxicity Screening", 0, 1, fill=True)
    pdf.ln(2)

    pdf.set_font("Arial", size=12)
    if risk_score < 40:
        prediction = "Safe Candidate"
        pdf.set_text_color(0, 100, 0)  # Green
    elif risk_score < 70:
        prediction = "Moderate Risk (Bioactive)"
        pdf.set_text_color(204, 102, 0)  # Orange
    else:
        prediction = "Toxic / High Risk"
        pdf.set_text_color(200, 0, 0)  # Red

    pdf.cell(0, 8, f"Predicted Class: {prediction}", 0, 1)
    pdf.cell(0, 8, f"Toxicity Risk Score: {risk_score:.2f}%", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Section 2: Identification
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Chemical Identification", 0, 1, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, f"SMILES: {smiles}")
    pdf.cell(0, 8, f"Formula: {adme_data['Molecular Formula']}", 0, 1)
    pdf.ln(5)

    # Section 3: Physicochemical Properties
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Physicochemical Profile", 0, 1, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=11)

    phys_keys = ["Molecular Weight", "Molar Refractivity", "Fraction Csp3", "LogP (Lipophilicity)", "TPSA",
                 "Rotatable Bonds", "H-Bond Donors", "H-Bond Acceptors"]
    for i, key in enumerate(phys_keys):
        pdf.cell(90, 8, f"{key}: {adme_data[key]}", 0, 0)
        if (i + 1) % 2 == 0:
            pdf.ln(8)
    pdf.ln(10)

    # Section 4: Drug-Likeness & PAINS
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "4. Drug-Likeness & PAINS Filters", 0, 1, fill=True)
    pdf.ln(2)

    rules = {
        "Lipinski Rule (Pfizer)": adme_data["Lipinski Rule"],
        "Veber Rule (GSK)": adme_data["Veber Rule"],
        "PAINS Filter Check": adme_data["PAINS Filter Check"],
        "QED Score (Drug-Likeness)": adme_data["QED Drug-Likeness"],
        "Bioavailability Model": adme_data["Bioavailability Model"]
    }

    for key, value in rules.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(70, 8, f"{key}:", 0, 0)

        # Color Logic
        if "Moderate" in value or "Bioactive" in value:
            pdf.set_text_color(204, 102, 0)  # Orange
        elif "FAILED" in value or "ALERT" in value or "Toxic" in value or "Low Absorption" in value:
            pdf.set_text_color(180, 0, 0)  # Red
        elif "PASSED" in value or "Passes" in value or "Safe" in value or "High" in value or "Yolk" in value:
            pdf.set_text_color(0, 100, 0)  # Green
        else:
            pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, value, 0, 1)

    return pdf.output(dest='S').encode('latin-1')


# --- SIDEBAR ---
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-ayurveda-logo-template_23-2149405626.jpg", width=120)
st.sidebar.title("AyurSafe AI 🧬")
show_login = query_params.get("access") == "login"
st.sidebar.markdown("---")

if st.session_state.is_admin:
    st.sidebar.success(f"👤 **{st.session_state.current_user}**")
    if st.sidebar.button("Log Out"):
        st.session_state.is_admin = False
        st.session_state.current_user = "Guest"
        st.rerun()
elif show_login:
    with st.sidebar.expander("🔐 Admin Login", expanded=True):
        login_email = st.text_input("Email")
        login_pass = st.text_input("Password", type="password")
        if st.button("Log In"):
            # Simple credentials check
            if login_pass == "admin":
                st.session_state.is_admin = True
                st.session_state.current_user = login_email
                st.rerun()

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Workflow:", ["Single Molecule Lab", "Batch Screening (CSV)"])
run_tox = st.sidebar.checkbox("Toxicity Prediction (AI)", value=True)
run_adme = st.sidebar.checkbox("ADME & Bioavailability", value=True)
run_radar = st.sidebar.checkbox("Generate Radar Plot", value=False)
st.sidebar.markdown("---")

# --- MAIN APP LOGIC ---

if mode == "Single Molecule Lab":
    st.title("🧪 In-Silico Drug Discovery Lab")

    col_input, col_vis = st.columns([1, 1])
    with col_input:
        smiles = st.text_area("Enter SMILES String:", "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
                              height=100)
        analyze_btn = st.button("🚀 Run Full Analysis")

    # LOGIC: Run Analysis and Save to Session State
    if analyze_btn and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("Invalid SMILES.")
            else:
                # 1. Run Calculations
                risk_val = 0
                if run_tox:
                    f = featurizer.featurize([smiles])
                    risk_val = model.predict_proba(f)[0][1] * 100

                adme_res = calculate_adme_properties(mol)

                # 2. Store in Session State (Persistence)
                st.session_state.analysis_results = {
                    "smiles": smiles,
                    "risk": risk_val,
                    "adme": adme_res,
                    "mol_block": Chem.MolToMolBlock(mol)  # Store 3D structure data
                }
        except Exception as e:
            st.error(f"Error: {e}")

    # LOGIC: Display Results (If they exist in Session State)
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results

        # 3D View
        with col_vis:
            st.markdown("**3D Structure**")
            view = py3Dmol.view(width=400, height=250)
            view.addModel(res['mol_block'], 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            showmol(view, height=250, width=400)

        st.markdown("---")

        # Toxicity
        if run_tox:
            st.subheader("1️⃣ Toxicity Profile (AI)")
            c1, c2 = st.columns([1, 3])
            c1.metric("Risk Score", f"{res['risk']:.1f}%")
            if res['risk'] < 40:
                c2.success("✅ **Predicted SAFE**")
            elif res['risk'] < 70:
                c2.warning("⚠️ **Moderate Risk**")
            else:
                c2.error("☠️ **High Toxicity**")

        # ADME
        if run_adme and res['adme']:
            st.markdown("---")
            st.subheader("2️⃣ ADME & Lipinski Rules")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mol. Weight", res['adme']['Molecular Weight'])
            c2.metric("LogP", res['adme']['LogP (Lipophilicity)'])
            c3.metric("QED", res['adme']['QED Drug-Likeness'])

            # Show PAINS Alert if present
            if "ALERT" in res['adme']['PAINS Filter Check']:
                st.error(res['adme']['PAINS Filter Check'])
            else:
                st.success(res['adme']['PAINS Filter Check'])

            # Show Egg Status
            st.info(f"Bioavailability: {res['adme']['Bioavailability Model']}")

        # Radar
        if run_radar and res['adme']:
            st.markdown("---")
            st.subheader("4️⃣ Bioactivity Radar")
            # Parse floats back from string for chart
            mw_val = float(res['adme']['Molecular Weight'].split()[0])
            tpsa_val = float(res['adme']['TPSA'].split()[0])
            logp_val = float(res['adme']['LogP (Lipophilicity)'])

            data = {
                "Size": min(mw_val / 500, 1.0),
                "Polarity": min(tpsa_val / 140, 1.0),
                "Insolubility": min(logp_val / 5, 1.0) if logp_val > 0 else 0,
                "Flexibility": min(int(res['adme']['Rotatable Bonds']) / 10, 1.0),
                "Saturation": float(res['adme']['Fraction Csp3'])
            }
            c1, c2 = st.columns([1, 2])
            c1.pyplot(draw_radar_chart(data, "Property Map"))

        # --- THE PDF BUTTON (NOW OUTSIDE THE BUTTON BLOCK) ---
        st.markdown("---")
        if res['adme']:
            pdf_bytes = create_pdf(res['smiles'], res['risk'], res['adme'])
            st.download_button(
                label="📄 Download Full Lab Report (PDF)",
                data=pdf_bytes,
                file_name="AyurSafe_Lab_Report.pdf",
                mime="application/pdf"
            )

elif mode == "Batch Screening (CSV)":
    st.title("📂 Bulk Research Screening")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if st.button("🚀 Run Batch Analysis"):
            # (Batch logic remains same, just adding PAINS check)
            res_list = []
            bar = st.progress(0)
            for i, row in df.iterrows():
                s = row.get('SMILES', '')
                try:
                    m = Chem.MolFromSmiles(s)
                    adme = calculate_adme_properties(m)
                    f = featurizer.featurize([s])
                    risk = round(model.predict_proba(f)[0][1], 3)

                    row_data = {
                        "SMILES": s,
                        "Toxicity": risk,
                        "MW": adme['Molecular Weight'],
                        "PAINS": adme['PAINS Filter Check']
                    }
                    res_list.append(row_data)
                except:
                    res_list.append({"SMILES": s, "Error": "Invalid"})
                bar.progress((i + 1) / len(df))

            final_df = pd.DataFrame(res_list)
            st.dataframe(final_df)
            st.download_button("📥 Download Data", final_df.to_csv(index=False), "AyurSafe_Results.csv")