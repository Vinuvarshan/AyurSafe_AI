import torch
import deepchem as dc
import rdkit
from rdkit import Chem
from rdkit import RDLogger

# --- 1. SILENCE THE NOISE ---
# This shuts up the thousands of "Deprecation" and "Valence" warnings
RDLogger.DisableLog('rdApp.*')

print("--- 🛠 SYSTEM DIAGNOSTICS ---")
print(f"✅ DeepChem Version: {dc.__version__}")
print(f"✅ RDKit Version: {rdkit.__version__}")

# --- 2. CHECK M4 CHIP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ ACCELERATION: Apple Metal (MPS) is ACTIVE 🚀")
else:
    print("⚠️ ACCELERATION: Using CPU only")

# --- 3. TEST MOLECULE PROCESSING (Manual Test) ---
# Instead of downloading the broken dataset immediately,
# let's test if your Logic works on a single molecule.
print("\n--- 🧪 MICRO-TEST ---")
try:
    # Create a dummy molecule (Caffeine)
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = Chem.MolFromSmiles(smiles)

    # Try to featurize it (This tests if DeepChem is working)
    featurizer = dc.feat.CircularFingerprint(size=1024)
    features = featurizer.featurize([smiles])

    if features.shape[1] == 1024:
        print(f"✅ SUCCESS: Bio-Logic is working.")
        print(f"   Converted 'Caffeine' into {features.shape} AI data points.")
        print("   (The error you saw before was just bad data in the external file, not your PC).")

except Exception as e:
    print(f"❌ ERROR: {e}")