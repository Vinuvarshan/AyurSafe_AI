import deepchem as dc
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from rdkit import RDLogger

# --- STEP 0: SILENCE THE NOISE ---
# This stops the console from filling up with warnings about weird atoms
RDLogger.DisableLog('rdApp.*')

print("🚀 Starting Model Training Sequence on M4 Chip...")

# --- STEP 1: LOAD THE DATA ---
print("⬇️  Loading Tox21 Dataset (This might take 1-2 minutes)...")
# We use 'ECFP' (fingerprints) to turn molecules into numbers (0s and 1s)
# The 'try/except' is a safety net in case the download server blinks
try:
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
    train_dataset, valid_dataset, test_dataset = datasets
except Exception as e:
    print(f"❌ Download Error: {e}")
    exit()

print(f"✅ Data Loaded! Training on {len(train_dataset)} safe molecules.")

# --- STEP 2: PREPARE THE DATA ---
# The dataset has 12 different toxicity tasks.
# We will focus on Task 0: "Nuclear Receptor Signaling" (A common toxicity check)
print("⚙️  Extracting and cleaning data...")

# Extract the data (X) and the labels (y)
X_train = train_dataset.X
y_train = train_dataset.y[:, 0]  # Only look at the first disease/toxicity type
w_train = train_dataset.w[:, 0]  # Weights (used to ignore missing data)

X_valid = valid_dataset.X
y_valid = valid_dataset.y[:, 0]
w_valid = valid_dataset.w[:, 0]

# Filter out empty data (some molecules weren't tested for this specific disease)
# We only keep rows where weight > 0
ids_train = np.where(w_train > 0)[0]
X_train_clean = X_train[ids_train]
y_train_clean = y_train[ids_train]

ids_valid = np.where(w_valid > 0)[0]
X_valid_clean = X_valid[ids_valid]
y_valid_clean = y_valid[ids_valid]

print(f"   - Clean Training Samples: {len(X_train_clean)}")
print(f"   - Clean Validation Samples: {len(X_valid_clean)}")

# --- STEP 3: TRAIN THE BRAIN ---
print("\n🧠 Training the AI Model...")
# We use a Random Forest. It's like asking 100 experts (trees) to vote on toxicity.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# n_jobs=-1 tells the computer to use ALL your M4 cores at once.

model.fit(X_train_clean, y_train_clean)
print("✅ Training Complete.")

# --- STEP 4: EVALUATE ---
print("\n🧐 Testing Accuracy...")
# Ask the model to predict the validation set
y_pred_valid = model.predict(X_valid_clean)
y_prob_valid = model.predict_proba(X_valid_clean)[:, 1] # Get probability scores

# Calculate scores
acc = accuracy_score(y_valid_clean, y_pred_valid)
roc_auc = roc_auc_score(y_valid_clean, y_prob_valid)

print(f"🏆 FINAL SCORES:")
print(f"   - Accuracy: {acc:.2%} (How often it is right)")
print(f"   - ROC-AUC:  {roc_auc:.3f} (How good it is at ranking risk)")

# --- STEP 5: SAVE THE PRODUCT ---
joblib.dump(model, 'toxicity_model.pkl')
print("\n💾 Model saved as 'toxicity_model.pkl'.")
print("   Ready for Phase 3 (Building the App).")