"""
41 Subjects Modality Ablation + Component Ablation Experiment
Baseline: run_neuma_41subj_combo.py (84.06%)
Output: results/neuma_41subj_ablation_results.json
"""
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =====================================================================
# 0. Basic Configuration (fully copied from combo.py)
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "ablation.json"

SEED = 42
MAX_EPOCHS = 300
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
LR_PATIENCE = 10
ET_DIM = 14
BEH_DIM = 7
EEG_DIM = 46
NUM_CLASSES = 2
ASC_L2 = 5e-2

BASELINE_ACC = 0.8406   # Target baseline


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_device() -> torch.device:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:0")
        print("[Device] Warning: Only 1 GPU detected, falling back to cuda:0")
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = build_device()
set_seed(SEED)

print("=" * 88)
print("41 Subjects Modality Ablation + Component Ablation Experiment")
print("=" * 88)
print(f"[Device] {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")


# =====================================================================
# 1. Data Loading — 41 Subjects (Excluding S26) fully copied from combo.py
# =====================================================================
def load_data():
    print(f"\n[Data] Loading: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_eeg_c1_raw = data["X_eeg_c1"].astype(np.float32)
    X_et_raw = data["X_et"].astype(np.float32)
    X_beh_raw = data["X_beh"].astype(np.float32)
    y = data["y_binary"].astype(np.int64)
    subject_ids = np.asarray(data["subject_ids"])

    print(f"[Data] Total samples after loading: {len(y)}, subjects: {len(np.unique(subject_ids))}")

    # Exclude S26 — compatible with string 'S26' and integer 26
    def is_s26(sid):
        if isinstance(sid, (int, np.integer)):
            return int(sid) == 26
        return str(sid).strip().upper() == 'S26'

    mask_keep = np.array([not is_s26(s) for s in subject_ids])
    X_eeg_c1_raw = X_eeg_c1_raw[mask_keep]
    X_et_raw = X_et_raw[mask_keep]
    X_beh_raw = X_beh_raw[mask_keep]
    y = y[mask_keep]
    subject_ids = subject_ids[mask_keep]
    print(f"[Data] After excluding S26: samples={len(y)}, subjects={len(np.unique(subject_ids))}")

    # Fill NaN with 0
    X_eeg_c1_raw = np.nan_to_num(X_eeg_c1_raw, nan=0.0).astype(np.float32)
    X_et_raw = np.nan_to_num(X_et_raw, nan=0.0).astype(np.float32)
    X_beh_raw = np.nan_to_num(X_beh_raw, nan=0.0).astype(np.float32)
    X_eeg_c1_zero = np.nan_to_num(X_eeg_c1_raw, nan=0.0).astype(np.float32)
    X_et_zero = np.nan_to_num(X_et_raw, nan=0.0).astype(np.float32)
    X_beh_zero = np.nan_to_num(X_beh_raw, nan=0.0).astype(np.float32)

    print(f"[Data] X_et={X_et_zero.shape}, X_eeg_c1={X_eeg_c1_zero.shape}, X_beh={X_beh_zero.shape}")
    print(f"[Data] Labels: 0={int(np.sum(y == 0))}, 1={int(np.sum(y == 1))}")
    print(f"[Data] Subjects: {len(np.unique(subject_ids))}, samples: {len(y)}")

    return (X_et_zero, X_eeg_c1_zero, X_beh_zero, y, subject_ids,
            X_et_raw, X_eeg_c1_raw, X_beh_raw)


# =====================================================================
# 2. Model Definition (fully copied from combo.py)
# =====================================================================
class A1EEGBranch(nn.Module):
    def __init__(self, eeg_dim=EEG_DIM, p1=35, p2=5, p3=6,
                 h_power=24, h_tbr=8, h_asym=8, h_fuse=32, dropout=0.3):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.power_branch = nn.Sequential(nn.Linear(p1, h_power), nn.ReLU(), nn.BatchNorm1d(h_power))
        self.tbr_branch = nn.Sequential(nn.Linear(p2, h_tbr), nn.ReLU(), nn.BatchNorm1d(h_tbr))
        self.asym_branch = nn.Sequential(nn.Linear(p3, h_asym), nn.ReLU(), nn.BatchNorm1d(h_asym))
        self.fuse = nn.Sequential(nn.Linear(h_power + h_tbr + h_asym, h_fuse), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x_power = x[:, :self.p1]
        x_tbr = x[:, self.p1:self.p1 + self.p2]
        x_asym = x[:, self.p1 + self.p2:self.p1 + self.p2 + self.p3]
        h = torch.cat([
            self.power_branch(x_power),
            self.tbr_branch(x_tbr),
            self.asym_branch(x_asym),
        ], dim=1)
        return self.fuse(h)


class T0_Baseline(nn.Module):
    """Base model with configurable dropout (fully copied from combo.py)"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,
                 dropout_et=0.3, dropout_beh=0.2):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h), nn.ReLU(), nn.BatchNorm1d(et_h), nn.Dropout(dropout_et),
        )
        self.eeg_branch = A1EEGBranch(h_fuse=eeg_h)
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h), nn.ReLU(), nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.ctx = nn.Sequential(nn.Linear(total_h, 32), nn.ReLU())
        self.gate_et = nn.Linear(32, et_h)
        self.gate_eeg = nn.Linear(32, eeg_h)
        self.gate_beh = nn.Linear(32, beh_h)
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1), nn.ReLU(), nn.BatchNorm1d(fuse1), nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2), nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)
        self.asc_net = nn.Sequential(
            nn.Linear(11, 8), nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )

    def _gated_forward(self, h_et, h_eeg, h_beh):
        ctx_in = torch.cat([h_et, h_eeg, h_beh], dim=1)
        ctx = self.ctx(ctx_in)
        g_et = torch.sigmoid(self.gate_et(ctx))
        g_eeg = torch.sigmoid(self.gate_eeg(ctx))
        g_beh = torch.sigmoid(self.gate_beh(ctx))
        gated = torch.cat([g_et * h_et, g_eeg * h_eeg, g_beh * h_beh], dim=1)
        h = self.fusion(gated)
        return self.utility(h)

    def forward(self, x_et, x_eeg, x_beh):
        h_et = self.et_branch(x_et)
        h_eeg = self.eeg_branch(x_eeg)
        h_beh = self.beh_branch(x_beh)
        utility = self._gated_forward(h_et, h_eeg, h_beh)
        x_tbr = x_eeg[:, 35:40]
        x_asym = x_eeg[:, 40:46]
        asc = self.asc_net(torch.cat([x_tbr, x_asym], dim=1))
        logits = utility + asc
        return logits, {}

    def get_asc_params(self):
        return list(self.asc_net.parameters())

    def get_non_asc_params(self):
        asc_ids = {id(p) for p in self.asc_net.parameters()}
        return [p for p in self.parameters() if id(p) not in asc_ids]


# =====================================================================
# 3. Component Ablation Model Variants
# =====================================================================

class NoGate_Model(nn.Module):
    """C1: No Gating — directly concat(h_et, h_eeg, h_beh) into fusion"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,
                 dropout_et=0.3, dropout_beh=0.2):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h), nn.ReLU(), nn.BatchNorm1d(et_h), nn.Dropout(dropout_et),
        )
        self.eeg_branch = A1EEGBranch(h_fuse=eeg_h)
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h), nn.ReLU(), nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1), nn.ReLU(), nn.BatchNorm1d(fuse1), nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2), nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)
        self.asc_net = nn.Sequential(
            nn.Linear(11, 8), nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )

    def forward(self, x_et, x_eeg, x_beh):
        h_et = self.et_branch(x_et)
        h_eeg = self.eeg_branch(x_eeg)
        h_beh = self.beh_branch(x_beh)
        # Directly concat without gate
        concat = torch.cat([h_et, h_eeg, h_beh], dim=1)
        h = self.fusion(concat)
        utility = self.utility(h)
        x_tbr = x_eeg[:, 35:40]
        x_asym = x_eeg[:, 40:46]
        asc = self.asc_net(torch.cat([x_tbr, x_asym], dim=1))
        logits = utility + asc
        return logits, {}

    def get_asc_params(self):
        return list(self.asc_net.parameters())

    def get_non_asc_params(self):
        asc_ids = {id(p) for p in self.asc_net.parameters()}
        return [p for p in self.parameters() if id(p) not in asc_ids]


class NoASC_Model(nn.Module):
    """C2: No ASC — logits = utility (without adding ASC output)"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,
                 dropout_et=0.3, dropout_beh=0.2):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h), nn.ReLU(), nn.BatchNorm1d(et_h), nn.Dropout(dropout_et),
        )
        self.eeg_branch = A1EEGBranch(h_fuse=eeg_h)
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h), nn.ReLU(), nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.ctx = nn.Sequential(nn.Linear(total_h, 32), nn.ReLU())
        self.gate_et = nn.Linear(32, et_h)
        self.gate_eeg = nn.Linear(32, eeg_h)
        self.gate_beh = nn.Linear(32, beh_h)
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1), nn.ReLU(), nn.BatchNorm1d(fuse1), nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2), nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)
        # Keep asc_net structure but don't use its output
        self.asc_net = nn.Sequential(
            nn.Linear(11, 8), nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )

    def _gated_forward(self, h_et, h_eeg, h_beh):
        ctx_in = torch.cat([h_et, h_eeg, h_beh], dim=1)
        ctx = self.ctx(ctx_in)
        g_et = torch.sigmoid(self.gate_et(ctx))
        g_eeg = torch.sigmoid(self.gate_eeg(ctx))
        g_beh = torch.sigmoid(self.gate_beh(ctx))
        gated = torch.cat([g_et * h_et, g_eeg * h_eeg, g_beh * h_beh], dim=1)
        h = self.fusion(gated)
        return self.utility(h)

    def forward(self, x_et, x_eeg, x_beh):
        h_et = self.et_branch(x_et)
        h_eeg = self.eeg_branch(x_eeg)
        h_beh = self.beh_branch(x_beh)
        utility = self._gated_forward(h_et, h_eeg, h_beh)
        # No asc, output utility directly
        logits = utility
        return logits, {}

    def get_asc_params(self):
        return list(self.asc_net.parameters())

    def get_non_asc_params(self):
        asc_ids = {id(p) for p in self.asc_net.parameters()}
        return [p for p in self.parameters() if id(p) not in asc_ids]


class NoEEGSub_Model(nn.Module):
    """C3: No EEG Sub-branch — replace A1EEGBranch with single-layer MLP"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,
                 dropout_et=0.3, dropout_beh=0.2):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h), nn.ReLU(), nn.BatchNorm1d(et_h), nn.Dropout(dropout_et),
        )
        # Replace A1EEGBranch with single-layer MLP
        self.eeg_branch = nn.Sequential(
            nn.Linear(EEG_DIM, eeg_h), nn.ReLU(), nn.Dropout(0.3)
        )
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h), nn.ReLU(), nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.ctx = nn.Sequential(nn.Linear(total_h, 32), nn.ReLU())
        self.gate_et = nn.Linear(32, et_h)
        self.gate_eeg = nn.Linear(32, eeg_h)
        self.gate_beh = nn.Linear(32, beh_h)
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1), nn.ReLU(), nn.BatchNorm1d(fuse1), nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2), nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)
        self.asc_net = nn.Sequential(
            nn.Linear(11, 8), nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )

    def _gated_forward(self, h_et, h_eeg, h_beh):
        ctx_in = torch.cat([h_et, h_eeg, h_beh], dim=1)
        ctx = self.ctx(ctx_in)
        g_et = torch.sigmoid(self.gate_et(ctx))
        g_eeg = torch.sigmoid(self.gate_eeg(ctx))
        g_beh = torch.sigmoid(self.gate_beh(ctx))
        gated = torch.cat([g_et * h_et, g_eeg * h_eeg, g_beh * h_beh], dim=1)
        h = self.fusion(gated)
        return self.utility(h)

    def forward(self, x_et, x_eeg, x_beh):
        h_et = self.et_branch(x_et)
        h_eeg = self.eeg_branch(x_eeg)
        h_beh = self.beh_branch(x_beh)
        utility = self._gated_forward(h_et, h_eeg, h_beh)
        x_tbr = x_eeg[:, 35:40]
        x_asym = x_eeg[:, 40:46]
        asc = self.asc_net(torch.cat([x_tbr, x_asym], dim=1))
        logits = utility + asc
        return logits, {}

    def get_asc_params(self):
        return list(self.asc_net.parameters())

    def get_non_asc_params(self):
        asc_ids = {id(p) for p in self.asc_net.parameters()}
        return [p for p in self.parameters() if id(p) not in asc_ids]


class NoGateNoASC_Model(nn.Module):
    """C4: No Gate + No ASC"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,
                 dropout_et=0.3, dropout_beh=0.2):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h), nn.ReLU(), nn.BatchNorm1d(et_h), nn.Dropout(dropout_et),
        )
        self.eeg_branch = A1EEGBranch(h_fuse=eeg_h)
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h), nn.ReLU(), nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1), nn.ReLU(), nn.BatchNorm1d(fuse1), nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2), nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)

    def forward(self, x_et, x_eeg, x_beh):
        h_et = self.et_branch(x_et)
        h_eeg = self.eeg_branch(x_eeg)
        h_beh = self.beh_branch(x_beh)
        concat = torch.cat([h_et, h_eeg, h_beh], dim=1)
        h = self.fusion(concat)
        logits = self.utility(h)
        return logits, {}


# =====================================================================
# 4. Data Augmentation Utilities (fully copied from combo.py)
# =====================================================================
def offline_gaussian_augment(X_et, X_eeg, X_beh, y, n_aug, seed, noise_std=0.03):
    rng = np.random.RandomState(seed)
    et_std = np.std(X_et, axis=0, keepdims=True)
    eeg_std = np.std(X_eeg, axis=0, keepdims=True)
    beh_std = np.std(X_beh, axis=0, keepdims=True)

    aug_et = [X_et]
    aug_eeg = [X_eeg]
    aug_beh = [X_beh]
    aug_y = [y]

    for _ in range(n_aug):
        aug_et.append((X_et + rng.normal(0.0, noise_std * np.maximum(et_std, 1e-6), size=X_et.shape)).astype(np.float32))
        aug_eeg.append((X_eeg + rng.normal(0.0, noise_std * np.maximum(eeg_std, 1e-6), size=X_eeg.shape)).astype(np.float32))
        aug_beh.append((X_beh + rng.normal(0.0, noise_std * np.maximum(beh_std, 1e-6), size=X_beh.shape)).astype(np.float32))
        aug_y.append(y.copy())

    return (
        np.concatenate(aug_et, axis=0),
        np.concatenate(aug_eeg, axis=0),
        np.concatenate(aug_beh, axis=0),
        np.concatenate(aug_y, axis=0),
    )


# =====================================================================
# 5. Training Utilities (fully copied from combo.py)
# =====================================================================
def compute_class_weights(y_arr):
    counts = np.bincount(y_arr, minlength=NUM_CLASSES)
    weights = len(y_arr) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(model, train_loader, val_tensors, y_tr, name,
                lr=LR, patience=PATIENCE, weight_decay=WEIGHT_DECAY,
                label_smoothing=0.0):
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(
            weight=compute_class_weights(y_tr),
            label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))

    if hasattr(model, 'get_non_asc_params') and hasattr(model, 'get_asc_params'):
        optimizer = optim.Adam([
            {'params': model.get_non_asc_params(), 'weight_decay': weight_decay},
            {'params': model.get_asc_params(), 'weight_decay': ASC_L2},
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    val_inputs = [t.to(DEVICE) for t in val_tensors[:-1]]
    val_y = val_tensors[-1].to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = [b.to(DEVICE) for b in batch[:-1]]
            batch_y = batch[-1].to(DEVICE)
            logits, _ = model(*inputs)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(*val_inputs)
            val_loss_fn = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))
            val_loss = val_loss_fn(val_logits, val_y).item()
        scheduler.step(val_loss)

        if epoch % 100 == 0:
            print(f"      [{name}] Epoch {epoch:3d} | train={np.mean(epoch_losses):.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"      [{name}] EarlyStopping @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


# =====================================================================
# 6. General Fold-Running Framework (Modality Ablation Version)
# =====================================================================
def run_modality_ablation_fold(
        exp_key, exp_name,
        X_et_full, X_eeg_full, X_beh_full, y, subject_ids,
        zero_et=False, zero_eeg=False, zero_beh=False,
        n_aug=2, noise_std=0.03
):
    """
    Modality ablation: after standardization and before augmentation,
    replace the corresponding modality data with all zeros.
    Keep model structure completely unchanged (always uses T0_Baseline).
    """
    print(f"\n{'=' * 75}")
    print(f"[{exp_key}] {exp_name}")
    print("=" * 75)
    if zero_et:
        print(f"  >> ET zeroed")
    if zero_eeg:
        print(f"  >> EEG zeroed")
    if zero_beh:
        print(f"  >> BEH zeroed")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et_full, y), start=1):
        # Standardize
        sc_et = StandardScaler()
        sc_eeg = StandardScaler()
        sc_beh = StandardScaler()

        et_tr = sc_et.fit_transform(X_et_full[train_idx]).astype(np.float32)
        eeg_tr = sc_eeg.fit_transform(X_eeg_full[train_idx]).astype(np.float32)
        beh_tr = sc_beh.fit_transform(X_beh_full[train_idx]).astype(np.float32)
        y_tr = y[train_idx]

        et_te = sc_et.transform(X_et_full[test_idx]).astype(np.float32)
        eeg_te = sc_eeg.transform(X_eeg_full[test_idx]).astype(np.float32)
        beh_te = sc_beh.transform(X_beh_full[test_idx]).astype(np.float32)
        y_te = y[test_idx]

        # ---- Modality zeroing (after standardization) ----
        if zero_et:
            et_tr = np.zeros_like(et_tr)
            et_te = np.zeros_like(et_te)
        if zero_eeg:
            eeg_tr = np.zeros_like(eeg_tr)
            eeg_te = np.zeros_like(eeg_te)
        if zero_beh:
            beh_tr = np.zeros_like(beh_tr)
            beh_te = np.zeros_like(beh_te)

        # Data augmentation
        et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
            et_tr, eeg_tr, beh_tr, y_tr,
            n_aug=n_aug,
            seed=SEED + fold_i * 100 + 17,
            noise_std=noise_std
        )

        # Split validation set
        rng_val = np.random.RandomState(SEED + fold_i)
        perm = rng_val.permutation(len(y_tr))
        n_val = max(1, len(y_tr) // 5)
        val_local = perm[:n_val]
        train_local = perm[n_val:]

        et_val_ = et_tr[val_local]
        eeg_val_ = eeg_tr[val_local]
        beh_val_ = beh_tr[val_local]
        y_val_ = y_tr[val_local]

        et_tr_ = et_tr[train_local]
        eeg_tr_ = eeg_tr[train_local]
        beh_tr_ = beh_tr[train_local]
        y_tr_ = y_tr[train_local]

        # Build DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED + fold_i)

        train_ds = TensorDataset(
            torch.tensor(et_tr_),
            torch.tensor(eeg_tr_),
            torch.tensor(beh_tr_),
            torch.tensor(y_tr_, dtype=torch.long),
        )
        drop_last = (len(train_ds) % BATCH_SIZE == 1)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            drop_last=drop_last, generator=generator
        )

        val_tensors = [
            torch.tensor(et_val_),
            torch.tensor(eeg_val_),
            torch.tensor(beh_val_),
            torch.tensor(y_val_, dtype=torch.long),
        ]

        test_tensors = [
            torch.tensor(et_te, device=DEVICE),
            torch.tensor(eeg_te, device=DEVICE),
            torch.tensor(beh_te, device=DEVICE),
        ]

        set_seed(SEED + fold_i)
        model = T0_Baseline(dropout_fuse=0.5).to(DEVICE)

        model = train_model(
            model, train_loader, val_tensors, y_tr_,
            name=f"{exp_key}-F{fold_i}",
            lr=LR, patience=PATIENCE,
            label_smoothing=0.0,
        )

        model.eval()
        with torch.no_grad():
            logits, _ = model(*test_tensors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_te, preds)
        fold_accs.append(float(acc))
        print(f"    [{exp_key} F{fold_i:02d}] Acc={acc:.4f}  (train={len(y_tr_)}, val={len(y_val_)}, test={len(y_te)})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"  [{exp_key}] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    return {
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "name": exp_name,
        "config": {
            "zero_et": zero_et, "zero_eeg": zero_eeg, "zero_beh": zero_beh,
            "n_aug": n_aug, "noise_std": noise_std,
        }
    }


# =====================================================================
# 7. Component Ablation Running Framework
# =====================================================================
def run_component_ablation_fold(
        exp_key, exp_name, model_class,
        X_et_full, X_eeg_full, X_beh_full, y, subject_ids,
        n_aug=2, noise_std=0.03
):
    """
    Component ablation: use different model variants with the same data pipeline as baseline.
    n_aug=0 skips Gaussian augmentation (C5).
    """
    print(f"\n{'=' * 75}")
    print(f"[{exp_key}] {exp_name}")
    print("=" * 75)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et_full, y), start=1):
        # Standardize
        sc_et = StandardScaler()
        sc_eeg = StandardScaler()
        sc_beh = StandardScaler()

        et_tr = sc_et.fit_transform(X_et_full[train_idx]).astype(np.float32)
        eeg_tr = sc_eeg.fit_transform(X_eeg_full[train_idx]).astype(np.float32)
        beh_tr = sc_beh.fit_transform(X_beh_full[train_idx]).astype(np.float32)
        y_tr = y[train_idx]

        et_te = sc_et.transform(X_et_full[test_idx]).astype(np.float32)
        eeg_te = sc_eeg.transform(X_eeg_full[test_idx]).astype(np.float32)
        beh_te = sc_beh.transform(X_beh_full[test_idx]).astype(np.float32)
        y_te = y[test_idx]

        # Data augmentation (skip when n_aug=0)
        if n_aug > 0:
            et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
                et_tr, eeg_tr, beh_tr, y_tr,
                n_aug=n_aug,
                seed=SEED + fold_i * 100 + 17,
                noise_std=noise_std
            )

        # Split validation set
        rng_val = np.random.RandomState(SEED + fold_i)
        perm = rng_val.permutation(len(y_tr))
        n_val = max(1, len(y_tr) // 5)
        val_local = perm[:n_val]
        train_local = perm[n_val:]

        et_val_ = et_tr[val_local]
        eeg_val_ = eeg_tr[val_local]
        beh_val_ = beh_tr[val_local]
        y_val_ = y_tr[val_local]

        et_tr_ = et_tr[train_local]
        eeg_tr_ = eeg_tr[train_local]
        beh_tr_ = beh_tr[train_local]
        y_tr_ = y_tr[train_local]

        # Build DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED + fold_i)

        train_ds = TensorDataset(
            torch.tensor(et_tr_),
            torch.tensor(eeg_tr_),
            torch.tensor(beh_tr_),
            torch.tensor(y_tr_, dtype=torch.long),
        )
        drop_last = (len(train_ds) % BATCH_SIZE == 1)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            drop_last=drop_last, generator=generator
        )

        val_tensors = [
            torch.tensor(et_val_),
            torch.tensor(eeg_val_),
            torch.tensor(beh_val_),
            torch.tensor(y_val_, dtype=torch.long),
        ]

        test_tensors = [
            torch.tensor(et_te, device=DEVICE),
            torch.tensor(eeg_te, device=DEVICE),
            torch.tensor(beh_te, device=DEVICE),
        ]

        set_seed(SEED + fold_i)
        model = model_class(dropout_fuse=0.5).to(DEVICE)

        model = train_model(
            model, train_loader, val_tensors, y_tr_,
            name=f"{exp_key}-F{fold_i}",
            lr=LR, patience=PATIENCE,
            label_smoothing=0.0,
        )

        model.eval()
        with torch.no_grad():
            logits, _ = model(*test_tensors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_te, preds)
        fold_accs.append(float(acc))
        print(f"    [{exp_key} F{fold_i:02d}] Acc={acc:.4f}  (train={len(y_tr_)}, val={len(y_val_)}, test={len(y_te)})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"  [{exp_key}] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    return {
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "name": exp_name,
        "config": {
            "model_class": model_class.__name__,
            "n_aug": n_aug, "noise_std": noise_std,
        }
    }


# =====================================================================
# 8. Incremental Save Utility
# =====================================================================
def save_results(all_results):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  [Save] {RESULT_PATH}")


# =====================================================================
# 9. Main Function
# =====================================================================
def main():
    (X_et, X_eeg, X_beh, y, subject_ids,
     X_et_raw, X_eeg_raw, X_beh_raw) = load_data()

    all_results = {}

    # ==================================================================
    # Part 1: Modality Ablation
    # ==================================================================
    print("\n" + "=" * 88)
    print("Part 1: Modality Ablation Experiment")
    print("=" * 88)

    modality_cfgs = [
        # key, name, zero_et, zero_eeg, zero_beh
        ("A0", "A0 Full Baseline ET+EEG+BEH",           False, False, False),
        ("A1", "A1 ET only (EEG=0, BEH=0)",         False, True,  True),
        ("A2", "A2 EEG only (ET=0, BEH=0)",          True,  False, True),
        ("A3", "A3 BEH only (ET=0, EEG=0)",          True,  True,  False),
        ("A4", "A4 ET+EEG (BEH=0)",                  False, False, True),
        ("A5", "A5 ET+BEH (EEG=0)",                  False, True,  False),
        ("A6", "A6 EEG+BEH (ET=0)",                  True,  False, False),
    ]

    for exp_key, exp_name, zero_et, zero_eeg, zero_beh in modality_cfgs:
        if exp_key in all_results:
            print(f"\n[{exp_key}] Result already exists, skipping: {all_results[exp_key]['acc_mean']*100:.2f}%")
            continue
        set_seed(SEED)
        r = run_modality_ablation_fold(
            exp_key, exp_name,
            X_et, X_eeg, X_beh, y, subject_ids,
            zero_et=zero_et, zero_eeg=zero_eeg, zero_beh=zero_beh,
            n_aug=2, noise_std=0.03
        )
        all_results[exp_key] = r
        save_results(all_results)

        # Baseline alignment check
        if exp_key == "A0":
            a0_acc = r["acc_mean"]
            diff = abs(a0_acc - BASELINE_ACC)
            print(f"\n[Baseline Check] A0={a0_acc*100:.4f}% target={BASELINE_ACC*100:.2f}% diff={diff*100:.4f}%")
            if diff > 0.01:
                print(f"[Error] A0 baseline {a0_acc*100:.2f}% deviates from target {BASELINE_ACC*100:.2f}% by more than 0.01%, stopping experiment!")
                print(f"[Hint] Please check data loading, random seed or model config consistency with combo.py.")
                return
            else:
                print(f"[Pass] A0 baseline aligned! Continuing ablation experiments...\n")

    # Part 1 summary
    print("\n" + "=" * 65)
    print("=== Part 1: Modality Ablation Ranking ===")
    for key, _, _, _, _ in modality_cfgs:
        if key in all_results:
            r = all_results[key]
            print(f"  {key}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']})")
    print("=" * 65)

    # ==================================================================
    # Part 2: Component Ablation
    # ==================================================================
    print("\n" + "=" * 88)
    print("Part 2: Component Ablation Experiment")
    print("=" * 88)

    component_cfgs = [
        # key, name, model_class, n_aug, noise_std
        ("C0", "C0 Full Model Baseline",               T0_Baseline,       2, 0.03),
        ("C1", "C1 No Gate (NoGate)",                  NoGate_Model,      2, 0.03),
        ("C2", "C2 No ASC (NoASC)",                    NoASC_Model,       2, 0.03),
        ("C3", "C3 No EEG Sub-branch (NoEEGSub)",      NoEEGSub_Model,    2, 0.03),
        ("C4", "C4 No Gate+No ASC (NoGateNoASC)",      NoGateNoASC_Model, 2, 0.03),
        ("C5", "C5 No Data Augmentation (n_aug=0)",     T0_Baseline,       0, 0.03),
    ]

    for exp_key, exp_name, model_cls, n_aug_val, noise_val in component_cfgs:
        # C0 is the same as A0, directly reuse result (share A0's result)
        if exp_key == "C0":
            print(f"\n[{exp_key}] {exp_name} — Reusing A0 baseline result")
            all_results["C0"] = dict(all_results["A0"])
            all_results["C0"]["name"] = exp_name
            save_results(all_results)
            print(f"  [{exp_key}] Mean={all_results['C0']['acc_mean']*100:.2f}%")
            continue

        if exp_key in all_results:
            print(f"\n[{exp_key}] Result already exists, skipping: {all_results[exp_key]['acc_mean']*100:.2f}%")
            continue

        set_seed(SEED)
        r = run_component_ablation_fold(
            exp_key, exp_name, model_cls,
            X_et, X_eeg, X_beh, y, subject_ids,
            n_aug=n_aug_val, noise_std=noise_val
        )
        all_results[exp_key] = r
        save_results(all_results)

    # Part 2 summary
    print("\n" + "=" * 65)
    print("=== Part 2: Component Ablation Ranking ===")
    for key, name, _, _, _ in component_cfgs:
        if key in all_results:
            r = all_results[key]
            print(f"  {key}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']})")
    print("=" * 65)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 88)
    print("=== Ablation Study Summary ===")
    baseline = all_results.get("A0", {}).get("acc_mean", 0.0)
    print(f"\nBaseline A0: {baseline*100:.2f}%\n")

    print("--- Modality Ablation ---")
    for key, _, _, _, _ in modality_cfgs:
        if key in all_results and key != "A0":
            r = all_results[key]
            delta = r["acc_mean"] - baseline
            print(f"  {key}: {r['acc_mean']*100:.2f}%  ({delta*100:+.2f}%)  {r['name']}")

    print("\n--- Component Ablation ---")
    for key, _, _, _, _ in component_cfgs:
        if key in all_results and key != "C0":
            r = all_results[key]
            delta = r["acc_mean"] - baseline
            print(f"  {key}: {r['acc_mean']*100:.2f}%  ({delta*100:+.2f}%)  {r['name']}")
    print("=" * 88)

    # Final save
    save_results(all_results)
    print(f"\n[Done] Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
