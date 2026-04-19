"""
Traditional ML Baseline Comparison Experiment
B0: DCM-DNN (our full model, copied from combo)
B1: Logistic Regression
B2: SVM (RBF kernel)
B3: Simple MLP (67->64->32->2)
B4: Matched MLP (~13588 params, 67->128->64->32->2)
"""
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =====================================================================
# 0. Basic Configuration
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "baseline_compare.json"

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
FEAT_DIM = ET_DIM + EEG_DIM + BEH_DIM  # 67


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
            return torch.device("cuda:1")
        print("[Device] Warning: Only 1 GPU detected, falling back to cuda:0")
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = build_device()
set_seed(SEED)

print("=" * 88)
print("41 Subjects Traditional ML Baseline Comparison (B0=DCM-DNN, B1=LR, B2=SVM, B3=SimpleMLP, B4=MatchedMLP)")
print("=" * 88)
print(f"[Device] {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")


# =====================================================================
# 1. Data Loading — 41 Subjects (Excluding S26) — Strictly Copied from combo
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

    return (X_et_zero, X_eeg_c1_zero, X_beh_zero, y, subject_ids)


# =====================================================================
# 2. Model Definition — Strictly Copied from combo
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
    """Base model with configurable dropout — copied from combo"""
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# 3. B3: Simple MLP (67→64→ReLU→Dropout(0.5)→32→ReLU→2)
# =====================================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits, {}


# =====================================================================
# 4. B4: Matched MLP (~13588 params, 67→128→BN→Dropout(0.5)→64→Dropout(0.3)→32→2)
# =====================================================================
class MatchedMLP(nn.Module):
    def __init__(self, input_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits, {}


# =====================================================================
# 5. Data Augmentation — Strictly Copied from combo
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


def offline_gaussian_augment_flat(X_flat, y, n_aug, seed, noise_std=0.03):
    """Gaussian augmentation for concatenated feature vectors"""
    rng = np.random.RandomState(seed)
    x_std = np.std(X_flat, axis=0, keepdims=True)
    aug_x = [X_flat]
    aug_y = [y]
    for _ in range(n_aug):
        aug_x.append((X_flat + rng.normal(0.0, noise_std * np.maximum(x_std, 1e-6), size=X_flat.shape)).astype(np.float32))
        aug_y.append(y.copy())
    return np.concatenate(aug_x, axis=0), np.concatenate(aug_y, axis=0)


# =====================================================================
# 6. Training Utilities
# =====================================================================
def compute_class_weights(y_arr):
    counts = np.bincount(y_arr, minlength=NUM_CLASSES)
    weights = len(y_arr) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_dnn_model(model, train_loader, val_tensors, y_tr, name,
                    lr=LR, patience=PATIENCE, weight_decay=WEIGHT_DECAY):
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
# 7. B0: DCM-DNN (Strictly copied from combo's run_experiment logic)
# =====================================================================
def run_b0_dcm_dnn(X_et, X_eeg, X_beh, y):
    print(f"\n{'=' * 75}")
    print("[B0] DCM-DNN (full model, copied from combo, target≈84.06%)")
    print("=" * 75)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []
    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        sc_et = StandardScaler()
        sc_eeg = StandardScaler()
        sc_beh = StandardScaler()

        et_tr = sc_et.fit_transform(X_et[train_idx]).astype(np.float32)
        eeg_tr = sc_eeg.fit_transform(X_eeg[train_idx]).astype(np.float32)
        beh_tr = sc_beh.fit_transform(X_beh[train_idx]).astype(np.float32)
        y_tr = y[train_idx]

        et_te = sc_et.transform(X_et[test_idx]).astype(np.float32)
        eeg_te = sc_eeg.transform(X_eeg[test_idx]).astype(np.float32)
        beh_te = sc_beh.transform(X_beh[test_idx]).astype(np.float32)
        y_te = y[test_idx]

        # D3 Gaussian augmentation (2 copies, std=0.03)
        et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
            et_tr, eeg_tr, beh_tr, y_tr,
            n_aug=2, seed=SEED + fold_i * 100 + 17, noise_std=0.03
        )

        # Split validation set (randomly take 20% from training set)
        rng_val = np.random.RandomState(SEED + fold_i)
        perm = rng_val.permutation(len(y_tr))
        n_val = max(1, len(y_tr) // 5)
        val_local = perm[:n_val]
        train_local = perm[n_val:]

        et_val_ = et_tr[val_local]; eeg_val_ = eeg_tr[val_local]; beh_val_ = beh_tr[val_local]; y_val_ = y_tr[val_local]
        et_tr_ = et_tr[train_local]; eeg_tr_ = eeg_tr[train_local]; beh_tr_ = beh_tr[train_local]; y_tr_ = y_tr[train_local]

        generator = torch.Generator()
        generator.manual_seed(SEED + fold_i)

        train_ds = TensorDataset(
            torch.tensor(et_tr_), torch.tensor(eeg_tr_), torch.tensor(beh_tr_),
            torch.tensor(y_tr_, dtype=torch.long),
        )
        drop_last = (len(train_ds) % BATCH_SIZE == 1)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=drop_last, generator=generator)

        val_tensors = [torch.tensor(et_val_), torch.tensor(eeg_val_),
                       torch.tensor(beh_val_), torch.tensor(y_val_, dtype=torch.long)]
        test_tensors = [torch.tensor(et_te, device=DEVICE),
                        torch.tensor(eeg_te, device=DEVICE),
                        torch.tensor(beh_te, device=DEVICE)]

        set_seed(SEED + fold_i)
        model = T0_Baseline(dropout_fuse=0.5).to(DEVICE)
        model = train_dnn_model(model, train_loader, val_tensors, y_tr_,
                                name=f"B0-F{fold_i}")

        model.eval()
        with torch.no_grad():
            logits, _ = model(*test_tensors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_te, preds)
        fold_accs.append(float(acc))
        print(f"    [B0 F{fold_i:02d}] Acc={acc:.4f}  (train={len(y_tr_)}, val={len(y_val_)}, test={len(y_te)})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    n_params = count_params(T0_Baseline(dropout_fuse=0.5))
    print(f"  [B0] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%  params={n_params}  time={elapsed:.1f}s")
    if mean_acc >= 0.84:
        print(f"  [B0] OK: ≥84% ✓")
    else:
        print(f"  [B0] Warning: {mean_acc*100:.2f}% < 84%, please check")

    return {
        "name": "B0 DCM-DNN (full model)",
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "n_params": n_params,
        "train_time_s": round(elapsed, 2),
    }


# =====================================================================
# 8. B1-B2: Traditional ML Baselines
# =====================================================================
def run_ml_baseline(name_key, name_str, clf_factory,
                    X_et, X_eeg, X_beh, y):
    print(f"\n{'=' * 75}")
    print(f"[{name_key}] {name_str}")
    print("=" * 75)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []
    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        # Concatenate 67-dim features
        X_tr_raw = np.concatenate([X_et[train_idx], X_eeg[train_idx], X_beh[train_idx]], axis=1)
        X_te_raw = np.concatenate([X_et[test_idx], X_eeg[test_idx], X_beh[test_idx]], axis=1)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # StandardScaler fit on train
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_raw).astype(np.float32)
        X_te_sc = sc.transform(X_te_raw).astype(np.float32)

        # D3 Gaussian augmentation (2 copies, std=0.03) — fair comparison
        X_tr_aug, y_tr_aug = offline_gaussian_augment_flat(
            X_tr_sc, y_tr,
            n_aug=2, seed=SEED + fold_i * 100 + 17, noise_std=0.03
        )

        clf = clf_factory()
        clf.fit(X_tr_aug, y_tr_aug)
        preds = clf.predict(X_te_sc)
        acc = accuracy_score(y_te, preds)
        fold_accs.append(float(acc))
        print(f"    [{name_key} F{fold_i:02d}] Acc={acc:.4f}  (train={len(y_tr_aug)}, test={len(y_te)})")

    elapsed = time.time() - t_start
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"  [{name_key}] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%  time={elapsed:.1f}s")

    return {
        "name": name_str,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "n_params": None,
        "train_time_s": round(elapsed, 2),
    }


# =====================================================================
# 9. B3/B4: MLP Baselines (Unified Interface)
# =====================================================================
def run_mlp_baseline(name_key, name_str, model_factory,
                     X_et, X_eeg, X_beh, y):
    print(f"\n{'=' * 75}")
    print(f"[{name_key}] {name_str}")
    print("=" * 75)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []
    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        # Concatenate 67-dim features
        X_tr_raw = np.concatenate([X_et[train_idx], X_eeg[train_idx], X_beh[train_idx]], axis=1)
        X_te_raw = np.concatenate([X_et[test_idx], X_eeg[test_idx], X_beh[test_idx]], axis=1)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # StandardScaler
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_raw).astype(np.float32)
        X_te_sc = sc.transform(X_te_raw).astype(np.float32)

        # D3 Gaussian augmentation
        X_tr_aug, y_tr_aug = offline_gaussian_augment_flat(
            X_tr_sc, y_tr,
            n_aug=2, seed=SEED + fold_i * 100 + 17, noise_std=0.03
        )

        # Split validation set (randomly take 20% from training set)
        rng_val = np.random.RandomState(SEED + fold_i)
        perm = rng_val.permutation(len(y_tr_aug))
        n_val = max(1, len(y_tr_aug) // 5)
        val_local = perm[:n_val]
        train_local = perm[n_val:]

        X_val_ = X_tr_aug[val_local]; y_val_ = y_tr_aug[val_local]
        X_tr_ = X_tr_aug[train_local]; y_tr_ = y_tr_aug[train_local]

        generator = torch.Generator()
        generator.manual_seed(SEED + fold_i)

        train_ds = TensorDataset(
            torch.tensor(X_tr_), torch.tensor(y_tr_, dtype=torch.long)
        )
        drop_last = (len(train_ds) % BATCH_SIZE == 1)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=drop_last, generator=generator)

        val_tensors = [torch.tensor(X_val_), torch.tensor(y_val_, dtype=torch.long)]

        set_seed(SEED + fold_i)
        model = model_factory().to(DEVICE)
        model = train_dnn_model(model, train_loader, val_tensors, y_tr_,
                                name=f"{name_key}-F{fold_i}")

        model.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te_sc, device=DEVICE)
            logits, _ = model(X_te_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_te, preds)
        fold_accs.append(float(acc))
        print(f"    [{name_key} F{fold_i:02d}] Acc={acc:.4f}  (train={len(y_tr_)}, val={len(y_val_)}, test={len(y_te)})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    n_params = count_params(model_factory())
    print(f"  [{name_key}] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%  params={n_params}  time={elapsed:.1f}s")

    return {
        "name": name_str,
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "n_params": n_params,
        "train_time_s": round(elapsed, 2),
    }


# =====================================================================
# 10. Main Function
# =====================================================================
def main():
    (X_et, X_eeg, X_beh, y, subject_ids) = load_data()

    all_results = {}

    # B0: DCM-DNN
    set_seed(SEED)
    r_b0 = run_b0_dcm_dnn(X_et, X_eeg, X_beh, y)
    all_results["B0"] = r_b0
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Intermediate Save] {RESULT_PATH}")

    # B1: Logistic Regression
    set_seed(SEED)
    r_b1 = run_ml_baseline(
        "B1", "B1 Logistic Regression",
        lambda: LogisticRegression(max_iter=1000, random_state=42),
        X_et, X_eeg, X_beh, y
    )
    all_results["B1"] = r_b1
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # B2: SVM (RBF)
    set_seed(SEED)
    r_b2 = run_ml_baseline(
        "B2", "B2 SVM (RBF kernel)",
        lambda: SVC(kernel='rbf', probability=True, random_state=42),
        X_et, X_eeg, X_beh, y
    )
    all_results["B2"] = r_b2
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # B3: Simple MLP
    set_seed(SEED)
    r_b3 = run_mlp_baseline(
        "B3", "B3 Simple MLP (67→64→32→2)",
        lambda: SimpleMLP(),
        X_et, X_eeg, X_beh, y
    )
    all_results["B3"] = r_b3
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # B4: Matched MLP
    set_seed(SEED)
    r_b4 = run_mlp_baseline(
        "B4", "B4 Matched MLP (67→128→64→32→2)",
        lambda: MatchedMLP(),
        X_et, X_eeg, X_beh, y
    )
    all_results["B4"] = r_b4
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # =====================================================================
    # Summary Leaderboard
    # =====================================================================
    print("\n" + "=" * 88)
    print("=== Baseline Comparison Final Ranking ===")
    print("=" * 88)
    print(f"{'Rank':<4} {'Key':<4} {'Mean Acc':>8} {'±Std':>8} {'Params':>10}  Name")
    print("-" * 88)

    valid_results = [(k, r) for k, r in all_results.items() if r.get("acc_mean") is not None]
    sorted_all = sorted(valid_results, key=lambda x: x[1]["acc_mean"], reverse=True)
    for rank, (k, r) in enumerate(sorted_all, 1):
        n_p = r.get("n_params") or "-"
        marker = " ***" if r["acc_mean"] >= 0.84 else ""
        print(f"{rank:<4} {k:<4} {r['acc_mean']*100:>7.2f}%  {r['acc_std']*100:>6.2f}%  {str(n_p):>10}  {r['name']}{marker}")

    print("=" * 88)
    best_key, best_r = sorted_all[0]
    print(f"\n[Best] {best_key}: {best_r['acc_mean']*100:.2f}% ± {best_r['acc_std']*100:.2f}%")

    # Compare with B0
    b0_acc = all_results.get("B0", {}).get("acc_mean")
    if b0_acc:
        print(f"\n[Comparison] DCM-DNN (B0): {b0_acc*100:.2f}%")
        for k, r in sorted_all:
            if k != "B0" and r.get("acc_mean") is not None:
                delta = (b0_acc - r["acc_mean"]) * 100
                sign = "↑" if delta > 0 else "↓"
                print(f"  B0 vs {k}: {sign}{abs(delta):.2f}%  ({r['name']})")

    print(f"\n[Output] Results saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
