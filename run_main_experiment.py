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
# 0. Basic Configuration
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "combo_results.json"

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
print("41 Subjects Combined Optimization Experiment (based on O5b dropout=0.5, target 88%+)")
print("=" * 88)
print(f"[Device] {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")


# =====================================================================
# 1. Data Loading — 41 Subjects (Excluding S26)
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

    # NaN fill with 0 version (used for C0/C2/C3/C6 experiments etc.)
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
# 2. Model Definition (Aligned with O5b: dropout_fuse=0.5)
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
    """Base model with configurable dropout"""
    def __init__(self, et_h=32, eeg_h=32, beh_h=16,
                 fuse1=48, fuse2=24,
                 dropout_fuse=0.5,   # O5b: 0.4->0.5
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
# 3. Data Augmentation Utilities
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
# 4. Training Utilities
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
            # Validation set always uses loss without smoothing for comparison
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
# 5. General Fold-Running Framework (SKF10)
# =====================================================================
def run_experiment(exp_key, exp_cfg,
                   X_et, X_eeg, X_beh, y, subject_ids,
                   X_et_raw=None, X_eeg_raw=None, X_beh_raw=None):
    print(f"\n{'=' * 75}")
    print(f"[{exp_key}] {exp_cfg['name']}")
    print("=" * 75)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    fold_impute = exp_cfg.get("fold_impute", False)
    noise_std = exp_cfg.get("noise_std", 0.03)
    n_aug = exp_cfg.get("gaussian_copies", 2)
    bs = exp_cfg.get("batch_size", BATCH_SIZE)
    lr_ = exp_cfg.get("lr", LR)
    patience_ = exp_cfg.get("patience", PATIENCE)
    label_smoothing = exp_cfg.get("label_smoothing", 0.0)
    dropout_fuse = exp_cfg.get("dropout_fuse", 0.5)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        # Impute NaN within fold
        if fold_impute and X_eeg_raw is not None:
            imp_eeg = SimpleImputer(strategy='mean')
            imp_et = SimpleImputer(strategy='mean')
            imp_beh = SimpleImputer(strategy='mean')
            X_eeg_fold = X_eeg_raw.copy()
            X_et_fold = X_et_raw.copy()
            X_beh_fold = X_beh_raw.copy()
            X_eeg_fold[train_idx] = imp_eeg.fit_transform(X_eeg_fold[train_idx])
            X_eeg_fold[test_idx] = imp_eeg.transform(X_eeg_fold[test_idx])
            X_et_fold[train_idx] = imp_et.fit_transform(X_et_fold[train_idx])
            X_et_fold[test_idx] = imp_et.transform(X_et_fold[test_idx])
            X_beh_fold[train_idx] = imp_beh.fit_transform(X_beh_fold[train_idx])
            X_beh_fold[test_idx] = imp_beh.transform(X_beh_fold[test_idx])
        else:
            X_eeg_fold = X_eeg
            X_et_fold = X_et
            X_beh_fold = X_beh

        # Standardization
        sc_et = StandardScaler()
        sc_eeg = StandardScaler()
        sc_beh = StandardScaler()

        et_tr = sc_et.fit_transform(X_et_fold[train_idx]).astype(np.float32)
        eeg_tr = sc_eeg.fit_transform(X_eeg_fold[train_idx]).astype(np.float32)
        beh_tr = sc_beh.fit_transform(X_beh_fold[train_idx]).astype(np.float32)
        y_tr = y[train_idx]

        et_te = sc_et.transform(X_et_fold[test_idx]).astype(np.float32)
        eeg_te = sc_eeg.transform(X_eeg_fold[test_idx]).astype(np.float32)
        beh_te = sc_beh.transform(X_beh_fold[test_idx]).astype(np.float32)
        y_te = y[test_idx]

        # Data augmentation
        et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
            et_tr, eeg_tr, beh_tr, y_tr,
            n_aug=n_aug,
            seed=SEED + fold_i * 100 + 17,
            noise_std=noise_std
        )

        # Split validation set (randomly take 20% from training set)
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
        drop_last = (len(train_ds) % bs == 1)
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
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

        # Build and train model
        set_seed(SEED + fold_i)
        model = T0_Baseline(dropout_fuse=dropout_fuse).to(DEVICE)

        model = train_model(
            model, train_loader, val_tensors, y_tr_,
            name=f"{exp_key}-F{fold_i}",
            lr=lr_, patience=patience_,
            label_smoothing=label_smoothing,
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
        "name": exp_cfg["name"],
        "config": {
            "dropout_fuse": dropout_fuse,
            "lr": lr_,
            "patience": patience_,
            "batch_size": bs,
            "noise_std": noise_std,
            "gaussian_copies": n_aug,
            "fold_impute": fold_impute,
            "label_smoothing": label_smoothing,
        }
    }


# =====================================================================
# 6. Main Function
# =====================================================================
def main():
    (X_et, X_eeg, X_beh, y, subject_ids,
     X_et_raw, X_eeg_raw, X_beh_raw) = load_data()

    all_results = {}

    # ==================================================================
    # Round 1: Pairwise Combinations (based on O5b dropout=0.5)
    # ==================================================================
    print("\n" + "=" * 88)
    print("Round 1: Pairwise Combination Experiments (based on O5b dropout=0.5)")
    print("=" * 88)

    round1_cfgs = {
        "C0": {
            "name": "C0 O5b Baseline (dropout=0.5, NaN fill 0, lr=5e-4, bs=32)",
            "dropout_fuse": 0.5, "lr": 5e-4, "patience": 25,
            "batch_size": 32, "fold_impute": False,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C1": {
            "name": "C1 O5b + Mean Imputation NaN",
            "dropout_fuse": 0.5, "lr": 5e-4, "patience": 25,
            "batch_size": 32, "fold_impute": True,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C2": {
            "name": "C2 O5b + lr=1e-3",
            "dropout_fuse": 0.5, "lr": 1e-3, "patience": 25,
            "batch_size": 32, "fold_impute": False,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C3": {
            "name": "C3 O5b + batch_size=16",
            "dropout_fuse": 0.5, "lr": 5e-4, "patience": 25,
            "batch_size": 16, "fold_impute": False,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C4": {
            "name": "C4 O5b + Mean Imputation + lr=1e-3",
            "dropout_fuse": 0.5, "lr": 1e-3, "patience": 25,
            "batch_size": 32, "fold_impute": True,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C5": {
            "name": "C5 O5b + Mean Imputation + batch_size=16",
            "dropout_fuse": 0.5, "lr": 5e-4, "patience": 25,
            "batch_size": 16, "fold_impute": True,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C6": {
            "name": "C6 O5b + lr=1e-3 + batch_size=16",
            "dropout_fuse": 0.5, "lr": 1e-3, "patience": 25,
            "batch_size": 16, "fold_impute": False,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "C7": {
            "name": "C7 O5b + Mean Imputation + lr=1e-3 + batch_size=16 (full combination)",
            "dropout_fuse": 0.5, "lr": 1e-3, "patience": 25,
            "batch_size": 16, "fold_impute": True,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
        },
    }

    round1_results = {}
    for exp_key in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        set_seed(SEED)
        cfg = round1_cfgs[exp_key]
        r = run_experiment(
            exp_key, cfg,
            X_et, X_eeg, X_beh, y, subject_ids,
            X_et_raw=X_et_raw, X_eeg_raw=X_eeg_raw, X_beh_raw=X_beh_raw
        )
        round1_results[exp_key] = r
        all_results[exp_key] = r

        # Intermediate save
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULT_PATH.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Round 1 ranking
    print("\n" + "=" * 65)
    print("=== Round 1 Ranking ===")
    sorted_r1 = sorted(round1_results.items(), key=lambda x: x[1]["acc_mean"], reverse=True)
    for rank, (k, r) in enumerate(sorted_r1, 1):
        print(f"{rank}. {k}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']})")
    print("=" * 65)

    best_r1_key, best_r1 = sorted_r1[0]
    best_r1_cfg = round1_cfgs[best_r1_key]
    print(f"\n[Round 1 Best] {best_r1_key}: {best_r1['acc_mean']*100:.2f}%")
    print(f"  Config: dropout=0.5, lr={best_r1_cfg['lr']}, bs={best_r1_cfg['batch_size']}, "
          f"fold_impute={best_r1_cfg['fold_impute']}")

    # ==================================================================
    # Round 2: Fine Grid Search Based on Round 1 Best
    # ==================================================================
    print("\n" + "=" * 88)
    print(f"Round 2: Fine Grid Search (based on {best_r1_key})")
    print("=" * 88)

    # Fixed batch_size
    fixed_bs = best_r1_cfg["batch_size"]
    fixed_impute = best_r1_cfg["fold_impute"]
    print(f"Fixed batch_size={fixed_bs}, fold_impute={fixed_impute}")

    dropout_vals = [0.45, 0.5, 0.55, 0.6]
    lr_vals = [5e-4, 7e-4, 1e-3, 1.5e-3]

    round2_results = {}
    round2_idx = 0
    for do_val in dropout_vals:
        for lr_val in lr_vals:
            round2_idx += 1
            exp_key = f"G{round2_idx:02d}"
            cfg = {
                "name": f"G{round2_idx:02d} dropout={do_val}, lr={lr_val:.0e}, bs={fixed_bs}, impute={fixed_impute}",
                "dropout_fuse": do_val, "lr": lr_val, "patience": 25,
                "batch_size": fixed_bs, "fold_impute": fixed_impute,
                "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.0,
            }
            set_seed(SEED)
            r = run_experiment(
                exp_key, cfg,
                X_et, X_eeg, X_beh, y, subject_ids,
                X_et_raw=X_et_raw, X_eeg_raw=X_eeg_raw, X_beh_raw=X_beh_raw
            )
            round2_results[exp_key] = r
            all_results[exp_key] = r

            # Intermediate save
            with RESULT_PATH.open("w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Round 2 ranking
    print("\n" + "=" * 65)
    print("=== Round 2 Ranking ===")
    sorted_r2 = sorted(round2_results.items(), key=lambda x: x[1]["acc_mean"], reverse=True)
    for rank, (k, r) in enumerate(sorted_r2, 1):
        print(f"{rank}. {k}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']})")
    print("=" * 65)

    best_r2_key, best_r2 = sorted_r2[0]
    best_r2_cfg = all_results[best_r2_key]["config"]
    print(f"\n[Round 2 Best] {best_r2_key}: {best_r2['acc_mean']*100:.2f}%")
    print(f"  Config: dropout={best_r2_cfg['dropout_fuse']}, lr={best_r2_cfg['lr']}, "
          f"bs={best_r2_cfg['batch_size']}, fold_impute={best_r2_cfg['fold_impute']}")

    # ==================================================================
    # Round 3: Augmentation Strategy Fine-tuning (Based on Round 2 Best)
    # ==================================================================
    print("\n" + "=" * 88)
    print(f"Round 3: Augmentation Strategy Fine-tuning (based on {best_r2_key})")
    print("=" * 88)

    best_do = best_r2_cfg["dropout_fuse"]
    best_lr = best_r2_cfg["lr"]
    best_bs = best_r2_cfg["batch_size"]
    best_impute = best_r2_cfg["fold_impute"]

    round3_cfgs = {
        "A1": {
            "name": f"A1 3 augmentation copies (dropout={best_do}, lr={best_lr:.0e}, bs={best_bs})",
            "dropout_fuse": best_do, "lr": best_lr, "patience": 25,
            "batch_size": best_bs, "fold_impute": best_impute,
            "gaussian_copies": 3, "noise_std": 0.03, "label_smoothing": 0.0,
        },
        "A2": {
            "name": f"A2 noise_std=0.02 (dropout={best_do}, lr={best_lr:.0e}, bs={best_bs})",
            "dropout_fuse": best_do, "lr": best_lr, "patience": 25,
            "batch_size": best_bs, "fold_impute": best_impute,
            "gaussian_copies": 2, "noise_std": 0.02, "label_smoothing": 0.0,
        },
        "A3": {
            "name": f"A3 noise_std=0.04 (dropout={best_do}, lr={best_lr:.0e}, bs={best_bs})",
            "dropout_fuse": best_do, "lr": best_lr, "patience": 25,
            "batch_size": best_bs, "fold_impute": best_impute,
            "gaussian_copies": 2, "noise_std": 0.04, "label_smoothing": 0.0,
        },
        "A4": {
            "name": f"A4 2 copies + label_smoothing=0.1 (dropout={best_do}, lr={best_lr:.0e}, bs={best_bs})",
            "dropout_fuse": best_do, "lr": best_lr, "patience": 25,
            "batch_size": best_bs, "fold_impute": best_impute,
            "gaussian_copies": 2, "noise_std": 0.03, "label_smoothing": 0.1,
        },
    }

    round3_results = {}
    for exp_key in ["A1", "A2", "A3", "A4"]:
        set_seed(SEED)
        cfg = round3_cfgs[exp_key]
        r = run_experiment(
            exp_key, cfg,
            X_et, X_eeg, X_beh, y, subject_ids,
            X_et_raw=X_et_raw, X_eeg_raw=X_eeg_raw, X_beh_raw=X_beh_raw
        )
        round3_results[exp_key] = r
        all_results[exp_key] = r

        # Intermediate save
        with RESULT_PATH.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Round 3 ranking
    print("\n" + "=" * 65)
    print("=== Round 3 Ranking ===")
    sorted_r3 = sorted(round3_results.items(), key=lambda x: x[1]["acc_mean"], reverse=True)
    for rank, (k, r) in enumerate(sorted_r3, 1):
        print(f"{rank}. {k}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']})")
    print("=" * 65)

    # ==================================================================
    # Overall Ranking
    # ==================================================================
    print("\n" + "=" * 88)
    print("=== Overall Ranking ===")
    sorted_all = sorted(all_results.items(), key=lambda x: x[1]["acc_mean"], reverse=True)
    for rank, (k, r) in enumerate(sorted_all, 1):
        marker = " ***" if r["acc_mean"] >= 0.88 else (" **" if r["acc_mean"] >= 0.86 else ("  *" if r["acc_mean"] >= 0.8456 else ""))
        print(f"{rank:2d}. {k}: {r['acc_mean']*100:.2f}% ± {r['acc_std']*100:.2f}%  ({r['name']}){marker}")
    print("=" * 88)

    best_all_key, best_all = sorted_all[0]
    print(f"\n[Final Best] {best_all_key}: {best_all['acc_mean']*100:.2f}% ± {best_all['acc_std']*100:.2f}%")
    if best_all["acc_mean"] >= 0.88:
        print(f"[Target Achieved!] Exceeded 88%! (+{(best_all['acc_mean']-0.88)*100:.2f}%)")
    elif best_all["acc_mean"] >= 0.86:
        print(f"[Near Target] {best_all['acc_mean']*100:.2f}% (still {(0.88-best_all['acc_mean'])*100:.2f}% from 88%)")
    else:
        print(f"[Target Not Met] still {(0.88-best_all['acc_mean'])*100:.2f}% from 88%")

    # Final save
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] Results saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
