import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "final_results.json"

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
GAUSSIAN_COPIES = 2
D3_NOISE_STD_FACTOR = 0.03
EXCLUDED_SUBJECT = 26
EXCLUDED_SUBJECT_LABEL = "S26"

REF_34SUBJ_GKF5 = 0.8328
REF_34SUBJ_SKF10 = 0.8305
REF_42SUBJ_SKF10 = 0.8118
SOTA = 0.8401
TARGET = 0.88


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
            torch.cuda.set_device(1)
            return torch.device("cuda:1")
        print("[Device] Warning: only 1 GPU detected, falling back to cuda:0")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = build_device()
set_seed(SEED)

print("=" * 96)
print("NeuMa 41-Subject Final Rerun - Excluding S26 + Strictly Copying DCM-DNN from run_neuma_augment_v2.py")
print("=" * 96)
print(f"[Device] {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")


# =====================================================================
# 1. Data Loading (Excluding S26, 41 subjects, fill NaN with 0 for X_et / X_eeg_c1)
# =====================================================================
def load_data():
    print(f"\n[Data] Loading: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)

    subject_ids_all = np.asarray(data["subject_ids"])
    subject_ids_str = subject_ids_all.astype(str)
    X_et = data["X_et"].astype(np.float32)
    X_eeg_c1 = data["X_eeg_c1"].astype(np.float32)
    X_beh = data["X_beh"].astype(np.float32)
    y = data["y_binary"].astype(np.int64)
    subject_ids = subject_ids_str

    # Exclude S26 — compatible with string 'S26' and integer 26
    def is_s26(sid):
        s = str(sid).strip().upper()
        return s == 'S26' or s == '26'

    mask_keep = np.array([not is_s26(s) for s in subject_ids])
    X_et = X_et[mask_keep]
    X_eeg_c1 = X_eeg_c1[mask_keep]
    X_beh = X_beh[mask_keep]
    y = y[mask_keep]
    subject_ids = subject_ids[mask_keep]
    print(f"[Data] After excluding S26: samples={len(y)}, subjects={len(np.unique(subject_ids))}")

    x_et_nan_before = int(np.isnan(X_et).sum())
    x_eeg_nan_before = int(np.isnan(X_eeg_c1).sum())
    X_et = np.nan_to_num(X_et, nan=0.0).astype(np.float32)
    X_eeg_c1 = np.nan_to_num(X_eeg_c1, nan=0.0).astype(np.float32)

    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) != 41:
        raise RuntimeError(f"Expected 41 unique subjects, got {len(unique_subjects)}")

    print(f"[Data] All subjects: {len(unique_subjects)}")
    print(f"[Data] X_et={X_et.shape}, X_eeg_c1={X_eeg_c1.shape}, X_beh={X_beh.shape}")
    print(f"[Data] Labels: 0={int(np.sum(y == 0))}, 1={int(np.sum(y == 1))}")
    print(f"[Data] Unique subjects: {len(unique_subjects)}")
    print(f"[Data] Sample count: {len(y)}")
    print(f"[Data] NaN filled with 0: X_et={x_et_nan_before}, X_eeg_c1={x_eeg_nan_before}")
    return X_et, X_eeg_c1, X_beh, y, subject_ids


# =====================================================================
# 2. Model Definition (Strictly Copied from run_neuma_augment_v2.py / T0_Baseline)
# =====================================================================
class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class A1EEGBranch(nn.Module):
    """A1: power(35→24), tbr(5→8), asym(6→8) each encoded separately then cat → 40 → h_eeg(32)"""
    def __init__(self):
        super().__init__()
        self.power_branch = nn.Sequential(nn.Linear(35, 24), nn.ReLU(), nn.BatchNorm1d(24))
        self.tbr_branch = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.BatchNorm1d(8))
        self.asym_branch = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.BatchNorm1d(8))
        self.fuse = nn.Sequential(nn.Linear(40, 32), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        x_power = x[:, :35]
        x_tbr = x[:, 35:40]
        x_asym = x[:, 40:46]
        h = torch.cat([
            self.power_branch(x_power),
            self.tbr_branch(x_tbr),
            self.asym_branch(x_asym),
        ], dim=1)
        return self.fuse(h)


class DCM_DNN_EEG(nn.Module):
    """
    Strictly reproduces the original S1 architecture (T0_Baseline):
    et(14)→32(BN,Drop0.3), A1 eeg→32, beh(7)→16(Drop0.2)
    F2 Gating: ctx(80→32)→gates→gated(80)→fusion(80→48→24)→utility(24→2)
    S1 ASC: [tbr(5),asym(6)]→Linear(11,8)→ReLU→Linear(8,2)
    """
    def __init__(self):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
        )
        self.eeg_branch = A1EEGBranch()
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, 16), nn.ReLU(), nn.Dropout(0.2),
        )
        self.ctx = nn.Sequential(nn.Linear(80, 32), nn.ReLU())
        self.gate_et = nn.Linear(32, 32)
        self.gate_eeg = nn.Linear(32, 32)
        self.gate_beh = nn.Linear(32, 16)
        self.fusion = nn.Sequential(
            nn.Linear(80, 48), nn.ReLU(), nn.BatchNorm1d(48), nn.Dropout(0.4),
            nn.Linear(48, 24), nn.ReLU(),
        )
        self.utility = nn.Linear(24, NUM_CLASSES)
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
# 3. Training Utils (Strictly Copied from run_neuma_augment_v2.py key logic)
# =====================================================================
def make_group_val_split(train_idx, groups, seed=SEED):
    unique_groups = np.unique(groups[train_idx])
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) / 5)))
    val_groups = set(shuffled[:n_val])
    inner_train_idx = train_idx[np.array([g not in val_groups for g in groups[train_idx]])]
    val_idx = train_idx[np.array([g in val_groups for g in groups[train_idx]])]
    if len(inner_train_idx) == 0 or len(val_idx) == 0:
        cut = max(1, len(train_idx) // 5)
        val_idx = train_idx[:cut]
        inner_train_idx = train_idx[cut:]
    return inner_train_idx, val_idx


def make_stratified_val_split(train_idx, y, seed):
    y_train = y[train_idx]
    unique, counts = np.unique(y_train, return_counts=True)
    if len(train_idx) < 20 or np.min(counts) < 2 or len(unique) < 2:
        inner_train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=seed, shuffle=True
        )
    else:
        inner_train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=seed, shuffle=True, stratify=y_train
        )
    return np.asarray(inner_train_idx), np.asarray(val_idx)


def compute_class_weights(y_arr):
    counts = np.bincount(y_arr, minlength=NUM_CLASSES)
    weights = len(y_arr) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def offline_gaussian_augment(X_et, X_eeg, X_beh, y, n_aug, seed):
    rng = np.random.RandomState(seed)
    et_std = np.std(X_et, axis=0, keepdims=True)
    eeg_std = np.std(X_eeg, axis=0, keepdims=True)
    beh_std = np.std(X_beh, axis=0, keepdims=True)

    aug_et = [X_et]
    aug_eeg = [X_eeg]
    aug_beh = [X_beh]
    aug_y = [y]

    for _ in range(n_aug):
        aug_et.append((X_et + rng.normal(0.0, 0.03 * np.maximum(et_std, 1e-6), size=X_et.shape)).astype(np.float32))
        aug_eeg.append((X_eeg + rng.normal(0.0, 0.03 * np.maximum(eeg_std, 1e-6), size=X_eeg.shape)).astype(np.float32))
        aug_beh.append((X_beh + rng.normal(0.0, 0.03 * np.maximum(beh_std, 1e-6), size=X_beh.shape)).astype(np.float32))
        aug_y.append(y.copy())

    return (
        np.concatenate(aug_et, axis=0),
        np.concatenate(aug_eeg, axis=0),
        np.concatenate(aug_beh, axis=0),
        np.concatenate(aug_y, axis=0),
    )


def build_fold_tensors(X_et, X_eeg, X_beh, y, inner_train_idx, val_idx, test_idx, fold_i):
    sc_et = StandardScaler()
    sc_eeg = StandardScaler()
    sc_beh = StandardScaler()

    et_tr = sc_et.fit_transform(X_et[inner_train_idx]).astype(np.float32)
    eeg_tr = sc_eeg.fit_transform(X_eeg[inner_train_idx]).astype(np.float32)
    beh_tr = sc_beh.fit_transform(X_beh[inner_train_idx]).astype(np.float32)
    y_tr = y[inner_train_idx]

    et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
        et_tr,
        eeg_tr,
        beh_tr,
        y_tr,
        n_aug=GAUSSIAN_COPIES,
        seed=SEED + fold_i * 100 + 17,
    )

    et_val = sc_et.transform(X_et[val_idx]).astype(np.float32)
    eeg_val = sc_eeg.transform(X_eeg[val_idx]).astype(np.float32)
    beh_val = sc_beh.transform(X_beh[val_idx]).astype(np.float32)
    y_val = y[val_idx]

    et_te = sc_et.transform(X_et[test_idx]).astype(np.float32)
    eeg_te = sc_eeg.transform(X_eeg[test_idx]).astype(np.float32)
    beh_te = sc_beh.transform(X_beh[test_idx]).astype(np.float32)
    y_te = y[test_idx]

    generator = torch.Generator()
    generator.manual_seed(SEED + fold_i)
    train_ds = TensorDataset(
        torch.tensor(et_tr),
        torch.tensor(eeg_tr),
        torch.tensor(beh_tr),
        torch.tensor(y_tr, dtype=torch.long),
    )
    drop_last = (len(train_ds) % BATCH_SIZE == 1)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=drop_last, generator=generator,
    )
    val_tensors = [
        torch.tensor(et_val),
        torch.tensor(eeg_val),
        torch.tensor(beh_val),
        torch.tensor(y_val, dtype=torch.long),
    ]
    test_tensors = [
        torch.tensor(et_te, device=DEVICE),
        torch.tensor(eeg_te, device=DEVICE),
        torch.tensor(beh_te, device=DEVICE),
    ]
    meta = {
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_te)),
        "train_subjects": int(len(np.unique(subject_ids_global[inner_train_idx]))),
        "val_subjects": int(len(np.unique(subject_ids_global[val_idx]))),
        "test_subjects": int(len(np.unique(subject_ids_global[test_idx]))),
    }
    return train_loader, val_tensors, test_tensors, y_tr, y_te, meta


def train_and_evaluate(model, train_loader, val_tensors, y_tr, model_name):
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))

    optimizer = optim.Adam([
        {'params': model.get_non_asc_params(), 'weight_decay': WEIGHT_DECAY},
        {'params': model.get_asc_params(), 'weight_decay': ASC_L2},
    ], lr=LR)

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
            val_loss = criterion(val_logits, val_y).item()
        scheduler.step(val_loss)

        if epoch % 100 == 0:
            print(
                f"      [{model_name}] Epoch {epoch:3d} | "
                f"train={np.mean(epoch_losses):.4f} val={val_loss:.4f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"      [{model_name}] EarlyStopping @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


# =====================================================================
# 4. Experiments
# =====================================================================
def run_experiment_f1(X_et, X_eeg, X_beh, y, subject_ids):
    print(f"\n{'=' * 84}")
    print("[F1] 41-Subject + Stratified 10-fold + D3 Augmentation")
    print("=" * 84)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []
    fold_details = []
    X_combined = np.concatenate([X_eeg, X_et, X_beh], axis=1).astype(np.float32)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_combined, y), start=1):
        print(f"  Fold {fold_i}/10 ...")
        inner_train_idx, val_idx = make_stratified_val_split(train_idx, y, seed=SEED + fold_i)
        train_loader, val_tensors, test_tensors, y_tr, y_te, meta = build_fold_tensors(
            X_et, X_eeg, X_beh, y, inner_train_idx, val_idx, test_idx, fold_i
        )

        set_seed(SEED + fold_i)
        model = DCM_DNN_EEG().to(DEVICE)
        model = train_and_evaluate(model, train_loader, val_tensors, y_tr, f"F1-F{fold_i}")

        model.eval()
        with torch.no_grad():
            logits, _ = model(*test_tensors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = float(accuracy_score(y_te, preds))
        fold_accs.append(acc)
        detail = {
            "fold": fold_i,
            "acc": acc,
            **meta,
        }
        fold_details.append(detail)
        print(
            f"    [F1 F{fold_i}] Acc={acc:.4f} | "
            f"train={meta['train_size']} val={meta['val_size']} test={meta['test_size']} | "
            f"train_subj={meta['train_subjects']} val_subj={meta['val_subjects']} test_subj={meta['test_subjects']}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"  [F1] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    return {
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "fold_details": fold_details,
        "cv": "StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
        "augmentation": "D3 Gaussian, 2 copies, noise_std=0.03*feature_std",
    }


def run_experiment_f2(X_et, X_eeg, X_beh, y, subject_ids):
    print(f"\n{'=' * 84}")
    print("[F2] 41-Subject + GroupKFold-5 + D3 Augmentation")
    print("=" * 84)

    gkf = GroupKFold(n_splits=5)
    fold_accs = []
    fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_et, y, subject_ids), start=1):
        print(f"  Fold {fold_i}/5 ...")
        inner_train_idx, val_idx = make_group_val_split(train_idx, subject_ids, seed=SEED + fold_i)
        train_loader, val_tensors, test_tensors, y_tr, y_te, meta = build_fold_tensors(
            X_et, X_eeg, X_beh, y, inner_train_idx, val_idx, test_idx, fold_i
        )

        set_seed(SEED + fold_i)
        model = DCM_DNN_EEG().to(DEVICE)
        model = train_and_evaluate(model, train_loader, val_tensors, y_tr, f"F2-F{fold_i}")

        model.eval()
        with torch.no_grad():
            logits, _ = model(*test_tensors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = float(accuracy_score(y_te, preds))
        fold_accs.append(acc)
        detail = {
            "fold": fold_i,
            "acc": acc,
            **meta,
        }
        fold_details.append(detail)
        print(
            f"    [F2 F{fold_i}] Acc={acc:.4f} | "
            f"train={meta['train_size']} val={meta['val_size']} test={meta['test_size']} | "
            f"train_subj={meta['train_subjects']} val_subj={meta['val_subjects']} test_subj={meta['test_subjects']}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"  [F2] Mean={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    return {
        "acc_mean": mean_acc,
        "acc_std": std_acc,
        "fold_accs": fold_accs,
        "fold_details": fold_details,
        "cv": "GroupKFold(n_splits=5)",
        "augmentation": "D3 Gaussian, 2 copies, noise_std=0.03*feature_std",
    }


def print_summary(results):
    print("\n" + "=" * 118)
    print("41-Subject Final Rerun Results Summary")
    print("=" * 118)
    print(f"{'Config':<12}{'Mean Acc':>12}{'Std':>10}{'vs34-GKF5':>12}{'vs34-SKF10':>13}{'vs42-SKF10':>13}{'vsSOTA':>11}{'vs88%':>10}")
    print("-" * 118)
    for key, info in results.items():
        print(
            f"{key:<12}"
            f"{info['acc_mean'] * 100:>11.2f}%"
            f"{info['acc_std'] * 100:>9.2f}%"
            f"{(info['acc_mean'] - REF_34SUBJ_GKF5) * 100:>+11.2f}%"
            f"{(info['acc_mean'] - REF_34SUBJ_SKF10) * 100:>+12.2f}%"
            f"{(info['acc_mean'] - REF_42SUBJ_SKF10) * 100:>+12.2f}%"
            f"{(info['acc_mean'] - SOTA) * 100:>+10.2f}%"
            f"{(info['acc_mean'] - TARGET) * 100:>+9.2f}%"
        )
    print("-" * 118)
    for key, info in results.items():
        folds = ", ".join(f"{x * 100:.2f}%" for x in info["fold_accs"])
        print(f"{key} per-fold: [{folds}]")
    print("=" * 118)


subject_ids_global = None


def main():
    global subject_ids_global

    X_et, X_eeg_c1, X_beh, y, subject_ids = load_data()
    subject_ids_global = subject_ids
    X_combined = np.concatenate([X_eeg_c1, X_et, X_beh], axis=1).astype(np.float32)

    print(f"[Features] X_eeg_c1({X_eeg_c1.shape[1]}) + X_et({X_et.shape[1]}) + X_beh({X_beh.shape[1]}) = {X_combined.shape[1]} dims")
    print(f"[Verify] Unique subjects should be 41, current={len(np.unique(subject_ids))}")
    results = {
        "F1_41subj_SKF10_D3": run_experiment_f1(X_et, X_eeg_c1, X_beh, y, subject_ids),
        "F2_41subj_GKF5_D3": run_experiment_f2(X_et, X_eeg_c1, X_beh, y, subject_ids),
    }

    print_summary(results)

    output = {
        "experiment": "NeuMa 41-subject final rerun after excluding S26",
        "source_script_alignment": "Strictly copied DCM_DNN_EEG and train_and_evaluate from run_neuma_augment_v2.py lineage",
        "data_source": str(NPZ_PATH),
        "result_path": str(RESULT_PATH),
        "device": str(DEVICE),
        "seed": SEED,
        "excluded_subject": int(EXCLUDED_SUBJECT),
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(subject_ids))),
        "label_distribution": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "input_features": {
            "X_eeg_c1": int(X_eeg_c1.shape[1]),
            "X_et": int(X_et.shape[1]),
            "X_beh": int(X_beh.shape[1]),
            "combined_dim": int(X_combined.shape[1]),
        },
        "nan_handling": {
            "X_et": "np.nan_to_num(..., 0)",
            "X_eeg_c1": "np.nan_to_num(..., 0)",
        },
        "hyperparams": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "lr_patience": LR_PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "asc_l2": ASC_L2,
            "gaussian_copies": GAUSSIAN_COPIES,
            "gaussian_noise_std_factor": D3_NOISE_STD_FACTOR,
            "fusion": "80->48->24->2",
        },
        "references": {
            "34subj_gkf5": REF_34SUBJ_GKF5,
            "34subj_skf10": REF_34SUBJ_SKF10,
            "42subj_skf10": REF_42SUBJ_SKF10,
            "sota": SOTA,
            "target": TARGET,
        },
        "results": results,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] Results saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
