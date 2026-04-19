import json
import random
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =====================================================================
# 0. Basic Configuration
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "robustness.json"
SOURCE_SCRIPT = ROOT / "experiments" / "run_main_experiment.py"
BASELINE_SCRIPT = ROOT / "experiments" / "run_baseline_compare.py"

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

REPEAT_RANDOM_STATES = [42, 123, 456, 789, 1024]
BOOTSTRAP_N_RESAMPLES = 10000
LR_REFERENCE_ACC = 0.8258


# =====================================================================
# 1. Random Seed / Device
# =====================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _probe_cuda_device(index: int) -> bool:
    try:
        device = torch.device(f"cuda:{index}")
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        return True
    except Exception as exc:
        print(f"[Device] cuda:{index} unavailable, skipping: {exc}")
        return False


def build_device() -> torch.device:
    if torch.cuda.is_available():
        preferred = []
        if torch.cuda.device_count() > 1:
            preferred.append(1)
        preferred.append(0)

        tried = set()
        for idx in preferred:
            if idx in tried or idx >= torch.cuda.device_count():
                continue
            tried.add(idx)
            if _probe_cuda_device(idx):
                if idx != 1:
                    print(f"[Device] Warning: preferred cuda:1 unavailable, falling back to cuda:{idx}")
                return torch.device(f"cuda:{idx}")

        print("[Device] All GPU probes failed, falling back to CPU")
    return torch.device("cpu")


DEVICE = build_device()
set_seed(SEED)

print("=" * 88)
print("41-Subject DCM-DNN Statistical Robustness Validation Experiment")
print("=" * 88)
print(f"[Device] {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")


# =====================================================================
# 2. Data Loading — Strictly Copied from run_neuma_41subj_combo.py
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

    # Fill NaN with 0 (used for C0/C2/C3/C6 experiments)
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
# 3. Model Definition — Strictly Copied from run_neuma_41subj_combo.py
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
# 4. Data Augmentation / Training Utils — Strictly Copied from run_neuma_41subj_combo.py
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
# 5. Utility Functions
# =====================================================================
def json_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def normalize_subject_id(subject_id):
    if isinstance(subject_id, (int, np.integer)):
        return f"S{int(subject_id):02d}"
    text = str(subject_id).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return f"S{int(match.group(1)):02d}"
    return text


def subject_sort_key(subject_id):
    normalized = normalize_subject_id(subject_id)
    match = re.search(r"(\d+)", normalized)
    if match:
        return (0, int(match.group(1)), normalized)
    return (1, normalized)


def save_results(payload):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stage_done(results, stage_name):
    completed = results.setdefault("completed_stages", [])
    if stage_name not in completed:
        completed.append(stage_name)


def offline_gaussian_augment_flat(X_flat, y, n_aug, seed, noise_std=0.03):
    rng = np.random.RandomState(seed)
    x_std = np.std(X_flat, axis=0, keepdims=True)
    aug_x = [X_flat]
    aug_y = [y]
    for _ in range(n_aug):
        aug_x.append((X_flat + rng.normal(0.0, noise_std * np.maximum(x_std, 1e-6), size=X_flat.shape)).astype(np.float32))
        aug_y.append(y.copy())
    return np.concatenate(aug_x, axis=0), np.concatenate(aug_y, axis=0)


def prepare_dcm_fold(train_idx, test_idx, X_et, X_eeg, X_beh, y, subject_ids,
                     split_seed, fold_i, batch_size=BATCH_SIZE):
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

    et_tr, eeg_tr, beh_tr, y_tr = offline_gaussian_augment(
        et_tr, eeg_tr, beh_tr, y_tr,
        n_aug=2,
        seed=split_seed + fold_i * 100 + 17,
        noise_std=0.03,
    )

    rng_val = np.random.RandomState(split_seed + fold_i)
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

    generator = torch.Generator()
    generator.manual_seed(split_seed + fold_i)

    train_ds = TensorDataset(
        torch.tensor(et_tr_),
        torch.tensor(eeg_tr_),
        torch.tensor(beh_tr_),
        torch.tensor(y_tr_, dtype=torch.long),
    )
    drop_last = (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
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

    meta = {
        "train_size": int(len(y_tr_)),
        "val_size": int(len(y_val_)),
        "test_size": int(len(y_te)),
        "train_subjects": int(len(np.unique(subject_ids[train_idx]))),
        "test_subjects": int(len(np.unique(subject_ids[test_idx]))),
        "test_subject_ids": [normalize_subject_id(s) for s in np.unique(subject_ids[test_idx])],
    }
    return train_loader, val_tensors, test_tensors, y_tr_, y_te, meta


def run_single_dcm_fold(train_idx, test_idx, X_et, X_eeg, X_beh, y, subject_ids,
                        split_seed, fold_i, fold_label):
    train_loader, val_tensors, test_tensors, y_tr_, y_te, meta = prepare_dcm_fold(
        train_idx, test_idx, X_et, X_eeg, X_beh, y, subject_ids,
        split_seed=split_seed,
        fold_i=fold_i,
        batch_size=BATCH_SIZE,
    )

    set_seed(split_seed + fold_i)
    model = T0_Baseline(dropout_fuse=0.5).to(DEVICE)
    model = train_model(
        model,
        train_loader,
        val_tensors,
        y_tr_,
        name=fold_label,
        lr=LR,
        patience=PATIENCE,
        label_smoothing=0.0,
    )

    model.eval()
    with torch.no_grad():
        logits, _ = model(*test_tensors)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_te, preds))
    result = {
        "acc": acc,
        "y_true": [int(v) for v in y_te.tolist()],
        "preds": [int(v) for v in preds.tolist()],
        "meta": meta,
    }

    del model, train_loader, val_tensors, test_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# =====================================================================
# 6. S1: 5-Repeat Stratified 10-fold
# =====================================================================
def run_s1_repeated_stratified(results, X_et, X_eeg, X_beh, y, subject_ids):
    print("\n" + "=" * 88)
    print("[S1] 5-Repeat Stratified 10-fold")
    print("=" * 88)

    stage = {
        "status": "running",
        "description": "5 random_states each running full 10-fold StratifiedKFold",
        "random_states": REPEAT_RANDOM_STATES,
        "repeats": [],
        "total_fold_accs": [],
    }
    results["stages"]["S1"] = stage
    save_results(results)

    for repeat_idx, random_state in enumerate(REPEAT_RANDOM_STATES, start=1):
        repeat_result = {
            "repeat_index": repeat_idx,
            "random_state": int(random_state),
            "status": "running",
            "fold_accs": [],
            "fold_details": [],
            "train_time_s": None,
        }
        stage["repeats"].append(repeat_result)
        save_results(results)

        t_start = time.time()
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
            fold_out = run_single_dcm_fold(
                train_idx, test_idx,
                X_et, X_eeg, X_beh, y, subject_ids,
                split_seed=random_state,
                fold_i=fold_i,
                fold_label=f"S1-R{repeat_idx}-F{fold_i}",
            )
            fold_acc = float(fold_out["acc"])
            repeat_result["fold_accs"].append(fold_acc)
            stage["total_fold_accs"].append(fold_acc)
            repeat_result["fold_details"].append({
                "fold": int(fold_i),
                "acc": fold_acc,
                **fold_out["meta"],
            })
            print(f"    [S1 R{repeat_idx} F{fold_i:02d}] Acc={fold_acc:.4f}")
            save_results(results)

        repeat_result["acc_mean"] = float(np.mean(repeat_result["fold_accs"]))
        repeat_result["acc_std"] = float(np.std(repeat_result["fold_accs"]))
        repeat_result["train_time_s"] = round(time.time() - t_start, 2)
        repeat_result["status"] = "completed"
        print(
            f"  [S1 R{repeat_idx}] rs={random_state} -> "
            f"{repeat_result['acc_mean'] * 100:.2f}% ± {repeat_result['acc_std'] * 100:.2f}%"
        )
        save_results(results)

    stage["overall_mean"] = float(np.mean(stage["total_fold_accs"]))
    stage["overall_std"] = float(np.std(stage["total_fold_accs"]))
    stage["n_total_folds"] = int(len(stage["total_fold_accs"]))
    stage["status"] = "completed"
    stage_done(results, "S1")
    save_results(results)


# =====================================================================
# 7. S2: Bootstrap 95% Confidence Interval
# =====================================================================
def run_s2_bootstrap(results):
    print("\n" + "=" * 88)
    print("[S2] Bootstrap 95% CI")
    print("=" * 88)

    source_accs = np.asarray(results["stages"]["S1"]["total_fold_accs"], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(source_accs), size=(BOOTSTRAP_N_RESAMPLES, len(source_accs)))
    bootstrap_means = source_accs[indices].mean(axis=1)

    stage = {
        "status": "completed",
        "description": "10000 bootstrap resamples from S1's 50 fold accuracies",
        "source": "S1.total_fold_accs",
        "n_source_folds": int(len(source_accs)),
        "n_resamples": int(BOOTSTRAP_N_RESAMPLES),
        "mean": float(np.mean(source_accs)),
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
    }
    results["stages"]["S2"] = stage
    stage_done(results, "S2")
    save_results(results)
    print(
        f"  [S2] mean={stage['mean'] * 100:.2f}% | "
        f"95% CI=[{stage['ci_lower'] * 100:.2f}%, {stage['ci_upper'] * 100:.2f}%]"
    )


# =====================================================================
# 8. S3: Leave-One-Subject-Out
# =====================================================================
def run_s3_loso(results, X_et, X_eeg, X_beh, y, subject_ids):
    print("\n" + "=" * 88)
    print("[S3] Leave-One-Subject-Out")
    print("=" * 88)

    stage = {
        "status": "running",
        "description": "41-fold LOSO: 1 subject for test, remaining 40 for training per fold",
        "cv": "LeaveOneGroupOut",
        "folds": [],
    }
    results["stages"]["S3"] = stage
    save_results(results)

    logo = LeaveOneGroupOut()
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X_et, y, groups=subject_ids), start=1):
        test_subject_raw = np.unique(subject_ids[test_idx])
        assert len(test_subject_raw) == 1, "LOSO test set should contain exactly 1 subject"
        subject_raw = test_subject_raw[0]
        subject_norm = normalize_subject_id(subject_raw)

        fold_out = run_single_dcm_fold(
            train_idx, test_idx,
            X_et, X_eeg, X_beh, y, subject_ids,
            split_seed=SEED + 5000,
            fold_i=fold_i,
            fold_label=f"S3-{subject_norm}",
        )
        fold_item = {
            "fold": int(fold_i),
            "subject_id": subject_norm,
            "subject_id_raw": json_scalar(subject_raw),
            "acc": float(fold_out["acc"]),
            **fold_out["meta"],
        }
        stage["folds"].append(fold_item)
        print(f"    [S3 {subject_norm}] Acc={fold_item['acc']:.4f} | test={fold_item['test_size']}")
        save_results(results)

    stage["folds"] = sorted(stage["folds"], key=lambda item: subject_sort_key(item["subject_id"]))
    stage["fold_accs_sorted"] = [float(item["acc"]) for item in stage["folds"]]
    stage["mean"] = float(np.mean(stage["fold_accs_sorted"]))
    stage["std"] = float(np.std(stage["fold_accs_sorted"]))
    stage["n_folds"] = int(len(stage["folds"]))
    stage["status"] = "completed"
    stage_done(results, "S3")
    save_results(results)
    print(f"  [S3] mean={stage['mean'] * 100:.2f}% ± {stage['std'] * 100:.2f}%")


# =====================================================================
# 9. S4: McNemar Paired Test
# =====================================================================
def run_s4_mcnemar(results, X_et, X_eeg, X_beh, y, subject_ids):
    print("\n" + "=" * 88)
    print("[S4] McNemar Paired Test: DCM-DNN vs Logistic Regression")
    print("=" * 88)

    stage = {
        "status": "running",
        "description": "Collect per-sample predictions of DCM-DNN and LR during a full 10-fold CV",
        "cv_random_state": 42,
        "lr_reference_accuracy": LR_REFERENCE_ACC,
        "folds": [],
        "sample_records": [],
    }
    results["stages"]["S4"] = stage
    save_results(results)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    dcm_fold_accs = []
    lr_fold_accs = []
    all_y_true = []
    all_dcm_pred = []
    all_lr_pred = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        dcm_out = run_single_dcm_fold(
            train_idx, test_idx,
            X_et, X_eeg, X_beh, y, subject_ids,
            split_seed=42,
            fold_i=fold_i,
            fold_label=f"S4-DCM-F{fold_i}",
        )
        dcm_preds = np.asarray(dcm_out["preds"], dtype=np.int64)
        y_te = np.asarray(dcm_out["y_true"], dtype=np.int64)
        dcm_acc = float(dcm_out["acc"])
        dcm_fold_accs.append(dcm_acc)

        X_tr_raw = np.concatenate([X_et[train_idx], X_eeg[train_idx], X_beh[train_idx]], axis=1)
        X_te_raw = np.concatenate([X_et[test_idx], X_eeg[test_idx], X_beh[test_idx]], axis=1)
        y_tr = y[train_idx]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_raw).astype(np.float32)
        X_te_sc = sc.transform(X_te_raw).astype(np.float32)
        X_tr_aug, y_tr_aug = offline_gaussian_augment_flat(
            X_tr_sc, y_tr,
            n_aug=2,
            seed=SEED + fold_i * 100 + 17,
            noise_std=0.03,
        )
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_aug, y_tr_aug)
        lr_preds = lr.predict(X_te_sc).astype(np.int64)
        lr_acc = float(accuracy_score(y_te, lr_preds))
        lr_fold_accs.append(lr_acc)

        all_y_true.extend(int(v) for v in y_te.tolist())
        all_dcm_pred.extend(int(v) for v in dcm_preds.tolist())
        all_lr_pred.extend(int(v) for v in lr_preds.tolist())

        fold_record = {
            "fold": int(fold_i),
            "dcm_acc": dcm_acc,
            "lr_acc": lr_acc,
            "test_size": int(len(test_idx)),
        }
        stage["folds"].append(fold_record)

        for local_i, global_idx in enumerate(test_idx.tolist()):
            subject_raw = subject_ids[global_idx]
            stage["sample_records"].append({
                "sample_index": int(global_idx),
                "subject_id": normalize_subject_id(subject_raw),
                "subject_id_raw": json_scalar(subject_raw),
                "fold": int(fold_i),
                "y_true": int(y_te[local_i]),
                "dcm_pred": int(dcm_preds[local_i]),
                "lr_pred": int(lr_preds[local_i]),
                "dcm_correct": bool(dcm_preds[local_i] == y_te[local_i]),
                "lr_correct": bool(lr_preds[local_i] == y_te[local_i]),
            })

        print(f"    [S4 F{fold_i:02d}] DCM={dcm_acc:.4f} | LR={lr_acc:.4f}")
        save_results(results)

    stage["sample_records"] = sorted(stage["sample_records"], key=lambda item: item["sample_index"])

    dcm_correct = np.asarray(all_dcm_pred) == np.asarray(all_y_true)
    lr_correct = np.asarray(all_lr_pred) == np.asarray(all_y_true)
    a = int(np.sum(dcm_correct & lr_correct))
    b = int(np.sum(dcm_correct & ~lr_correct))
    c = int(np.sum(~dcm_correct & lr_correct))
    d = int(np.sum(~dcm_correct & ~lr_correct))
    table = [[a, b], [c, d]]

    chi2_stat = 0.0 if (b + c) == 0 else float(((b - c) ** 2) / (b + c))
    mcnemar_method = "scipy.binomtest(exact fallback)"
    p_value = 1.0 if (b + c) == 0 else float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        mcnemar_result = mcnemar(table, exact=True)
        p_value = float(mcnemar_result.pvalue)
        mcnemar_method = "statsmodels.mcnemar(exact=True)"
    except Exception:
        pass

    stage["contingency_table"] = {
        "table": table,
        "a_dcm_correct_lr_correct": a,
        "b_dcm_correct_lr_wrong": b,
        "c_dcm_wrong_lr_correct": c,
        "d_dcm_wrong_lr_wrong": d,
    }
    stage["dcm_acc_mean"] = float(np.mean(dcm_fold_accs))
    stage["dcm_acc_std"] = float(np.std(dcm_fold_accs))
    stage["lr_acc_mean"] = float(np.mean(lr_fold_accs))
    stage["lr_acc_std"] = float(np.std(lr_fold_accs))
    stage["chi2_statistic"] = chi2_stat
    stage["p_value"] = p_value
    stage["method"] = mcnemar_method
    stage["n_samples"] = int(len(all_y_true))
    stage["status"] = "completed"
    stage_done(results, "S4")
    save_results(results)
    print(f"  [S4] table={table} | chi2={chi2_stat:.4f} | p={p_value:.6f} | {mcnemar_method}")


# =====================================================================
# 10. Main Function
# =====================================================================
def main():
    t0 = time.time()
    results = {
        "experiment": "NeuMa 41-subject DCM-DNN statistical robustness validation",
        "source_script_alignment": "Strictly copied load_data / A1EEGBranch / T0_Baseline / offline_gaussian_augment / train_model from run_neuma_41subj_combo.py",
        "source_script": str(SOURCE_SCRIPT),
        "baseline_reference_script": str(BASELINE_SCRIPT),
        "data_source": str(NPZ_PATH),
        "result_path": str(RESULT_PATH),
        "python_executable": sys.executable,
        "device": str(DEVICE),
        "seed": SEED,
        "repeat_random_states": REPEAT_RANDOM_STATES,
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "completed_stages": [],
        "status": "running",
        "stages": {},
    }
    save_results(results)

    try:
        (X_et, X_eeg, X_beh, y, subject_ids,
         X_et_raw, X_eeg_raw, X_beh_raw) = load_data()

        X_combined = np.concatenate([X_et, X_eeg, X_beh], axis=1)
        results["n_samples"] = int(len(y))
        results["n_subjects"] = int(len(np.unique(subject_ids)))
        results["label_distribution"] = {
            "0_not_buy": int(np.sum(y == 0)),
            "1_buy": int(np.sum(y == 1)),
        }
        results["input_features"] = {
            "X_et": int(X_et.shape[1]),
            "X_eeg_c1": int(X_eeg.shape[1]),
            "X_beh": int(X_beh.shape[1]),
            "combined_dim": int(X_combined.shape[1]),
        }
        results["nan_handling"] = {
            "copied_from_combo": True,
            "X_et": "np.nan_to_num(..., 0.0)",
            "X_eeg_c1": "np.nan_to_num(..., 0.0)",
            "X_beh": "np.nan_to_num(..., 0.0)",
            "raw_arrays_also_retained": True,
        }
        results["hyperparams"] = {
            "dropout_fuse": 0.5,
            "dropout_et": 0.3,
            "dropout_beh": 0.2,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "lr_patience": LR_PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "asc_l2": ASC_L2,
            "gaussian_copies": 2,
            "gaussian_noise_std_factor": 0.03,
        }
        save_results(results)

        run_s1_repeated_stratified(results, X_et, X_eeg, X_beh, y, subject_ids)
        run_s2_bootstrap(results)
        run_s3_loso(results, X_et, X_eeg, X_beh, y, subject_ids)
        run_s4_mcnemar(results, X_et, X_eeg, X_beh, y, subject_ids)

        results["status"] = "completed"
        results["total_runtime_s"] = round(time.time() - t0, 2)
        save_results(results)
        print(f"\n[Done] Results saved to: {RESULT_PATH}")

    except Exception as exc:
        results["status"] = "failed"
        results["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        results["failed_runtime_s"] = round(time.time() - t0, 2)
        save_results(results)
        raise


if __name__ == "__main__":
    main()
