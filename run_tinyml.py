import copy
import json
import os
import platform
import random
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =====================================================================
# 0. Basic Configuration (strictly aligned with run_neuma_41subj_combo.py)
# =====================================================================
ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = ROOT / "results" / "neuma_42subj_v2_features.npz"
RESULT_PATH = ROOT / "results" / "tinyml.json"
ARTIFACT_DIR = ROOT / "results" / "tinyml_artifacts"
STATE_DICT_DIR = ARTIFACT_DIR / "state_dicts"
ONNX_PATH = ARTIFACT_DIR / "dcm_dnn_fold01.onnx"
C_HEADER_PATH = ARTIFACT_DIR / "dcm_dnn_weights.h"

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
BASELINE_TARGET_ACC = 0.8406
GAUSSIAN_COPIES = 2
D3_NOISE_STD_FACTOR = 0.03
CPU_DEVICE = torch.device("cpu")
ESP32_SPECS = {
    "flash_kb": 4096,
    "sram_kb": 520,
    "clock_mhz": 240,
}
PC_CLOCK_MHZ_ASSUMED = 3500
ESP32_OVERHEAD_FACTOR = 2.0


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
        print("[Device] Warning: only 1 GPU detected, falling back to cuda:0")
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = build_device()
set_seed(SEED)

print("=" * 96)
print("NeuMa 41-Subject DCM-DNN TinyML Quantization and ESP32 Deployment Feasibility Study")
print("Strictly copies the C0 model and data pipeline from run_neuma_41subj_combo.py")
print("=" * 96)
print(f"[Device] Training on: {DEVICE}")
if torch.cuda.is_available():
    idx = DEVICE.index if DEVICE.index is not None else 0
    print(f"[GPU]  {torch.cuda.get_device_name(idx)}")
print("[Device] Quantization/latency testing on: cpu")


# =====================================================================
# 1. Data Loading — Strictly Copied from run_neuma_41subj_combo.py
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

    X_eeg_c1_raw = np.nan_to_num(X_eeg_c1_raw, nan=0.0).astype(np.float32)
    X_et_raw = np.nan_to_num(X_et_raw, nan=0.0).astype(np.float32)
    X_beh_raw = np.nan_to_num(X_beh_raw, nan=0.0).astype(np.float32)
    X_eeg_c1_zero = np.nan_to_num(X_eeg_c1_raw, nan=0.0).astype(np.float32)
    X_et_zero = np.nan_to_num(X_et_raw, nan=0.0).astype(np.float32)
    X_beh_zero = np.nan_to_num(X_beh_raw, nan=0.0).astype(np.float32)

    print(f"[Data] X_et={X_et_zero.shape}, X_eeg_c1={X_eeg_c1_zero.shape}, X_beh={X_beh_zero.shape}")
    print(f"[Data] Labels: 0={int(np.sum(y == 0))}, 1={int(np.sum(y == 1))}")
    print(f"[Data] Subjects: {len(np.unique(subject_ids))}, samples: {len(y)}")

    return (
        X_et_zero,
        X_eeg_c1_zero,
        X_beh_zero,
        y,
        subject_ids,
        X_et_raw,
        X_eeg_c1_raw,
        X_beh_raw,
    )


# =====================================================================
# 2. Model Definition — Strictly Copied from run_neuma_41subj_combo.py
# =====================================================================
class A1EEGBranch(nn.Module):
    def __init__(
        self,
        eeg_dim=EEG_DIM,
        p1=35,
        p2=5,
        p3=6,
        h_power=24,
        h_tbr=8,
        h_asym=8,
        h_fuse=32,
        dropout=0.3,
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.power_branch = nn.Sequential(nn.Linear(p1, h_power), nn.ReLU(), nn.BatchNorm1d(h_power))
        self.tbr_branch = nn.Sequential(nn.Linear(p2, h_tbr), nn.ReLU(), nn.BatchNorm1d(h_tbr))
        self.asym_branch = nn.Sequential(nn.Linear(p3, h_asym), nn.ReLU(), nn.BatchNorm1d(h_asym))
        self.fuse = nn.Sequential(nn.Linear(h_power + h_tbr + h_asym, h_fuse), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x_power = x[:, : self.p1]
        x_tbr = x[:, self.p1 : self.p1 + self.p2]
        x_asym = x[:, self.p1 + self.p2 : self.p1 + self.p2 + self.p3]
        h = torch.cat(
            [
                self.power_branch(x_power),
                self.tbr_branch(x_tbr),
                self.asym_branch(x_asym),
            ],
            dim=1,
        )
        return self.fuse(h)


class T0_Baseline(nn.Module):
    """Base model with configurable dropout"""

    def __init__(
        self,
        et_h=32,
        eeg_h=32,
        beh_h=16,
        fuse1=48,
        fuse2=24,
        dropout_fuse=0.5,
        dropout_et=0.3,
        dropout_beh=0.2,
    ):
        super().__init__()
        self.et_branch = nn.Sequential(
            nn.Linear(ET_DIM, et_h),
            nn.ReLU(),
            nn.BatchNorm1d(et_h),
            nn.Dropout(dropout_et),
        )
        self.eeg_branch = A1EEGBranch(h_fuse=eeg_h)
        self.beh_branch = nn.Sequential(
            nn.Linear(BEH_DIM, beh_h),
            nn.ReLU(),
            nn.Dropout(dropout_beh),
        )
        total_h = et_h + eeg_h + beh_h
        self.ctx = nn.Sequential(nn.Linear(total_h, 32), nn.ReLU())
        self.gate_et = nn.Linear(32, et_h)
        self.gate_eeg = nn.Linear(32, eeg_h)
        self.gate_beh = nn.Linear(32, beh_h)
        self.fusion = nn.Sequential(
            nn.Linear(total_h, fuse1),
            nn.ReLU(),
            nn.BatchNorm1d(fuse1),
            nn.Dropout(dropout_fuse),
            nn.Linear(fuse1, fuse2),
            nn.ReLU(),
        )
        self.utility = nn.Linear(fuse2, NUM_CLASSES)
        self.asc_net = nn.Sequential(
            nn.Linear(11, 8),
            nn.ReLU(),
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
# 3. Data Augmentation Utils — Strictly Copied from run_neuma_41subj_combo.py
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
        aug_et.append(
            (X_et + rng.normal(0.0, noise_std * np.maximum(et_std, 1e-6), size=X_et.shape)).astype(np.float32)
        )
        aug_eeg.append(
            (X_eeg + rng.normal(0.0, noise_std * np.maximum(eeg_std, 1e-6), size=X_eeg.shape)).astype(np.float32)
        )
        aug_beh.append(
            (X_beh + rng.normal(0.0, noise_std * np.maximum(beh_std, 1e-6), size=X_beh.shape)).astype(np.float32)
        )
        aug_y.append(y.copy())

    return (
        np.concatenate(aug_et, axis=0),
        np.concatenate(aug_eeg, axis=0),
        np.concatenate(aug_beh, axis=0),
        np.concatenate(aug_y, axis=0),
    )


# =====================================================================
# 4. Training Utils — Strictly Copied from run_neuma_41subj_combo.py
# =====================================================================
def compute_class_weights(y_arr):
    counts = np.bincount(y_arr, minlength=NUM_CLASSES)
    weights = len(y_arr) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(model, train_loader, val_tensors, y_tr, name, lr=LR, patience=PATIENCE, weight_decay=WEIGHT_DECAY, label_smoothing=0.0):
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr), label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr))

    if hasattr(model, "get_non_asc_params") and hasattr(model, "get_asc_params"):
        optimizer = optim.Adam(
            [
                {"params": model.get_non_asc_params(), "weight_decay": weight_decay},
                {"params": model.get_asc_params(), "weight_decay": ASC_L2},
            ],
            lr=lr,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE)

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
# 5. TinyML Helper Utils
# =====================================================================
class LogitsOnlyWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model

    def forward(self, x_et, x_eeg, x_beh):
        logits, _ = self.core_model(x_et, x_eeg, x_beh)
        return logits


class StaticQuantWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.quant_et = quant.QuantStub()
        self.quant_eeg = quant.QuantStub()
        self.quant_beh = quant.QuantStub()
        self.core_model = core_model
        self.dequant = quant.DeQuantStub()

    def forward(self, x_et, x_eeg, x_beh):
        x_et = self.quant_et(x_et)
        x_eeg = self.quant_eeg(x_eeg)
        x_beh = self.quant_beh(x_beh)
        logits, _ = self.core_model(x_et, x_eeg, x_beh)
        return self.dequant(logits)


def ensure_dirs():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DICT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(payload):
    ensure_dirs()
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def model_factory():
    return T0_Baseline(dropout_fuse=0.5)


def to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def extract_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def make_fold_payload(train_idx, test_idx, X_et, X_eeg, X_beh, y, fold_i):
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

    et_tr_aug, eeg_tr_aug, beh_tr_aug, y_tr_aug = offline_gaussian_augment(
        et_tr,
        eeg_tr,
        beh_tr,
        y_tr,
        n_aug=GAUSSIAN_COPIES,
        seed=SEED + fold_i * 100 + 17,
        noise_std=D3_NOISE_STD_FACTOR,
    )

    rng_val = np.random.RandomState(SEED + fold_i)
    perm = rng_val.permutation(len(y_tr_aug))
    n_val = max(1, len(y_tr_aug) // 5)
    val_local = perm[:n_val]
    train_local = perm[n_val:]

    payload = {
        "et_train_all": et_tr_aug,
        "eeg_train_all": eeg_tr_aug,
        "beh_train_all": beh_tr_aug,
        "y_train_all": y_tr_aug,
        "et_train": et_tr_aug[train_local],
        "eeg_train": eeg_tr_aug[train_local],
        "beh_train": beh_tr_aug[train_local],
        "y_train": y_tr_aug[train_local],
        "et_val": et_tr_aug[val_local],
        "eeg_val": eeg_tr_aug[val_local],
        "beh_val": beh_tr_aug[val_local],
        "y_val": y_tr_aug[val_local],
        "et_test": et_te,
        "eeg_test": eeg_te,
        "beh_test": beh_te,
        "y_test": y_te,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    return payload


def build_train_loader(fold_payload, fold_i, batch_size=BATCH_SIZE):
    generator = torch.Generator()
    generator.manual_seed(SEED + fold_i)

    train_ds = TensorDataset(
        torch.tensor(fold_payload["et_train"]),
        torch.tensor(fold_payload["eeg_train"]),
        torch.tensor(fold_payload["beh_train"]),
        torch.tensor(fold_payload["y_train"], dtype=torch.long),
    )
    drop_last = len(train_ds) % batch_size == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )
    return train_loader


def build_val_tensors(fold_payload):
    return [
        torch.tensor(fold_payload["et_val"]),
        torch.tensor(fold_payload["eeg_val"]),
        torch.tensor(fold_payload["beh_val"]),
        torch.tensor(fold_payload["y_val"], dtype=torch.long),
    ]


def get_model_size_bytes(model, state_dict_only=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        if state_dict_only:
            torch.save(model.state_dict(), f.name)
        else:
            torch.save(model, f.name)
        size = os.path.getsize(f.name)
    os.unlink(f.name)
    return int(size)


def evaluate_model_cpu(model, et_np, eeg_np, beh_np, y_np, use_half=False):
    model.eval()
    with torch.no_grad():
        x_et = torch.tensor(et_np, device=CPU_DEVICE)
        x_eeg = torch.tensor(eeg_np, device=CPU_DEVICE)
        x_beh = torch.tensor(beh_np, device=CPU_DEVICE)
        if use_half:
            x_et = x_et.half()
            x_eeg = x_eeg.half()
            x_beh = x_beh.half()
        output = model(x_et, x_eeg, x_beh)
        logits = extract_logits(output)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float(accuracy_score(y_np, preds))
    return acc


def export_c_header(model, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        f.write("// DCM-DNN Model Weights for ESP32 Deployment\n")
        f.write(f"// Total parameters: {sum(p.numel() for p in model.parameters())}\n\n")
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().astype(np.float32).flatten()
            c_name = name.replace(".", "_")
            f.write(f"const float {c_name}[{len(data)}] = {{\n")
            for i in range(0, len(data), 8):
                chunk = data[i : i + 8]
                chunk_text = ", ".join(f"{float(v):.6f}f" for v in chunk)
                if i + 8 < len(data):
                    f.write(f"  {chunk_text},\n")
                else:
                    f.write(f"  {chunk_text}\n")
            f.write("};\n\n")


def tensor_nbytes(obj):
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, (list, tuple)):
        return int(sum(tensor_nbytes(v) for v in obj))
    if isinstance(obj, dict):
        return int(sum(tensor_nbytes(v) for v in obj.values()))
    return 0


def estimate_inference_ram_kb(model, sample_inputs):
    records = []
    hooks = []

    def hook_fn(_module, _inputs, outputs):
        records.append(tensor_nbytes(outputs))

    for module in model.modules():
        if module is model:
            continue
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        output = model(*sample_inputs)
        output_bytes = tensor_nbytes(output)

    for handle in hooks:
        handle.remove()

    input_bytes = sum(int(t.numel() * t.element_size()) for t in sample_inputs)
    max_activation = max(records) if records else 0
    total_estimated = input_bytes + output_bytes + max_activation
    return float(total_estimated / 1024.0)


def benchmark_latency(model, sample_single, sample_batch):
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            _ = model(*sample_single)
        single_times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            _ = model(*sample_single)
            single_times.append(time.perf_counter() - t0)

        for _ in range(30):
            _ = model(*sample_batch)
        batch_times = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = model(*sample_batch)
            batch_times.append(time.perf_counter() - t0)

    return {
        "cpu_single_ms": float(np.mean(single_times) * 1000.0),
        "cpu_single_std_ms": float(np.std(single_times) * 1000.0),
        "cpu_batch100_ms": float(np.mean(batch_times) * 1000.0),
        "cpu_batch100_std_ms": float(np.std(batch_times) * 1000.0),
    }


def summarize_quant_result(fold_accs, size_bytes=None, per_fold_sizes=None, skip_reasons=None):
    result = {
        "acc_mean": None,
        "acc_std": None,
        "model_size_kb": None,
        "fold_accs": [],
    }
    valid_accs = [float(v) for v in fold_accs if isinstance(v, (float, int))]
    result["fold_accs"] = [None if v is None else float(v) for v in fold_accs]
    if valid_accs:
        result["acc_mean"] = float(np.mean(valid_accs))
        result["acc_std"] = float(np.std(valid_accs))
    if size_bytes is not None:
        result["model_size_kb"] = float(size_bytes / 1024.0)
    if per_fold_sizes:
        result["fold_model_size_kb"] = [float(v / 1024.0) if v is not None else None for v in per_fold_sizes]
    if skip_reasons:
        result["skip_reasons"] = skip_reasons
    return result


def choose_deploy_candidate(results):
    for key in ["quantized_int8_static", "quantized_int8_dynamic", "quantized_fp16", "baseline_fp32"]:
        info = results.get(key, {})
        if isinstance(info, dict) and info.get("model_size_kb") is not None:
            return key, info
    return "baseline_fp32", results["baseline_fp32"]


def export_and_verify_onnx(core_model, fold_payload):
    result = {
        "path": str(ONNX_PATH),
        "model_size_kb": None,
        "acc_verified": False,
        "acc": None,
        "reason": None,
    }

    wrapper = LogitsOnlyWrapper(core_model).to(CPU_DEVICE).eval()
    dummy_et = torch.randn(1, ET_DIM, device=CPU_DEVICE)
    dummy_eeg = torch.randn(1, EEG_DIM, device=CPU_DEVICE)
    dummy_beh = torch.randn(1, BEH_DIM, device=CPU_DEVICE)

    try:
        torch.onnx.export(
            wrapper,
            (dummy_et, dummy_eeg, dummy_beh),
            str(ONNX_PATH),
            input_names=["et", "eeg", "beh"],
            output_names=["logits"],
            dynamic_axes={
                "et": {0: "batch"},
                "eeg": {0: "batch"},
                "beh": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=13,
        )
        result["model_size_kb"] = float(os.path.getsize(ONNX_PATH) / 1024.0)
    except Exception as exc:
        result["reason"] = f"ONNX export failed: {exc}"
        return result

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {
                "et": fold_payload["et_test"].astype(np.float32),
                "eeg": fold_payload["eeg_test"].astype(np.float32),
                "beh": fold_payload["beh_test"].astype(np.float32),
            },
        )
        preds = np.argmax(outputs[0], axis=1)
        acc = float(accuracy_score(fold_payload["y_test"], preds))
        result["acc_verified"] = True
        result["acc"] = acc
    except ImportError:
        result["reason"] = "onnxruntime not installed, skipping ONNX verification"
    except Exception as exc:
        result["reason"] = f"ONNX verification failed: {exc}"

    return result


def run_static_quantization(core_model, fold_payload):
    if "fbgemm" not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"Current PyTorch does not support fbgemm, supported={torch.backends.quantized.supported_engines}")

    torch.backends.quantized.engine = "fbgemm"
    wrapped = StaticQuantWrapper(copy.deepcopy(core_model).to(CPU_DEVICE).eval())
    wrapped.qconfig = quant.get_default_qconfig("fbgemm")
    prepared = quant.prepare(wrapped, inplace=False)

    calib_count = min(128, len(fold_payload["et_train_all"]))
    calib_ds = TensorDataset(
        torch.tensor(fold_payload["et_train_all"][:calib_count], dtype=torch.float32),
        torch.tensor(fold_payload["eeg_train_all"][:calib_count], dtype=torch.float32),
        torch.tensor(fold_payload["beh_train_all"][:calib_count], dtype=torch.float32),
    )
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)

    prepared.eval()
    with torch.no_grad():
        for batch in calib_loader:
            prepared(batch[0], batch[1], batch[2])

    converted = quant.convert(prepared, inplace=False).eval()
    acc = evaluate_model_cpu(
        converted,
        fold_payload["et_test"],
        fold_payload["eeg_test"],
        fold_payload["beh_test"],
        fold_payload["y_test"],
        use_half=False,
    )
    size_bytes = get_model_size_bytes(converted)
    return acc, size_bytes


def init_output_template(X_et, X_eeg, X_beh, y, subject_ids):
    return {
        "experiment": "DCM-DNN Model Quantization and ESP32 Deployment Feasibility Study",
        "source_script_alignment": "Strictly copied complete model code and data loading logic from run_neuma_41subj_combo.py (C0 baseline)",
        "data_source": str(NPZ_PATH),
        "result_path": str(RESULT_PATH),
        "artifact_dir": str(ARTIFACT_DIR),
        "device": {
            "train_device": str(DEVICE),
            "quant_eval_device": "cpu",
            "python_executable_expected": r"D:\soft\anaconda_env\envs\py39\python.exe",
        },
        "seed": SEED,
        "cv_strategy": "StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(subject_ids))),
        "label_distribution": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "input_features": {
            "X_et": int(X_et.shape[1]),
            "X_eeg_c1": int(X_eeg.shape[1]),
            "X_beh": int(X_beh.shape[1]),
            "combined_dim": int(X_et.shape[1] + X_eeg.shape[1] + X_beh.shape[1]),
        },
        "hyperparams": {
            "config": "C0",
            "dropout_fuse": 0.5,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "gaussian_copies": GAUSSIAN_COPIES,
            "gaussian_noise_std_factor": D3_NOISE_STD_FACTOR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "lr_patience": LR_PATIENCE,
            "max_epochs": MAX_EPOCHS,
        },
        "baseline_target_acc": BASELINE_TARGET_ACC,
        "baseline_fp32": {"acc_mean": None, "acc_std": None, "model_size_kb": None, "fold_accs": []},
        "quantized_fp16": {"acc_mean": None, "acc_std": None, "model_size_kb": None, "fold_accs": []},
        "quantized_int8_dynamic": {"acc_mean": None, "acc_std": None, "model_size_kb": None, "fold_accs": []},
        "quantized_int8_static": {"acc_mean": None, "acc_std": None, "model_size_kb": None, "fold_accs": []},
        "onnx": {"model_size_kb": None, "acc_verified": False},
        "latency": {"cpu_single_ms": None, "cpu_batch100_ms": None},
        "esp32_assessment": {},
        "c_header_path": str(C_HEADER_PATH),
        "comparison_with_sota": {},
        "fold_artifacts": [],
        "notes": [],
        "host_info": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pc_clock_mhz_assumed": PC_CLOCK_MHZ_ASSUMED,
            "esp32_overhead_factor": ESP32_OVERHEAD_FACTOR,
        },
    }


def main():
    ensure_dirs()
    (X_et, X_eeg, X_beh, y, subject_ids, _X_et_raw, _X_eeg_raw, _X_beh_raw) = load_data()
    output = init_output_template(X_et, X_eeg, X_beh, y, subject_ids)
    save_json(to_serializable(output))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    baseline_fold_accs = []
    baseline_size_bytes = []
    fp16_fold_accs = []
    fp16_size_bytes = []
    fp16_skip_reasons = []
    int8_dyn_fold_accs = []
    int8_dyn_size_bytes = []
    int8_dyn_skip_reasons = []
    int8_static_fold_accs = []
    int8_static_size_bytes = []
    int8_static_skip_reasons = []

    representative_model = None
    representative_payload = None
    representative_state_path = None

    print("\n" + "=" * 96)
    print("Phase 1/2: Training full model and running per-fold quantization accuracy tests")
    print("=" * 96)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_et, y), start=1):
        print(f"\n[Fold {fold_i:02d}] Start")
        fold_payload = make_fold_payload(train_idx, test_idx, X_et, X_eeg, X_beh, y, fold_i)
        train_loader = build_train_loader(fold_payload, fold_i, batch_size=BATCH_SIZE)
        val_tensors = build_val_tensors(fold_payload)

        set_seed(SEED + fold_i)
        model = model_factory().to(DEVICE)
        model = train_model(
            model,
            train_loader,
            val_tensors,
            fold_payload["y_train"],
            name=f"C0-TinyML-F{fold_i}",
            lr=LR,
            patience=PATIENCE,
            label_smoothing=0.0,
        )

        model.eval()
        with torch.no_grad():
            logits, _ = model(
                torch.tensor(fold_payload["et_test"], device=DEVICE),
                torch.tensor(fold_payload["eeg_test"], device=DEVICE),
                torch.tensor(fold_payload["beh_test"], device=DEVICE),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        baseline_acc = float(accuracy_score(fold_payload["y_test"], preds))
        baseline_fold_accs.append(baseline_acc)

        model_cpu = copy.deepcopy(model).to(CPU_DEVICE).eval()
        state_path = STATE_DICT_DIR / f"fold_{fold_i:02d}_fp32_state_dict.pt"
        torch.save(model_cpu.state_dict(), state_path)
        base_size = os.path.getsize(state_path)
        baseline_size_bytes.append(int(base_size))

        if representative_model is None:
            representative_model = copy.deepcopy(model_cpu)
            representative_payload = fold_payload
            representative_state_path = state_path

        fold_record = {
            "fold": fold_i,
            "state_dict_path": str(state_path),
            "train_size": int(len(fold_payload["y_train"])),
            "val_size": int(len(fold_payload["y_val"])),
            "test_size": int(len(fold_payload["y_test"])),
            "baseline_fp32_acc": baseline_acc,
        }
        print(f"[Fold {fold_i:02d}] FP32 Acc={baseline_acc:.4f} | state_dict={state_path.name}")

        try:
            model_fp16 = copy.deepcopy(model_cpu).half().eval()
            fp16_acc = evaluate_model_cpu(
                model_fp16,
                fold_payload["et_test"],
                fold_payload["eeg_test"],
                fold_payload["beh_test"],
                fold_payload["y_test"],
                use_half=True,
            )
            fp16_size = get_model_size_bytes(model_fp16)
            fp16_fold_accs.append(fp16_acc)
            fp16_size_bytes.append(fp16_size)
            fold_record["fp16_acc"] = fp16_acc
            print(f"[Fold {fold_i:02d}] FP16 Acc={fp16_acc:.4f}")
        except Exception as exc:
            reason = f"fold_{fold_i:02d}: {exc}"
            fp16_fold_accs.append(None)
            fp16_size_bytes.append(None)
            fp16_skip_reasons.append(reason)
            fold_record["fp16_error"] = str(exc)
            print(f"[Fold {fold_i:02d}] FP16 skipped: {exc}")

        try:
            model_int8_dynamic = quant.quantize_dynamic(copy.deepcopy(model_cpu), {nn.Linear}, dtype=torch.qint8)
            model_int8_dynamic.eval()
            dyn_acc = evaluate_model_cpu(
                model_int8_dynamic,
                fold_payload["et_test"],
                fold_payload["eeg_test"],
                fold_payload["beh_test"],
                fold_payload["y_test"],
                use_half=False,
            )
            dyn_size = get_model_size_bytes(model_int8_dynamic)
            int8_dyn_fold_accs.append(dyn_acc)
            int8_dyn_size_bytes.append(dyn_size)
            fold_record["int8_dynamic_acc"] = dyn_acc
            print(f"[Fold {fold_i:02d}] INT8 Dynamic Acc={dyn_acc:.4f}")
        except Exception as exc:
            reason = f"fold_{fold_i:02d}: {exc}"
            int8_dyn_fold_accs.append(None)
            int8_dyn_size_bytes.append(None)
            int8_dyn_skip_reasons.append(reason)
            fold_record["int8_dynamic_error"] = str(exc)
            print(f"[Fold {fold_i:02d}] INT8 Dynamic skipped: {exc}")

        try:
            static_acc, static_size = run_static_quantization(model_cpu, fold_payload)
            int8_static_fold_accs.append(static_acc)
            int8_static_size_bytes.append(static_size)
            fold_record["int8_static_acc"] = static_acc
            print(f"[Fold {fold_i:02d}] INT8 Static Acc={static_acc:.4f}")
        except Exception as exc:
            reason = f"fold_{fold_i:02d}: {exc}"
            int8_static_fold_accs.append(None)
            int8_static_size_bytes.append(None)
            int8_static_skip_reasons.append(reason)
            fold_record["int8_static_error"] = str(exc)
            print(f"[Fold {fold_i:02d}] INT8 Static skipped: {exc}")

        output["fold_artifacts"].append(fold_record)
        output["baseline_fp32"] = summarize_quant_result(
            baseline_fold_accs,
            size_bytes=int(np.mean(baseline_size_bytes)) if baseline_size_bytes else None,
            per_fold_sizes=baseline_size_bytes,
        )
        output["baseline_fp32"]["fold_state_dict_paths"] = [item["state_dict_path"] for item in output["fold_artifacts"]]
        output["baseline_fp32"]["baseline_matches_expected"] = abs(output["baseline_fp32"]["acc_mean"] - BASELINE_TARGET_ACC) < 0.01 if output["baseline_fp32"]["acc_mean"] is not None else False
        output["quantized_fp16"] = summarize_quant_result(
            fp16_fold_accs,
            size_bytes=int(np.mean([v for v in fp16_size_bytes if v is not None])) if any(v is not None for v in fp16_size_bytes) else None,
            per_fold_sizes=fp16_size_bytes,
            skip_reasons=fp16_skip_reasons,
        )
        output["quantized_int8_dynamic"] = summarize_quant_result(
            int8_dyn_fold_accs,
            size_bytes=int(np.mean([v for v in int8_dyn_size_bytes if v is not None])) if any(v is not None for v in int8_dyn_size_bytes) else None,
            per_fold_sizes=int8_dyn_size_bytes,
            skip_reasons=int8_dyn_skip_reasons,
        )
        output["quantized_int8_static"] = summarize_quant_result(
            int8_static_fold_accs,
            size_bytes=int(np.mean([v for v in int8_static_size_bytes if v is not None])) if any(v is not None for v in int8_static_size_bytes) else None,
            per_fold_sizes=int8_static_size_bytes,
            skip_reasons=int8_static_skip_reasons,
        )
        save_json(to_serializable(output))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("Phase 3/4/5/6/7: Model size, ONNX export, latency, ESP32 assessment, C header export")
    print("=" * 96)

    if representative_model is None or representative_payload is None:
        raise RuntimeError("No trained fold model obtained, cannot proceed to subsequent phases")

    export_c_header(representative_model, C_HEADER_PATH)
    print(f"[Export] C header: {C_HEADER_PATH}")

    onnx_info = export_and_verify_onnx(representative_model, representative_payload)
    output["onnx"] = onnx_info
    if onnx_info.get("reason"):
        output["notes"].append(onnx_info["reason"])
    print(f"[ONNX] {onnx_info}")

    latency_model = LogitsOnlyWrapper(copy.deepcopy(representative_model).to(CPU_DEVICE).eval())
    sample_single = (
        torch.tensor(representative_payload["et_test"][:1], dtype=torch.float32),
        torch.tensor(representative_payload["eeg_test"][:1], dtype=torch.float32),
        torch.tensor(representative_payload["beh_test"][:1], dtype=torch.float32),
    )
    batch_n = min(100, len(representative_payload["et_test"]))
    sample_batch = (
        torch.tensor(representative_payload["et_test"][:batch_n], dtype=torch.float32),
        torch.tensor(representative_payload["eeg_test"][:batch_n], dtype=torch.float32),
        torch.tensor(representative_payload["beh_test"][:batch_n], dtype=torch.float32),
    )
    latency_info = benchmark_latency(latency_model, sample_single, sample_batch)
    output["latency"] = latency_info
    print(f"[Latency] {latency_info}")

    inference_ram_kb = estimate_inference_ram_kb(latency_model, sample_single)
    deploy_format, deploy_info = choose_deploy_candidate(output)
    deploy_model_size_kb = float(deploy_info.get("model_size_kb") or output["baseline_fp32"]["model_size_kb"] or 0.0)
    esp32_latency_ms = float(
        latency_info["cpu_single_ms"] * (PC_CLOCK_MHZ_ASSUMED / ESP32_SPECS["clock_mhz"]) * ESP32_OVERHEAD_FACTOR
    )
    output["esp32_assessment"] = {
        "deployment_candidate": deploy_format,
        "esp32_specs": ESP32_SPECS,
        "model_size_kb": deploy_model_size_kb,
        "estimated_inference_ram_kb": inference_ram_kb,
        "model_fits_flash": deploy_model_size_kb <= ESP32_SPECS["flash_kb"],
        "model_fits_sram": inference_ram_kb <= ESP32_SPECS["sram_kb"],
        "estimated_latency_ms": esp32_latency_ms,
        "flash_usage_percent": float(deploy_model_size_kb / ESP32_SPECS["flash_kb"] * 100.0),
        "sram_usage_percent": float(inference_ram_kb / ESP32_SPECS["sram_kb"] * 100.0),
        "assumptions": {
            "pc_clock_mhz_assumed": PC_CLOCK_MHZ_ASSUMED,
            "esp32_overhead_factor": ESP32_OVERHEAD_FACTOR,
            "latency_formula": "cpu_single_ms × (PC_freq / ESP32_freq) × overhead_factor",
        },
    }

    fp32_kb = float(output["baseline_fp32"]["model_size_kb"] or 0.0)
    sota_low_mb = 50.0
    sota_high_mb = 100.0
    if fp32_kb > 0:
        low_ratio = (sota_low_mb * 1024.0) / fp32_kb
        high_ratio = (sota_high_mb * 1024.0) / fp32_kb
        ratio_text = f"~{low_ratio:.1f}x to ~{high_ratio:.1f}x smaller"
    else:
        ratio_text = None
    output["comparison_with_sota"] = {
        "dcm_dnn_size_kb": fp32_kb,
        "sota_estimated_size_mb": "~50-100MB (CNN+LSTM+LeNet Stacking)",
        "compression_ratio": ratio_text,
    }

    output["baseline_fp32"]["representative_state_dict_path"] = str(representative_state_path)
    output["baseline_fp32"]["target_acc"] = BASELINE_TARGET_ACC
    output["baseline_fp32"]["target_acc_delta"] = (
        float(output["baseline_fp32"]["acc_mean"] - BASELINE_TARGET_ACC) if output["baseline_fp32"]["acc_mean"] is not None else None
    )
    output["c_header_path"] = str(C_HEADER_PATH)

    save_json(to_serializable(output))

    print("\n" + "=" * 96)
    print("Experiment complete")
    print(f"[Output] JSON: {RESULT_PATH}")
    print(f"[Output] ONNX: {ONNX_PATH}")
    print(f"[Output] Header: {C_HEADER_PATH}")
    print("=" * 96)


if __name__ == "__main__":
    main()
