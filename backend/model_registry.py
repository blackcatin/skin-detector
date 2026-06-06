import os
import json
from datetime import datetime
import shutil

# ==========================================
# PATH CONFIG
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
STAGING_DIR = os.path.join(MODEL_DIR, "staging")
ACTIVE_DIR = os.path.join(MODEL_DIR, "active")
REJECTED_DIR = os.path.join(MODEL_DIR, "rejected")

CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")
REGISTRY_JSON_FILE = os.path.join(CURRENT_DIR, "model_history.json")

DEFAULT_MODEL = "model_v1.pth"


# ==========================================
# INIT FOLDERS
# ==========================================
for d in [MODEL_DIR, STAGING_DIR, ACTIVE_DIR, REJECTED_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================
# UTIL
# ==========================================
def load_history():
    if not os.path.exists(REGISTRY_JSON_FILE):
        return []
    try:
        with open(REGISTRY_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    with open(REGISTRY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def update_pointer(model_filename):
    with open(CURRENT_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(model_filename)


# ==========================================
# CORE SAFE REGISTRY
# ==========================================
def update_model_registry(
    new_model_path,
    version,
    accuracy,
    final_loss,
    total_data_retrain,
    execution_time,
    upload_success=False
):
    history = load_history()
    filename = os.path.basename(new_model_path)

    try:
        acc = float(str(accuracy).replace("%", ""))
    except:
        acc = 0.0

    best_accuracy = max(
        [float(str(h.get("accuracy", "0")).replace("%", "")) for h in history],
        default=0.0
    )

    # ── 1. Tentukan status ──────────────────
    is_best   = acc > best_accuracy
    status    = "STAGING_BEST" if is_best else "REJECTED"
    target_dir = STAGING_DIR if is_best else REJECTED_DIR

    # ── 2. Pindahkan file (tidak dihapus) ───
    new_location = os.path.join(target_dir, filename)
    if os.path.exists(new_model_path):
        shutil.move(new_model_path, new_location)

    # ── 3. Promote jika upload sukses ───────
    if is_best and upload_success:
        active_path = os.path.join(ACTIVE_DIR, filename)
        shutil.move(new_location, active_path)
        update_pointer(filename)
        status = "ACTIVE_PRODUCTION"

    # ── 4. Simpan history ───────────────────
    entry = {
        "version":            version,
        "model_path":         filename,
        "accuracy":           f"{acc}%",
        "final_loss":         final_loss,
        "total_data_retrain": total_data_retrain,
        "execution_time":     f"{execution_time}s",
        "status":             status,
        "upload_success":     upload_success,
        "is_best_model":      is_best,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history.append(entry)
    save_history(history)

    return entry  # ← sekarang selalu tercapai


# ── Compat layer — letakkan SETELAH fungsi utama ──
def update_model_path(
    new_model_path,
    version,
    accuracy,
    final_loss,
    total_data_retrain,
    execution_time,
    upload_success=False
):
    return update_model_registry(
        new_model_path=new_model_path,
        version=version,
        accuracy=accuracy,
        final_loss=final_loss,
        total_data_retrain=total_data_retrain,
        execution_time=execution_time,
        upload_success=upload_success
    )


# ==========================================
# GET CURRENT MODEL
# ==========================================
def get_current_model_path():
    if not os.path.exists(CURRENT_MODEL_FILE):
        # Fallback: cari file .pth terbaru langsung di trained_models/
        pth_files = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and os.path.isfile(os.path.join(MODEL_DIR, f))],
            reverse=True
        )
        if pth_files:
            return os.path.join(MODEL_DIR, pth_files[0])
        return os.path.join(ACTIVE_DIR, DEFAULT_MODEL)

    try:
        with open(CURRENT_MODEL_FILE, "r") as f:
            name = f.read().strip()

        # Pastikan ada ekstensi .pth
        if not name.endswith(".pth"):
            name = name + ".pth"

        # Cek di ACTIVE_DIR dulu, lalu fallback ke MODEL_DIR langsung
        active_path = os.path.join(ACTIVE_DIR, name)
        if os.path.exists(active_path):
            return active_path

        direct_path = os.path.join(MODEL_DIR, name)
        if os.path.exists(direct_path):
            return direct_path

        return active_path  # biarkan app.py handle jika tidak ditemukan

    except:
        return os.path.join(ACTIVE_DIR, DEFAULT_MODEL)

def get_current_model_version_name():
    """
    Ambil versi model aktif dari current_model.txt
    """
    try:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
        CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")

        if not os.path.exists(CURRENT_MODEL_FILE):
            return "model_v1"

        with open(CURRENT_MODEL_FILE, "r", encoding="utf-8") as f:
            filename = f.read().strip()

        # model_v3.pth → model_v3
        return filename.replace(".pth", "")
    except:
        return "model_v1"