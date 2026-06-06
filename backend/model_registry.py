import os
import json
import shutil
from datetime import datetime

# ==========================================
# PATH CONFIG
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR    = os.path.join(CURRENT_DIR, "trained_models")
ACTIVE_DIR   = os.path.join(MODEL_DIR, "active")
REJECTED_DIR = os.path.join(MODEL_DIR, "rejected")

CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")
REGISTRY_JSON_FILE = os.path.join(CURRENT_DIR, "model_history.json")

DEFAULT_MODEL = "Fix_best_model_v1.pth"

for d in [MODEL_DIR, ACTIVE_DIR, REJECTED_DIR]:
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
    print(f"📌 current_model.txt diupdate → {model_filename}")


def get_current_active_filename() -> str:
    if not os.path.exists(CURRENT_MODEL_FILE):
        return DEFAULT_MODEL
    try:
        with open(CURRENT_MODEL_FILE, "r") as f:
            name = f.read().strip()
        if not name.endswith(".pth"):
            name += ".pth"
        return name
    except:
        return DEFAULT_MODEL


# ==========================================
# CORE REGISTRY
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
    history  = load_history()
    filename = os.path.basename(new_model_path)

    try:
        acc = float(str(accuracy).replace("%", ""))
    except:
        acc = 0.0

    best_accuracy = max(
        [float(str(h.get("accuracy", "0")).replace("%", "")) for h in history],
        default=0.0
    )

    is_best = acc > best_accuracy

    # Tentukan old_filename SEBELUM blok if-else
    old_filename = get_current_active_filename() if is_best else None

    if is_best:
        # Pindahkan model baru ke active/
        active_path = os.path.join(ACTIVE_DIR, filename)
        if os.path.exists(new_model_path):
            shutil.copy2(new_model_path, active_path)
            os.remove(new_model_path)
            print(f"✅ Model baru disimpan ke active/: {filename}")

        # Update pointer
        update_pointer(filename)
        status = "ACTIVE"

    else:
        # Model ditolak → simpan ke rejected/
        rejected_path = os.path.join(REJECTED_DIR, filename)
        if os.path.exists(new_model_path):
            shutil.move(new_model_path, rejected_path)
            print(f"❌ Model ditolak, disimpan ke rejected/: {filename}")
        status = "REJECTED"

    # Buat entry SETELAH semua variabel siap
    entry = {
        "version":                   version,
        "model_path":                filename,
        "accuracy":                  f"{acc}%",
        "baseline_accuracy":       f"{best_accuracy}%",
        "final_loss":                final_loss,
        "total_data_retrain":        total_data_retrain,
        "execution_time":            f"{execution_time}s",
        "status":                    status,
        "is_best_model":             is_best,
        "previous_model_filename":   old_filename,
        "timestamp":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history.append(entry)
    save_history(history)

    return entry


# ==========================================
# GET CURRENT MODEL PATH
# ==========================================
def get_current_model_path():
    filename = get_current_active_filename()

    active_path = os.path.join(ACTIVE_DIR, filename)
    if os.path.exists(active_path):
        return active_path

    if os.path.exists(ACTIVE_DIR):
        pth_files = sorted(
            [f for f in os.listdir(ACTIVE_DIR) if f.endswith(".pth")],
            reverse=True
        )
        if pth_files:
            update_pointer(pth_files[0])
            return os.path.join(ACTIVE_DIR, pth_files[0])

    direct_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path

    return active_path


def get_current_model_version_name():
    try:
        return get_current_active_filename().replace(".pth", "")
    except:
        return "model_v1"


# ==========================================
# COMPAT LAYER
# ==========================================
def update_model_path(
    new_model_path, version, accuracy, final_loss,
    total_data_retrain, execution_time, upload_success=False
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