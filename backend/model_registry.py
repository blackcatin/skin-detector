import os
import json
from datetime import datetime

# Mengunci path secara dinamis agar folder trained_models berada di jalur yang benar
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")
REGISTRY_JSON_FILE = os.path.join(CURRENT_DIR, "model_history.json")
DEFAULT_MODEL = "model_v1.pth"

def get_model_history():
    """Mengambil seluruh riwayat latihan model dari file JSON."""
    if not os.path.exists(REGISTRY_JSON_FILE):
        return []
    try:
        with open(REGISTRY_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def get_current_model_path():
    """
    Mengambil path model aktif saat ini (Production).
    Jika file pointer belum ada, otomatis beralih ke model awal (model_v1.pth).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(CURRENT_MODEL_FILE):
        return os.path.join(MODEL_DIR, DEFAULT_MODEL)
    
    try:
        with open(CURRENT_MODEL_FILE, "r", encoding="utf-8") as f:
            model_name = f.read().strip()
        
        full_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(full_path):
            return full_path
        return os.path.join(MODEL_DIR, DEFAULT_MODEL)
    except Exception:
        return os.path.join(MODEL_DIR, DEFAULT_MODEL)

def update_current_pointer(filename_pth):
    """Mengubah isi file current_model.txt untuk menunjuk ke otak model yang aktif."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CURRENT_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(filename_pth)

def update_model_path(new_model_path, version, accuracy, final_loss, total_data_retrain, execution_time):
    """
    Logika Inti MLOps: Menyaring model terbaik.
    Membandingkan akurasi model baru dengan riwayat masa lalu.
    Jika gagal, file fisik .pth langsung dimusnahkan demi menghemat storage lokal.
    """
    history = get_model_history()
    filename_new = os.path.basename(new_model_path)
    
    # 1. Cari berapa nilai akurasi tertinggi (Best Accuracy) dari riwayat masa lalu
    best_accuracy_so_far = 0.0
    for item in history:
        try:
            acc_val = float(str(item.get("accuracy", "0")).replace("%", ""))
            if acc_val > best_accuracy_so_far:
                best_accuracy_so_far = acc_val
        except ValueError:
            continue

    # Konversi akurasi model baru yang dikirim dari proses training
    try:
        current_acc_float = float(str(accuracy).replace("%", ""))
    except ValueError:
        current_acc_float = 74.20  # Fallback baseline jika terjadi kegagalan baca
    
    # 2. SELEKSI KUALITAS: Apakah model baru ini adalah yang TERBAIK?
    is_best_model = False
    
    if len(history) == 0 or current_acc_float > best_accuracy_so_far:
        is_best_model = True
        # Naik pangkat jadi Production: Update pointer agar web menggunakan file .pth baru ini
        update_current_pointer(filename_new)
        status_label = "Production (Best)"
        print(f"🏆 [MLOps Registry] Model Baru {version} LEBIH AKURAT ({current_acc_float}%)! Diaktifkan ke Production.")
    else:
        status_label = "Staging (Rejected & Deleted)"
        print(f"⚠️ [MLOps Registry] Model Baru {version} gagal mengalahkan rekor terbaik ({best_accuracy_so_far}%).")
        
        # 🔥 CRITICAL CLEANUP: Hapus file fisik .pth yang gagal agar tidak menumpuk di server lokal!
        if os.path.exists(new_model_path):
            try:
                os.remove(new_model_path)
                print(f"🗑️ [Zero-Storage] Berkas gagal {filename_new} seberat ~44MB telah dimusnahkan dari lokal server.")
            except Exception as e:
                print(f"⚠️ Gagal menghapus berkas model cadangan: {e}")

    # 3. Buat catatan entri log baru yang terstruktur untuk JSON history
    new_entry = {
        "version": version,
        "model_path": filename_new, # Simpan nama filenya saja agar ringkas
        "accuracy": f"{current_acc_float}%",
        "final_loss": final_loss,
        "total_data_retrain": total_data_retrain,
        "execution_time": f"{execution_time}s",
        "retrained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status_label
    }
    
    history.append(new_entry)
    
    # 4. Simpan kembali ke berkas model_history.json
    with open(REGISTRY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    return new_entry

if __name__ == "__main__":
    print("=== Model Registry Test ===")
    print("Model aktif saat ini:", get_current_model_path())