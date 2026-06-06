"""
Delete model files from Hugging Face Hub
Usage: python delete_hf_models.py
"""

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIG — sesuaikan di sini
# ==========================================
HF_TOKEN  = os.getenv("HF_TOKEN")          # dari .env, atau isi manual di bawah
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"     # ← uncomment jika mau hardcode sementara

REPO_ID   = os.getenv("HF_REPO_ID", "blackcatin/resnet18-syifai")
REPO_TYPE = "model"

# File-file yang mau dihapus
FILES_TO_DELETE = [
    "Fix_best_model_v3.pth",
    "Fix_best_model_v4.pth",
    "Fix_best_model_v5.pth",
    # Tambah lagi sesuai kebutuhan:
    # "Fix_best_model_v6.pth",
]

# ==========================================
# MAIN
# ==========================================
def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN tidak ditemukan. Set di .env atau hardcode di script.")
        return

    api = HfApi(token=HF_TOKEN)

    # Cek dulu file apa saja yang ada di repo
    print(f"\n📦 Repo: {REPO_ID}")
    print("📂 File yang ada di repo saat ini:")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        existing = list(files)
        for f in existing:
            print(f"   - {f}")
    except Exception as e:
        print(f"❌ Gagal list repo: {e}")
        return

    print(f"\n🗑️  File yang akan dihapus:")
    to_delete = [f for f in FILES_TO_DELETE if f in existing]
    not_found = [f for f in FILES_TO_DELETE if f not in existing]

    for f in to_delete:
        print(f"   ✅ {f}")
    for f in not_found:
        print(f"   ⚠️  {f} — tidak ditemukan di repo, dilewati")

    if not to_delete:
        print("\n💡 Tidak ada file yang perlu dihapus.")
        return

    confirm = input(f"\n❓ Hapus {len(to_delete)} file dari {REPO_ID}? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Dibatalkan.")
        return

    # Hapus satu per satu
    success = 0
    for filename in to_delete:
        try:
            api.delete_file(
                path_in_repo   = filename,
                repo_id        = REPO_ID,
                repo_type      = REPO_TYPE,
                commit_message = f"Delete old experiment model: {filename}"
            )
            print(f"🗑️  Deleted: {filename}")
            success += 1
        except Exception as e:
            print(f"❌ Gagal hapus {filename}: {e}")

    print(f"\n✅ Selesai — {success}/{len(to_delete)} file berhasil dihapus.")

if __name__ == "__main__":
    main()