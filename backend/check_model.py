import torch

path = "Fix_best_model.pth"

try:
    ckpt = torch.load(path, map_location="cpu")
    print("✅ File berhasil dibaca.")
    print("Tipe data:", type(ckpt))

    if isinstance(ckpt, dict):
        print("🔍 Ini kemungkinan besar state_dict.")
        print("Keys:", list(ckpt.keys())[:10])
    else:
        print("🔍 Ini kemungkinan besar full model:", ckpt.__class__)
except Exception as e:
    print("❌ Error:", e)
