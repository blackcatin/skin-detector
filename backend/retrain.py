import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

# ==========================================
# CONFIG PATH
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "trained_models")
RETRAIN_DIR = os.path.join(BASE_DIR, "uploads", "retraining")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")  # 15K data lama

BATCH_SIZE = 16
EPOCHS     = 3
LR         = 0.0001

device = torch.device("cpu")

# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Mapping nama folder disk → nama kelas resmi
FOLDER_TO_LABEL = {
    "Acne and Rosacea Photos":                                        "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis Photos":                                       "Atopic Dermatitis",
    "Bullous Disease Photos":                                         "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial Infections":             "Cellulitis, Impetigo & other Bacterial Infections",
    "Eczema Photos":                                                  "Eczema",
    "Exanthems and Drug Eruptions":                                   "Exanthems & Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases":              "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes HPV and other STDs Photos":                               "Herpes, HPV & other STDs",
    "Light Diseases and Disorders of Pigmentation":                   "Light Diseases & Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases":                     "Lupus & other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles":                            "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus and other Nail Disease":                             "Nail Fungus & other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis":                 "Poison Ivy & other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases":          "Psoriasis, Lichen Planus & related diseases",
    "Scabies Lyme Disease and other Infestations and Bites":          "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses and other Benign Tumors":                   "Seborrheic Keratoses & other Benign Tumors",
    "Systemic Disease":                                               "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections":         "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria Hives":                                                "Urticaria / Hives",
    "Vascular Tumors":                                                "Vascular Tumors",
    "Vasculitis Photos":                                              "Vasculitis",
    "Warts Molluscum and other Viral Infections":                     "Warts, Molluscum & other Viral Infections",
}

# Dari nama kelas resmi → index (untuk training)
CLASS_LABELS = list(FOLDER_TO_LABEL.values()) + ["Systemic Disease"]
# Deduplikasi dan urutkan sesuai urutan asli
CLASS_LABELS = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis, Impetigo & other Bacterial Infections",
    "Eczema",
    "Exanthems & Drug Eruptions",
    "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes, HPV & other STDs",
    "Light Diseases & Disorders of Pigmentation",
    "Lupus & other Connective Tissue diseases",
    "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus & other Nail Disease",
    "Poison Ivy & other Contact Dermatitis",
    "Psoriasis, Lichen Planus & related diseases",
    "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses & other Benign Tumors",
    "Systemic Disease",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria / Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Warts, Molluscum & other Viral Infections"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# ==========================================
# SAMPLING DARI FOLDER LOKAL (DATA LAMA)
# Ambil stratified sample: N gambar per kelas
# ==========================================
def sample_from_local_dataset(samples_per_class: int = 10) -> list:
    records = []

    if not os.path.exists(DATASET_DIR):
        print(f"⚠️ DATASET_DIR tidak ditemukan: {DATASET_DIR}")
        return records

    for folder_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, folder_name)

        if not os.path.isdir(class_dir):
            continue

        # Cari nama kelas resmi dari nama folder
        class_name = FOLDER_TO_LABEL.get(folder_name)
        if not class_name:
            print(f"⚠️ Kelas tidak dikenal, dilewati: {folder_name}")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            continue

        sampled = random.sample(image_files, min(samples_per_class, len(image_files)))

        for fname in sampled:
            records.append({
                "image_path":      os.path.join(class_dir, fname),
                "predicted_class": class_name,  # ← nama resmi, bukan nama folder
                "source":          "stratified"
            })

    print(f"📂 Stratified dari lokal: {len(records)} gambar ({samples_per_class} per kelas)")
    return records

# ==========================================
# CUSTOM DATASET
# ==========================================
class SubsetDataset(Dataset):
    """
    Menerima list dict:
    [{"image_path": "...", "predicted_class": "...", "source": "new/hard/stratified"}]
    """
    def __init__(self, records: list):
        valid = []
        for r in records:
            if os.path.exists(r["image_path"]):
                valid.append(r)
        skipped = len(records) - len(valid)
        if skipped > 0:
            print(f"⚠️ {skipped} file tidak ditemukan, dilewati.")
        self.records = valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img = transform(img)
        except Exception as e:
            print(f"⚠️ Gagal baca gambar: {rec['image_path']} — {e}")
            img = torch.zeros(3, 224, 224)

        label  = LABEL_TO_IDX.get(rec["predicted_class"], 0)
        source = rec.get("source", "stratified")
        return img, label, source


def build_weighted_loader(records: list) -> DataLoader:
    """
    Data baru dan hard examples diberi bobot 3x
    supaya model lebih fokus belajar dari data sulit.
    """
    dataset = SubsetDataset(records)

    if len(dataset) == 0:
        raise Exception("Semua file gambar tidak ditemukan. Training dibatalkan.")

    weights = [
        3.0 if r.get("source") in ("new", "hard") else 1.0
        for r in dataset.records
    ]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


# ==========================================
# TRAINING ENGINE
# ==========================================
def run_pytorch_training(model, model_save_path, subset_records: list):

    # ── Ambil stratified sample dari folder lokal ──
    local_samples = sample_from_local_dataset(samples_per_class=10)

    # ── Gabungkan: data baru (DB) + data lama (lokal) ──
    all_records = subset_records + local_samples

    print(f"\n🏋️ [PyTorch Engine] Memulai fine-tuning ResNet18...")
    print(f"📊 Total dataset training: {len(all_records)} data")
    print(f"   ↳ Baru (DB)    : {sum(1 for r in all_records if r.get('source') == 'new')}")
    print(f"   ↳ Hard (DB)    : {sum(1 for r in all_records if r.get('source') == 'hard')}")
    print(f"   ↳ Stratified   : {sum(1 for r in all_records if r.get('source') == 'stratified')}")

    loader = build_weighted_loader(all_records)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    model.train()

    final_loss     = 0.0
    final_accuracy = 0.0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct      = 0
        total        = 0

        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            running_loss += loss.item()

        avg_loss       = running_loss / max(1, len(loader))
        acc            = correct / max(1, total)
        final_loss     = avg_loss
        final_accuracy = acc

        print(f"📈 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    scheduler.step()

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved: {model_save_path}")

    model.eval()
    
    return {
        "accuracy":           round(final_accuracy, 4),
        "final_loss":         round(final_loss, 4),
        "total_data_retrain": len(all_records),
        "epochs":             EPOCHS
    }