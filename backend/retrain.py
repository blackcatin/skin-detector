import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random  

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from torchvision import models

# ==========================================
# CONFIG PATH
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "trained_models")
RETRAIN_DIR = os.path.join(BASE_DIR, "uploads", "retraining")

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

# ==========================================
# CLASS LABELS (sama persis dengan app.py)
# ==========================================
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
# CUSTOM DATASET — pakai list dari DB
# (tidak load folder 15K lagi)
# ==========================================
class SubsetDataset(Dataset):
    """
    Menerima list dict dari DB:
    [{"image_path": "...", "predicted_class": "...", "source": "new/hard/stratified"}, ...]
    """
    def __init__(self, records: list):
        # Filter hanya record yang file-nya benar-benar ada
        self.records = [r for r in records if os.path.exists(r["image_path"])]
        skipped = len(records) - len(self.records)
        if skipped > 0:
            print(f"⚠️ {skipped} file tidak ditemukan, dilewati.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img = transform(img)
        except Exception as e:
            # Jika gambar corrupt, ganti dengan tensor kosong
            print(f"⚠️ Gagal baca gambar: {rec['image_path']} — {e}")
            img = torch.zeros(3, 224, 224)

        label = LABEL_TO_IDX.get(rec["predicted_class"], 0)
        return img, label, rec.get("source", "stratified")


def build_weighted_loader(records: list) -> DataLoader:
    """
    Data baru dan hard examples diberi bobot 3x lebih tinggi
    supaya model lebih fokus belajar dari data yang sulit.
    """
    dataset = SubsetDataset(records)

    if len(dataset) == 0:
        raise Exception("Semua file gambar tidak ditemukan. Training dibatalkan.")

    weights = []
    for rec in dataset.records:
        source = rec.get("source", "stratified")
        weights.append(3.0 if source in ("new", "hard") else 1.0)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


# ==========================================
# TRAINING ENGINE
# ==========================================
def run_pytorch_training(model_save_path, subset_records: list):
    """
    Menjalankan fine-tuning dengan strategi freezing layer dan rehearsal.
    """
    num_classes = len(CLASS_LABELS)
    # 1. Panggil fungsi untuk load model yang sudah di-freeze
    model = get_model_for_training(num_classes)
    model = model.to(device) # Konsistensi device

    print(f"\n🏋️ [PyTorch Engine] Memulai fine-tuning ResNet18...")
    print(f"📊 Total subset: {len(subset_records)} data")
    
    # 2. Build Loader
    loader = build_weighted_loader(subset_records)

    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    # Hanya optimalkan layer terakhir (fc) yang membutuhkan gradien
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    model.train()

    final_loss = 0.0
    final_accuracy = 0.0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        acc = correct / max(1, total)
        final_loss = avg_loss
        final_accuracy = acc
        print(f"📈 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    scheduler.step()

    # 4. Simpan model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved: {model_save_path}")

    return {
        "accuracy": round(final_accuracy, 4),
        "final_loss": round(final_loss, 4),
        "total_data_retrain": len(subset_records),
        "epochs": EPOCHS
    }

def get_model_for_training(num_classes):
    # 1. Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    
    # 2. FREEZE LAYERS (Strategi Efisiensi)
    # Bekukan semua layer agar tidak dihitung gradiennya
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Ganti Fully Connected Layer (hanya layer ini yang akan dilatih)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_training_subset(all_data, new_data_ids, sample_size_old=200):
    """
    Rehearsal Strategy:
    Mengambil semua data baru + sampel acak data lama (Hard Examples)
    """
    new_data = [d for d in all_data if d['id'] in new_data_ids]
    old_data_sample = random.sample([d for d in all_data if d['id'] not in new_data_ids], sample_size_old)
    
    return new_data + old_data_sample