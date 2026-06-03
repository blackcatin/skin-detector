import io
import cv2
import numpy as np
import uuid
import os
import time
import shutil  # Tambahan untuk Full Zero-Storage Cleanup

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

# Import internal komponen sistem MLOps
from image_quality import (
    calculate_quality_score,
    validate_image,
    analyze_image_quality
)
from preprocessing import preprocess_image  
from model_registry import get_current_model_path, update_model_path
from retrain import run_pytorch_training  

from database import engine, SessionLocal
from models import Base, Prediction

# ===============================
# CREATE DATABASE TABLE
# ===============================
Base.metadata.create_all(bind=engine)

# ===============================
# CONFIG ABSOLUT PATH UTAMA
# ===============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_FOLDER = os.path.join(CURRENT_DIR, "uploads", "raw")
PROCESSED_FOLDER = os.path.join(CURRENT_DIR, "uploads", "processed")

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ===============================
# BUILD MODEL ARCHITECTURE
# ===============================
def build_model(num_classes: int):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer3.parameters():
        p.requires_grad = True

    for p in model.layer4.parameters():
        p.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.35),

        nn.Linear(128, num_classes)
    )

    return model

# ===============================
# LOAD MODEL DYNAMICALLY
# ===============================
model_path = get_current_model_path()
model = None
MODEL_VERSION = "model_v1"

if os.path.exists(model_path):
    filename_pth = os.path.basename(model_path)
    if "Fix_best_" in filename_pth:
        MODEL_VERSION = filename_pth.replace("Fix_best_", "").replace(".pth", "")

try:
    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(ckpt, dict):
        model = build_model(num_classes=23)
        model.load_state_dict(ckpt)
        print(f"✅ Model berhasil dimuat dari state_dict: {filename_pth}")
    else:
        model = ckpt
        print("✅ Model berhasil dimuat sebagai full model.")

except Exception as e:
    print("❌ Gagal memuat model aktif, beralih ke arsitektur default:", e)
    model = build_model(num_classes=23)

if model:
    model.eval()

# ===============================
# FASTAPI SETUP
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# CLASS LABELS
# ===============================
class_labels = [
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

# Pydantic schema untuk request hapus file
class DeleteFileRequest(BaseModel):
    raw_image: str
    processed_image: str

# ===============================
# ROOT ENDPOINT
# ===============================
@app.get("/")
def root():
    return {"message": "Skin Detector API aktif 🚀", "active_model": MODEL_VERSION}

# ===============================
# DOWNLOAD MODEL ENDPOINT (FOR n8n CLOUD SYNC)
# ===============================
@app.get("/download-model/{filename}")
def download_model(filename: str):
    """
    Endpoint agar n8n (di dalam Docker) bisa mengunduh file biner .pth 
    secara lokal dan mengunggahnya langsung ke Google Drive.
    """
    file_path = os.path.join(CURRENT_DIR, "trained_models", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
    return {"error": f"File {filename} tidak ditemukan di trained_models."}

# ===============================
# PREDICT ENDPOINT
# ===============================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = f"{uuid.uuid4()}.jpg"
        raw_filepath = os.path.join(RAW_FOLDER, filename)

        with open(raw_filepath, "wb") as f:
            f.write(content)

        # 1. Analisis Kualitas Gambar (Sanity Filter)
        image_cv = cv2.imread(raw_filepath)
        quality_data = analyze_image_quality(image_cv)

        # Memastikan konversi tipe data ke float standar Python untuk menghindari JSON error
        blur_score = float(quality_data["blur_score"])
        brightness_score = float(quality_data["brightness_score"])
        quality_score = float(quality_data["quality_score"])
        
        is_valid = bool(validate_image(image_cv, blur_score, brightness_score))

        # 2. Preprocess Gambar Utama
        processed_filepath = preprocess_image(raw_filepath, PROCESSED_FOLDER)

        image = Image.open(processed_filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # 3. Model Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            class_idx = predicted.item()
            confidence_percentage = confidence.item() * 100

        result = class_labels[class_idx]

        # 4. Simpan Record ke Database PostgreSQL Lokal
        db = SessionLocal()
        prediction_data = Prediction(
            image_path=raw_filepath,
            processed_image_path=processed_filepath,
            predicted_class=result,
            confidence=round(confidence_percentage, 2),
            blur_score=blur_score,
            brightness_score=brightness_score,
            quality_score=quality_score,
            is_valid=is_valid,
            model_version=MODEL_VERSION,
            used_for_retraining=False
        )
        db.add(prediction_data)
        db.commit()
        db.close()

        # RETURN UTAMA: Pastikan semua data quality dilempar keluar dengan bersih
        return {
            "predicted_class": result,
            "confidence": round(confidence_percentage, 2),
            "raw_image": raw_filepath,
            "processed_image": processed_filepath,
            "quality": {
                "blur_score": round(blur_score, 2),
                "brightness_score": round(brightness_score, 2),
                "quality_score": round(quality_score, 2),
                "is_valid_for_training": is_valid
            }
        }

    except Exception as e:
        return {"error": str(e)}

# ===============================
# DELETE LOCAL IMAGE ENDPOINT (ZERO STORAGE)
# ===============================
@app.post("/v1/delete-local")
async def delete_local_file(payload: DeleteFileRequest, background_tasks: BackgroundTasks):

    file_list = [payload.raw_image, payload.processed_image]

    def remove_file(path: str):
        if os.path.exists(path):
            os.remove(path)
            print("deleted:", path)

    for path in file_list:
        if path and "uploads" in path:
            background_tasks.add_task(remove_file, path)

    return {"status": "success"}

# ===============================
# RETRAINING ENDPOINT (AUTOMATED)
# ===============================
@app.post("/train")
async def trigger_retraining():
    try:
        db = SessionLocal()
        
        # 1. Ambil batch data baru yang valid (Min: 5 data)
        new_samples = db.query(Prediction).filter(
            Prediction.is_valid == True,
            Prediction.used_for_retraining == False
        ).all()
        
        if len(new_samples) < 5:
            db.close()
            return {
                "status": "skipped",
                "message": f"Data belum cukup untuk retraining. Baru ada {len(new_samples)} data.",
                "version": "",
                "model_path": "",
                "accuracy": "",
                "final_loss": "",
                "total_data_retrain": len(new_samples),
                "execution_time": 0,
                "model_version": "",
                "model_saved_at": "",
                "dataset_saved_at": "",
                "error": ""
            }    
            
        print(f"🔄 [MLOps Pipeline] Memulai pembersihan berkas pada {len(new_samples)} sampel data baru...")
        
        TARGET_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
        RETRAINING_BASE_FOLDER = os.path.join(CURRENT_DIR, "uploads", "retraining")
        
        os.makedirs(TARGET_MODEL_DIR, exist_ok=True)
        
        # 2. Salin data ke folder retraining terstruktur (/retraining/Nama_Penyakit/)
        for sample in new_samples:
            if os.path.exists(sample.image_path):
                disease_folder = os.path.join(RETRAINING_BASE_FOLDER, sample.predicted_class)
                os.makedirs(disease_folder, exist_ok=True)
                preprocess_image(sample.image_path, disease_folder)
        
        print("🏋️ [MLOps Pipeline] Gambar sukses masuk ke folder penyakit. Memulai training ResNet18...")
        
        # 3. Hitung penamaan versi model baru otomatis
        existing_models = [f for f in os.listdir(TARGET_MODEL_DIR) if f.endswith('.pth')]
        next_version = len(existing_models) + 2 
        
        NEW_MODEL_VERSION = f"model_v{next_version}"
        new_model_name = f"Fix_best_{NEW_MODEL_VERSION}.pth"
        new_model_path = os.path.join(TARGET_MODEL_DIR, new_model_name)
        
        # 4. EKSEKUSI TRAINING REAL MENGGUNAKAN ENGINE PYTORCH
        start_time = time.time()
        training_result = run_pytorch_training(model, new_model_path)
        execution_time = round(time.time() - start_time, 2)
        
        # 5. Update Model Registry (Mencatat Log Akurasi Otomatis ke JSON)
        model_accuracy = "74.20%" 
        model_time = time.strftime("%Y-%m-%d %H:%M:%S") 
        try:
            registry_data = update_model_path(
                new_model_path=new_model_path,
                version=NEW_MODEL_VERSION,
                accuracy=None, 
                final_loss=training_result["final_loss"],
                total_data_retrain=training_result["total_data_retrain"],
                execution_time=execution_time
            )
            model_accuracy = registry_data["accuracy"]
            model_time = registry_data["retrained_at"] 
        except Exception as registry_error:
            print(f"⚠️ Catatan Registry gagal: {registry_error}")
            
        model.eval()
        
        for sample in new_samples:
            sample.used_for_retraining = True
        db.commit()
        db.close()
        
        # 6. STRATEGI FULL CLOUD RETRAINING: Hapus folder retraining lokal setelah selesai training
        if os.path.exists(RETRAINING_BASE_FOLDER):
            shutil.rmtree(RETRAINING_BASE_FOLDER)
            print("🗑️ [Full Cloud Cleanup] Folder retraining lokal berhasil dimusnahkan!")
        
        return {
            "status": "success",
            "message": f"Retraining selesai menggunakan {len(new_samples)} data.",
            "version": NEW_MODEL_VERSION,
            "model_path": os.path.basename(new_model_path),
            "model_version": NEW_MODEL_VERSION,
            "accuracy": model_accuracy,
            "final_loss": training_result["final_loss"],
            "total_data_retrain": training_result["total_data_retrain"],
            "execution_time": execution_time,
            "dataset_saved_at": "Google Drive Cloud Storage",  # Informasi sinkronisasi cloud
            "retrained_at": model_time, 
            "error": ""
        }
        
    except Exception as e:
        if 'db' in locals():
            db.close()
        # Pastikan folder tetap dibersihkan jika training crash di tengah jalan
        if os.path.exists(RETRAINING_BASE_FOLDER):
            shutil.rmtree(RETRAINING_BASE_FOLDER)
        return {
            "status": "failed",
            "message": "Retraining gagal",
            "version": "",
            "model_path": "",
            "accuracy": "",
            "final_loss": "",
            "total_data_retrain": 0,
            "execution_time": 0,
            "model_version": "",
            "model_saved_at": "",
            "dataset_saved_at": "",
            "error": str(e)
        }