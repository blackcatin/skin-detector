import io
import cv2
import traceback
import numpy as np
import uuid
import os
import time
import shutil
import mlflow
import mlflow.pytorch
import tempfile
import requests

from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import func, text
from fastapi import UploadFile, File
from fastapi import Request
from huggingface_hub import HfApi
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from PIL import Image
from model_registry import (
    get_current_model_path,
    get_current_model_version_name,
    get_current_active_filename,   # ← pastikan ini ada
    load_history,
    update_model_registry
)

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

from image_quality import (
    calculate_quality_score,
    validate_image,
    analyze_image_quality
)
from preprocessing import preprocess_image
from model_registry import (
    get_current_model_path,
    get_current_model_version_name,
    load_history,
    update_model_registry
)
from retrain import RETRAIN_DIR, run_pytorch_training

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

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SyifAI_Skin_Disease_Retraining")

RAW_FOLDER       = os.path.join(CURRENT_DIR, "uploads", "raw")
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
# LOAD MODEL
# ===============================
model = None

MODEL_VERSION = get_current_model_version_name()
model_path    = get_current_model_path()

print(f"📦 [Registry Manifest] Versi model aktif: {MODEL_VERSION}")

try:
    if os.path.exists(model_path):
        print(f"📂 Loading model lokal: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            model = build_model(num_classes=23)
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        print("✅ Model berhasil dimuat")
    else:
        print("⚠️ File model tidak ditemukan")
        model = build_model(num_classes=23)

except Exception as e:
    print("❌ Gagal memuat model:", e)
    model = build_model(num_classes=23)

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

# ===============================
# ROOT
# ===============================
@app.get("/")
def root():
    return {"message": "Skin Detector API aktif 🚀", "active_model": MODEL_VERSION}

@app.get("/get-previous-model-info")
def get_previous_model_info():
    """
    Dipanggil n8n SEBELUM training selesai — ambil info model yang
    sedang aktif sekarang (yang akan menjadi 'model lama' setelah training).
    """
    current_filename = get_current_model_version_name() + ".pth"
    current_path     = get_current_model_path()

    if not os.path.exists(current_path):
        return {"has_previous": False}

    return {
        "has_previous":    True,
        "filename":        current_filename,
        "path":            current_path,
        "size_mb":         round(os.path.getsize(current_path) / 1024 / 1024, 2)
    }


@app.get("/download-model/{filename}")
def download_model(filename: str, background_tasks: BackgroundTasks):
    # Guard: jika filename kosong atau "None"
    if not filename or filename == "None":
        return {"status": "skipped", "message": "Tidak ada model lama untuk diarsip"}

    search_dirs = [
        os.path.join(CURRENT_DIR, "trained_models", "active"),
        os.path.join(CURRENT_DIR, "trained_models"),
    ]

    file_path = None
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            file_path = candidate
            break

    if not file_path:
        return {"error": f"File {filename} tidak ditemukan."}

    # Tidak hapus file — model aktif harus tetap ada di lokal
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

@app.get("/download-previous-model")
def download_previous_model(background_tasks: BackgroundTasks):
    history = load_history()  # dari model_registry
    # Ambil model ACTIVE sebelumnya (index -2, karena -1 adalah yang baru)
    active_models = [h for h in history if h["status"] == "ACTIVE"]
    if len(active_models) < 2:
        return {"error": "Tidak ada model lama untuk diarsip"}
    
    prev_model = active_models[-2]  # model sebelumnya
    filename   = prev_model["model_path"]
    # ... cari file dan return FileResponse

# ===============================
# PREDICT
# ===============================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content      = await file.read()
        filename     = f"{uuid.uuid4()}.jpg"
        raw_filepath = os.path.join(RAW_FOLDER, filename)

        with open(raw_filepath, "wb") as f:
            f.write(content)

        image_cv         = cv2.imread(raw_filepath)
        quality_data     = analyze_image_quality(image_cv)
        blur_score       = float(quality_data["blur_score"])
        brightness_score = float(quality_data["brightness_score"])
        quality_score    = float(quality_data["quality_score"])
        is_valid         = bool(validate_image(image_cv, blur_score, brightness_score))

        processed_filepath = preprocess_image(raw_filepath, PROCESSED_FOLDER)

        image        = Image.open(processed_filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs    = model(image_tensor)
            probs      = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            class_idx             = predicted.item()
            confidence_percentage = confidence.item() * 100

        result = class_labels[class_idx]

        db = SessionLocal()
        prediction_data = Prediction(
            image_path           = raw_filepath,
            processed_image_path = processed_filepath,
            predicted_class      = result,
            confidence           = round(confidence_percentage, 2),
            blur_score           = blur_score,
            brightness_score     = brightness_score,
            quality_score        = quality_score,
            is_valid             = is_valid,
            model_version        = MODEL_VERSION,
            used_for_retraining  = False,
            confidence_score     = confidence_percentage / 100.0,
            is_hard_example      = (confidence_percentage / 100.0) < 0.6
        )
        db.add(prediction_data)
        db.commit()
        db.close()

        return {
            "filename":        filename,
            "predicted_class": result,
            "confidence":      round(confidence_percentage, 2),
            "raw_image":       raw_filepath,
            "processed_image": processed_filepath,
            "quality": {
                "blur_score":            round(blur_score, 2),
                "brightness_score":      round(brightness_score, 2),
                "quality_score":         round(quality_score, 2),
                "is_valid_for_training": is_valid
            }
        }

    except Exception as e:
        print("❌ Predict error:", traceback.format_exc())
        return {"status": "failed", "message": "Predict gagal", "error": str(e)}


def download_image_from_url(url, target_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"⚠️ Gagal download {url}: {e}")
    return False

# ===============================
# TRAIN
# ===============================
@app.post("/train")
async def trigger_retraining():
    db = SessionLocal()

    try:
        # ── 1. Ambil data baru ──────────────────────────────────────
        new_samples = db.query(Prediction).filter(
            Prediction.is_valid          == True,
            Prediction.used_for_retraining == False
        ).all()

        if len(new_samples) < 5:
            return {
                "status":        "failed",
                "message":       f"Data baru hanya {len(new_samples)}, minimum 5.",
                "is_best_model": False
            }

        print(f"📊 Data baru: {len(new_samples)}")

        # ── 2. Hard examples ────────────────────────────────────────
        hard_samples = db.query(Prediction).filter(
            Prediction.is_valid            == True,
            Prediction.used_for_retraining == True,
            Prediction.is_hard_example     == True
        ).order_by(Prediction.confidence_score.asc()).limit(150).all()

        # ── 3. Stratified sample ────────────────────────────────────
        stratified_rows = db.execute(text("""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY predicted_class ORDER BY RANDOM()
                ) AS rn
                FROM predictions
                WHERE used_for_retraining = TRUE AND is_valid = TRUE
            ) t WHERE rn <= 10
        """)).fetchall()

        print(f"   ↳ Hard examples : {len(hard_samples)}")
        print(f"   ↳ Stratified    : {len(stratified_rows)}")

        # ── 4. Bangun subset_records ────────────────────────────────
        subset_records = []

        for s in new_samples:
            image_path = s.processed_image_path or s.image_path
            if not image_path:
                continue
            subset_records.append({
                "image_path":      image_path,
                "predicted_class": s.predicted_class,
                "source":          "new"
            })

        for s in hard_samples:
            image_path = s.processed_image_path or s.image_path
            if not image_path:
                continue
            subset_records.append({
                "image_path":      image_path,
                "predicted_class": s.predicted_class,
                "source":          "hard"
            })

        for row in stratified_rows:
            mapping    = row._mapping
            image_path = mapping.get("processed_image_path") or mapping.get("image_path")
            pred_class = mapping.get("predicted_class")
            if not image_path or not pred_class:
                continue
            subset_records.append({
                "image_path":      image_path,
                "predicted_class": pred_class,
                "source":          "stratified"
            })

        print(f"📚 Total subset: {len(subset_records)} data")

        # ── 5. Setup path model baru ────────────────────────────────
        TARGET_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
        os.makedirs(TARGET_MODEL_DIR, exist_ok=True)

        current_version_name = get_current_model_version_name()
        try:
            next_version = int(
                current_version_name
                .replace("Fix_best_model_v", "")
                .replace("model_v", "")
            ) + 1
        except:
            next_version = 2

        NEW_MODEL_VERSION = f"model_v{next_version}"
        new_model_path    = os.path.join(TARGET_MODEL_DIR, f"Fix_best_{NEW_MODEL_VERSION}.pth")

        # ── 6. Training ─────────────────────────────────────────────
        start_time      = time.time()
        training_result = run_pytorch_training(model, new_model_path, subset_records)
        model.eval()  # ← WAJIB ditambah ini, kembalikan mode inference
        print("🔄 Model dikembalikan ke mode eval setelah training.")

        if not os.path.exists(new_model_path) or os.path.getsize(new_model_path) < 1000:
            raise Exception("Training gagal menghasilkan model valid.")

        raw_accuracy = training_result.get("accuracy", 0.0)
        final_loss   = training_result.get("final_loss", 0.0)
        exec_time    = round(time.time() - start_time, 2)

        # ── 7. Update registry ──────────────────────────────────────
        registry_result = update_model_registry(
            new_model_path    = new_model_path,
            version           = NEW_MODEL_VERSION,
            accuracy          = f"{float(raw_accuracy) * 100:.1f}%",
            final_loss        = final_loss,
            total_data_retrain= len(subset_records),
            execution_time    = exec_time,
            upload_success    = False
        )

        is_best = registry_result["is_best_model"]
        print("✅ Training selesai.")

        return {
            "status":                    "success",
            "is_best_model":             is_best,
            "model_version":             NEW_MODEL_VERSION,
            "model_path":                os.path.basename(new_model_path),
            "previous_model_filename":   registry_result.get("previous_model_filename"),  # ← tambah ini
            "accuracy":                  f"{float(raw_accuracy) * 100:.1f}%",
            "final_loss":                final_loss,
            "total_data_retrain":        len(subset_records),
            "total_new_data":            len(new_samples),
            "execution_time":            exec_time,
            "retrained_at":              time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"❌ Error fatal /train: {e}")
        print(traceback.format_exc())
        return {
            "status":        "failed",
            "is_best_model": False,
            "error":         str(e)
        }

    finally:
        db.close()

# ===============================
# UPLOAD MODEL KE HUGGING FACE
# ===============================
@app.post("/upload-model")
async def upload_model(request: Request):
    print("=== UPLOAD REQUEST (RAW BINARY MODE) ===")
    temp_path = None

    try:
        content = await request.body()
        if len(content) < 1000:
            raise HTTPException(
                status_code=400,
                detail=f"File rusak/terlalu kecil ({len(content)} byte), gagal upload!"
            )

        token   = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_REPO_ID", "blackcatin/resnet18-syifai")

        if not token:
            raise HTTPException(status_code=500, detail="HF_TOKEN tidak ditemukan di .env")

        # Ambil nama file dari header Content-Disposition yang dikirim n8n
        # n8n mengirim: attachment; filename="Fix_best_model_v3.pth"
        filename  = "Fix_best_model_latest.pth"  # fallback
        cd_header = request.headers.get("content-disposition")
        if cd_header and "filename=" in cd_header:
            raw_name = cd_header.split("filename=")[1].strip('"').strip("'").strip()
            if raw_name and raw_name not in ("None", ""):
                filename = raw_name

        temp_path = os.path.join(CURRENT_DIR, f"temp_{filename}")
        with open(temp_path, "wb") as f:
            f.write(content)

        api = HfApi(token=token)
        print(f"Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj  = temp_path,
            path_in_repo     = filename,
            repo_id          = repo_id,
            repo_type        = "model",
            commit_message   = f"Auto-upload model {filename} via n8n pipeline"
        )

        print("✅ Upload ke Hugging Face Hub sukses!")
        return {"status": "success", "uploaded_file": filename}

    except Exception as e:
        print("❌ ERROR UPLOAD:", repr(e))
        if "RepositoryNotFoundError" in str(e):
            raise HTTPException(status_code=404, detail="Repo tidak ditemukan. Cek HF_REPO_ID di .env!")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("🗑️ Berkas transit dihapus.")

# ===============================
# ARCHIVE RETRAINING DATA
# ===============================
@app.post("/archive-retraining-data")
async def archive_retraining_data():
    db = SessionLocal()

    try:
        samples = db.query(Prediction).filter(
            Prediction.is_valid            == True,
            Prediction.used_for_retraining == False
        ).all()

        archived = []

        for sample in samples:
            if os.path.exists(sample.image_path):
                archived.append({
                    "id":              sample.id,
                    "predicted_class": sample.predicted_class,
                    "raw_image":       sample.image_path,
                    "processed_image": sample.processed_image_path
                })
                sample.used_for_retraining = True

        db.commit()

        return {
            "status":     "success",
            "total_data": len(archived),
            "data":       archived
        }

    except Exception as e:
        print("❌ Archive error:", traceback.format_exc())
        return {"status": "failed", "error": str(e)}

    finally:
        db.close()

@app.get("/read-raw-image/{filename}")
def read_raw_image(filename: str):
    """Kirim file gambar raw sebagai binary untuk diupload n8n ke GDrive."""
    path = os.path.join(RAW_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File tidak ditemukan: {filename}")
    return FileResponse(path, media_type="image/jpeg", filename=filename)

# ===============================
# DELETE OLD MODEL FROM LOCAL
# ===============================
@app.delete("/delete-old-model")
def delete_old_model(filename: str):
    if not filename or filename in ("None", ""):
        return {"status": "skipped", "message": "Tidak ada filename yang diberikan"}

    current_active = get_current_active_filename()
    if filename == current_active:
        return {
            "status":  "skipped",
            "message": f"File {filename} adalah model aktif saat ini, tidak dihapus"
        }

    search_dirs = [
        os.path.join(CURRENT_DIR, "trained_models", "active"),
        os.path.join(CURRENT_DIR, "trained_models"),
    ]

    for d in search_dirs:
        target = os.path.join(d, filename)
        if os.path.exists(target):
            os.remove(target)
            print(f"🗑️ Model lama dihapus dari lokal: {target}")
            return {"status": "deleted", "filename": filename, "path": target}

    return {"status": "not_found", "message": f"File {filename} tidak ditemukan di lokal"}