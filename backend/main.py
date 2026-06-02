import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SyifAI Filter Backend API")

# Izinkan CORS agar frontend (React/Streamlit) atau n8n bisa mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_image_blur(image_bytes, threshold=20.0):
    """
    Menghitung tingkat keburaman gambar menggunakan Laplacian Variance.
    Jika skor < threshold, berarti gambar terlalu blur (tidak layak untuk training).
    """
    # Mengubah data biner (bytes) menjadi format matriks yang dipahami OpenCV
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        # Jika gambar rusak atau tidak bisa didecode, otomatis dianggap tidak valid
        return True, 0.0
    
    # Hitung variance dari Laplacian (Skor Keburaman)
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # True jika blur (di bawah threshold), False jika tajam/clear
    is_blur = blur_score < threshold
    return is_blur, round(blur_score, 2)

@app.get("/")
def root():
    return {"message": "SyifAI Filter Backend is running 🚀"}

@app.post("/v1/filter-image")
async def filter_image(file: UploadFile = File(...)):
    try:
        # 1. Baca data biner gambar yang diunggah
        image_bytes = await file.read()
        
        # 2. Jalankan fungsi deteksi blur
        is_blurry, score = check_image_blur(image_bytes)
        
        # 3. Tentukan keputusan untuk pipeline MLOps n8n
        # Gambar yang blur TETAP dideteksi penyakitnya nanti, tapi DITOLAK masuk dataset training
        action = "reject_from_training" if is_blurry else "allow_training"
        
        return {
            "filename": file.filename,
            "blur_score": score,
            "is_blurry": is_blurry,
            "is_valid_for_training": not is_blurry,
            "action_required": action
        }
        
    except Exception as e:
        return {"error": f"Gagal memproses gambar: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Jalankan server lokal di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)