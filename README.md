# 🩺 SyifAI - Skin Disease Detector

SyifAI adalah aplikasi deteksi penyakit kulit berbasis AI yang menggunakan model ResNet18 untuk melakukan klasifikasi gambar penyakit kulit. Sistem dibangun menggunakan React, FastAPI, PyTorch, PostgreSQL, MLflow, dan n8n untuk mendukung otomatisasi retraining serta manajemen model secara end-to-end.

---

## ✨ Features

* 🖼️ Upload gambar penyakit kulit
* 🤖 Prediksi menggunakan model ResNet18
* 🌐 REST API dengan FastAPI
* 📊 Experiment tracking dengan MLflow
* 🗄️ Penyimpanan data menggunakan PostgreSQL
* 🔄 Automated retraining menggunakan n8n
* 🤗 Integrasi Hugging Face untuk model registry
* ☁️ Backup model ke Google Drive
* 📱 Notifikasi Telegram

---

## 🛠️ Tech Stack

### 🎨 Frontend

* React
* Tailwind CSS

### ⚙️ Backend

* FastAPI
* PyTorch
* torchvision

### 🗄️ Database

* PostgreSQL

### 🚀 MLOps

* MLflow
* n8n

### 💾 Model Storage

* Hugging Face Hub
* Google Drive

---

## 📁 Project Structure

```text
skin-detector/
├── src/
├── public/
├── backend/
│   ├── app.py
│   ├── retrain.py
│   ├── check_model.py
│   ├── model_history.json
│   └── trained_models/
├── mlflow.db
├── package.json
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/blackcatin/skin-detector.git
cd skin-detector
```

---

## 🎨 Frontend Setup

Install dependencies:

```bash
npm install
```

Run development server:

```bash
npm start
```

Frontend will run at:

```text
http://localhost:3000
```

---

## ⚙️ Backend Setup

Masuk ke folder backend:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Buat file `.env`:

```env
HF_TOKEN=your_huggingface_token
```

Jalankan backend:

```bash
uvicorn app:app --reload
```

Backend will run at:

```text
http://localhost:8000
```

---

## 🗄️ PostgreSQL Setup

Buat database PostgreSQL dan sesuaikan konfigurasi:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=syifai
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## 📊 MLflow Setup

Jalankan MLflow UI:

```bash
mlflow ui
```

Dashboard tersedia di:

```text
http://localhost:5000
```

MLflow digunakan untuk:

* 📈 Tracking eksperimen training
* 📊 Monitoring metric model
* 🏆 Perbandingan performa model
* 🔍 Evaluasi hasil retraining

---

## 🔌 API Endpoint

### Predict Skin Disease

```http
POST /predict
```

Response:

```json
{
  "prediction": "Melanoma",
  "confidence": 0.95
}
```

---

## 🔄 Automated MLOps Workflow

Workflow otomatis menggunakan n8n:

```text
Schedule Trigger
      ↓
PostgreSQL Check
      ↓
Retraining Trigger
      ↓
MLflow Logging
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Hugging Face Upload
      ↓
Google Drive Backup
      ↓
Telegram Notification
      ↓
Storage Cleanup
```

---

## ⚠️ Important Notes

Jangan pernah mengunggah file berikut ke repository:

```text
.env
backend/.env
```

Gunakan environment variables untuk menyimpan credential dan token.

---

## 👨‍💻 Author

**Firlana Umi**

Faculty of Science and Technology
Skin Disease Detection & MLOps Automation Project
