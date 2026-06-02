import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, Brain, AlertCircle, Info, CheckCircle2, Sparkles, Zap, Shield, Heart, Star, ArrowRight } from 'lucide-react';

const SkinDiseasePredictor = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showParticles, setShowParticles] = useState(false);
  const [imageData, setImageData] = useState(null);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Mapping dari backend label ke frontend key
  const backendToFrontendMapping = {
    "Acne and Rosacea": "Acne and Rosacea Photos",
    "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis": "Atopic Dermatitis Photos",
    "Bullous Disease": "Bullous Disease Photos",
    "Cellulitis, Impetigo & other Bacterial Infections": "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema": "Eczema Photos",
    "Exanthems & Drug Eruptions": "Exanthems and Drug Eruptions",
    "Hair Loss, Alopecia & other Hair Diseases": "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes, HPV & other STDs": "Herpes HPV and other STDs Photos",
    "Light Diseases & Disorders of Pigmentation": "Light Diseases and Disorders of Pigmentation",
    "Lupus & other Connective Tissue diseases": "Lupus and other Connective Tissue diseases",
    "Melanoma, Skin Cancer, Nevi & Moles": "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus & other Nail Disease": "Nail Fungus and other Nail Disease",
    "Poison Ivy & other Contact Dermatitis": "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis, Lichen Planus & related diseases": "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies, Lyme Disease & other Infestations & Bites": "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses & other Benign Tumors": "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease": "Systemic Disease",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria / Hives": "Urticaria Hives",
    "Vascular Tumors": "Vascular Tumors",
    "Vasculitis": "Vasculitis Photos",
    "Warts, Molluscum & other Viral Infections": "Warts Molluscum and other Viral Infections",
  };

  // Particle animation
  useEffect(() => {
    setShowParticles(true);
  }, []);

  // Database penjelasan penyakit
  const diseaseInfo = {
    "Acne and Rosacea Photos": {
      name: "Jerawat dan Rosacea",
      description: "Kondisi kulit yang ditandai dengan komedo, papula, pustula, atau kemerahan pada wajah.",
      symptoms: ["Komedo", "Papula", "Pustula", "Kemerahan"],
      causes: ["Produksi minyak berlebih", "Bakteri P. acnes", "Hormon", "Stres"],
      treatment: ["Pembersih wajah lembut", "Obat topikal", "Konsultasi dokter kulit"],
      severity: "Ringan - Sedang",
      color: "from-orange-400 to-red-500",
      bgColor: "from-orange-50 to-red-50",
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
      name: "Keratosis Aktinik & Karsinoma Basal Sel",
      description: "Lesi kulit yang bersifat prakanker atau kanker, biasanya muncul akibat paparan sinar matahari berlebih.",
      symptoms: ["Bercak kasar", "Kemerahan", "Luka yang sulit sembuh"],
      causes: ["Paparan sinar UV", "Usia lanjut", "Kulit terang"],
      treatment: ["Cryotherapy", "Biopsi dan eksisi", "Terapi laser"],
      severity: "Tinggi",
      color: "from-red-600 to-red-800",
      bgColor: "from-red-50 to-red-100",
    },
    "Atopic Dermatitis Photos": {
      name: "Dermatitis Atopik (Eksim)",
      description: "Kondisi kulit kronis yang menyebabkan kulit kering, gatal, dan ruam merah.",
      symptoms: ["Kulit kering", "Ruam merah", "Gatal", "Kemerahan berulang"],
      causes: ["Genetik", "Alergi", "Stres"],
      treatment: ["Pelembab rutin", "Kortikosteroid topikal", "Hindari pemicu"],
      severity: "Sedang - Kronis",
      color: "from-yellow-400 to-orange-500",
      bgColor: "from-yellow-50 to-orange-50",
    },
    "Bullous Disease Photos": {
      name: "Penyakit Bullous",
      description: "Gangguan kulit yang menyebabkan lepuhan besar berisi cairan di permukaan kulit.",
      symptoms: ["Lepuhan besar", "Gatal", "Nyeri kulit"],
      causes: ["Autoimun", "Infeksi", "Genetik"],
      treatment: ["Kortikosteroid", "Imunosupresan", "Perawatan luka"],
      severity: "Sedang - Tinggi",
      color: "from-purple-400 to-pink-500",
      bgColor: "from-purple-50 to-pink-50",
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
      name: "Infeksi Bakteri Kulit",
      description: "Infeksi kulit yang disebabkan oleh bakteri, seperti selulitis atau impetigo.",
      symptoms: ["Kemerahan", "Bengkak", "Nyeri", "Luka bernanah"],
      causes: ["Bakteri Staphylococcus", "Bakteri Streptococcus", "Cedera kulit"],
      treatment: ["Antibiotik oral/topikal", "Perawatan luka", "Konsultasi dokter"],
      severity: "Sedang - Tinggi",
      color: "from-red-400 to-red-600",
      bgColor: "from-red-50 to-red-100",
    },
    "Eczema Photos": {
      name: "Eksim",
      description: "Kondisi kulit kering dan meradang yang sering kambuh.",
      symptoms: ["Kulit kering", "Gatal", "Ruam merah"],
      causes: ["Genetik", "Alergi", "Iritasi"],
      treatment: ["Pelembab rutin", "Kortikosteroid topikal", "Hindari pemicu"],
      severity: "Sedang",
      color: "from-yellow-400 to-yellow-600",
      bgColor: "from-yellow-50 to-yellow-100",
    },
    "Exanthems and Drug Eruptions": {
      name: "Ruam & Reaksi Obat",
      description: "Ruam kulit yang muncul akibat infeksi atau reaksi terhadap obat tertentu.",
      symptoms: ["Bercak merah", "Gatal", "Bintik-bintik"],
      causes: ["Infeksi virus/bakteri", "Reaksi obat"],
      treatment: ["Hentikan obat penyebab", "Antihistamin", "Perawatan simptomatik"],
      severity: "Ringan - Sedang",
      color: "from-pink-400 to-pink-600",
      bgColor: "from-pink-50 to-pink-100",
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
      name: "Rambut Rontok & Alopecia",
      description: "Kehilangan rambut sebagian atau total yang bisa bersifat sementara atau permanen.",
      symptoms: ["Rontok rambut", "Botak sebagian", "Penipisan rambut"],
      causes: ["Genetik", "Autoimun", "Stres", "Nutrisi kurang"],
      treatment: ["Minoxidil", "Kortikosteroid", "Terapi PRP"],
      severity: "Ringan - Sedang",
      color: "from-brown-400 to-orange-500",
      bgColor: "from-brown-50 to-orange-50",
    },
    "Herpes HPV and other STDs Photos": {
      name: "Infeksi Menular Seksual",
      description: "Infeksi kulit akibat virus, termasuk herpes dan HPV.",
      symptoms: ["Luka/lesi kulit", "Gatal", "Nyeri"],
      causes: ["Virus HSV", "Virus HPV", "Kontak seksual"],
      treatment: ["Antivirus", "Perawatan simptomatik", "Konsultasi dokter"],
      severity: "Sedang - Tinggi",
      color: "from-purple-400 to-red-500",
      bgColor: "from-purple-50 to-red-50",
    },
    "Light Diseases and Disorders of Pigmentation": {
      name: "Gangguan Pigmentasi",
      description: "Perubahan warna kulit akibat melanin yang berlebihan atau berkurang.",
      symptoms: ["Bercak putih/gelap", "Kulit tidak merata"],
      causes: ["Genetik", "Paparan sinar", "Autoimun"],
      treatment: ["Krim pencerah", "Terapi laser", "Konsultasi dokter"],
      severity: "Ringan - Sedang",
      color: "from-yellow-400 to-green-500",
      bgColor: "from-yellow-50 to-green-50",
    },
    "Lupus and other Connective Tissue diseases": {
      name: "Lupus & Penyakit Jaringan Ikat",
      description: "Penyakit autoimun yang dapat mempengaruhi kulit, sendi, dan organ lainnya.",
      symptoms: ["Ruam wajah", "Nyeri sendi", "Kelelahan"],
      causes: ["Autoimun", "Genetik", "Lingkungan"],
      treatment: ["Imunosupresan", "Kortikosteroid", "Perawatan simptomatik"],
      severity: "Sedang - Tinggi",
      color: "from-blue-400 to-purple-500",
      bgColor: "from-blue-50 to-purple-50",
    },
    "Melanoma Skin Cancer Nevi and Moles": {
      name: "Melanoma & Tahi Lalat",
      description: "Kanker kulit yang berkembang dari sel melanosit atau tahi lalat abnormal.",
      symptoms: ["Bercak hitam/gelap", "Bercak tidak simetris", "Bercak berubah bentuk"],
      causes: ["Paparan sinar UV", "Genetik", "Kulit terang"],
      treatment: ["Eksisi bedah", "Kemoterapi/topikal", "Pemantauan rutin"],
      severity: "Tinggi",
      color: "from-red-700 to-black",
      bgColor: "from-red-50 to-gray-100",
    },
    "Nail Fungus and other Nail Disease": {
      name: "Infeksi Kuku & Jamur",
      description: "Infeksi pada kuku yang dapat menyebabkan perubahan warna, ketebalan, dan bentuk kuku.",
      symptoms: ["Kuku tebal", "Kuku rapuh", "Perubahan warna"],
      causes: ["Jamur dermatofit", "Luka kuku", "Kelembaban tinggi"],
      treatment: ["Antijamur topikal/oral", "Perawatan kuku", "Konsultasi dokter"],
      severity: "Ringan - Sedang",
      color: "from-green-400 to-teal-500",
      bgColor: "from-green-50 to-teal-50",
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
      name: "Dermatitis Kontak",
      description: "Ruam kulit akibat kontak dengan alergen atau iritan seperti poison ivy.",
      symptoms: ["Ruam merah", "Gatal", "Lepuhan kecil"],
      causes: ["Tanaman alergen", "Kosmetik", "Logam"],
      treatment: ["Antihistamin", "Salep antiinflamasi", "Hindari pemicu"],
      severity: "Ringan - Sedang",
      color: "from-green-400 to-lime-500",
      bgColor: "from-green-50 to-lime-50",
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
      name: "Psoriasis & Lichen Planus",
      description: "Penyakit autoimun yang menyebabkan plak bersisik dan gatal pada kulit.",
      symptoms: ["Plak bersisik", "Kulit menebal", "Gatal"],
      causes: ["Genetik", "Autoimun", "Stres"],
      treatment: ["Terapi UV", "Obat imunosupresan", "Salep kortikosteroid"],
      severity: "Sedang - Tinggi",
      color: "from-blue-400 to-purple-500",
      bgColor: "from-blue-50 to-purple-50",
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
      name: "Skabies, Lyme & Gigitan Serangga",
      description: "Infestasi kulit oleh parasit atau gigitan serangga yang menimbulkan gatal dan ruam.",
      symptoms: ["Gatal hebat", "Bintik merah", "Lepuhan kecil"],
      causes: ["Parasit", "Kutu", "Gigitan serangga"],
      treatment: ["Obat anti-parasit", "Krim topikal", "Konsultasi dokter"],
      severity: "Sedang",
      color: "from-purple-400 to-pink-500",
      bgColor: "from-purple-50 to-pink-50",
    },
    "Seborrheic Keratoses and other Benign Tumors": {
      name: "Seboroik Keratosis & Tumor Jinak",
      description: "Pertumbuhan kulit jinak, biasanya berwarna coklat atau hitam dan menonjol.",
      symptoms: ["Lesi coklat/hitam", "Permukaan kasar"],
      causes: ["Penuaan", "Genetik"],
      treatment: ["Cryotherapy", "Eksisi jika mengganggu", "Pemantauan rutin"],
      severity: "Ringan",
      color: "from-yellow-400 to-orange-500",
      bgColor: "from-yellow-50 to-orange-50",
    },
    "Systemic Disease": {
      name: "Penyakit Sistemik",
      description: "Penyakit yang mempengaruhi organ dan sistem tubuh, juga bisa memunculkan gejala kulit.",
      symptoms: ["Ruam", "Nyeri sendi", "Kelelahan"],
      causes: ["Autoimun", "Infeksi", "Genetik"],
      treatment: ["Terapi medis sesuai diagnosis", "Konsultasi dokter"],
      severity: "Sedang - Tinggi",
      color: "from-blue-400 to-green-500",
      bgColor: "from-blue-50 to-green-50",
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
      name: "Infeksi Jamur Kulit",
      description: "Infeksi kulit yang disebabkan oleh jamur dermatofit atau kandida.",
      symptoms: ["Lepuhan merah", "Gatal", "Kuku berubah warna"],
      causes: ["Jamur dermatofit", "Kelembaban tinggi", "Kontak kulit"],
      treatment: ["Antijamur topikal/oral", "Perawatan kebersihan kulit"],
      severity: "Ringan - Sedang",
      color: "from-green-400 to-teal-500",
      bgColor: "from-green-50 to-teal-50",
    },
    "Urticaria Hives": {
      name: "Urtikaria (Biduran)",
      description: "Reaksi kulit berupa bentol merah gatal akibat alergi atau iritasi.",
      symptoms: ["Bentol merah", "Gatal hebat", "Hilang-timbul"],
      causes: ["Alergi makanan", "Obat", "Stres"],
      treatment: ["Antihistamin", "Hindari pemicu", "Kompress dingin"],
      severity: "Ringan - Sedang",
      color: "from-yellow-400 to-orange-500",
      bgColor: "from-yellow-50 to-orange-50",
    },
    "Vascular Tumors": {
      name: "Tumor Vaskular",
      description: "Pertumbuhan abnormal pembuluh darah pada kulit.",
      symptoms: ["Noda merah", "Benjolan kecil", "Nyeri ringan"],
      causes: ["Genetik", "Pertumbuhan abnormal pembuluh darah"],
      treatment: ["Eksisi jika perlu", "Pemantauan rutin"],
      severity: "Ringan",
      color: "from-red-400 to-pink-500",
      bgColor: "from-red-50 to-pink-50",
    },
    "Vasculitis Photos": {
      name: "Vaskulitis",
      description: "Peradangan pembuluh darah yang bisa menimbulkan bercak dan luka pada kulit.",
      symptoms: ["Bercak merah/ungu", "Nyeri", "Luka kecil"],
      causes: ["Autoimun", "Infeksi", "Obat"],
      treatment: ["Kortikosteroid", "Imunosupresan", "Pemantauan dokter"],
      severity: "Sedang - Tinggi",
      color: "from-purple-400 to-red-500",
      bgColor: "from-purple-50 to-red-50",
    },
    "Warts Molluscum and other Viral Infections": {
      name: "Kutil & Infeksi Virus",
      description: "Pertumbuhan kulit akibat virus seperti HPV atau molluscum contagiosum.",
      symptoms: ["Benjolan kecil", "Kutil menonjol", "Tidak gatal/nyeri"],
      causes: ["Virus HPV", "Molluscum contagiosum", "Kontak kulit"],
      treatment: ["Eksisi", "Krioterapi", "Perawatan simptomatik"],
      severity: "Ringan",
      color: "from-green-400 to-teal-500",
      bgColor: "from-green-50 to-teal-50",
    },
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setPrediction(null);
        
        // Create image element untuk analisis
        const img = new Image();
        img.onload = () => {
          setImageData(img);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const simulatePrediction = async () => {
    if (!fileInputRef.current?.files[0]) return;
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", fileInputRef.current.files[0]);

      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to get prediction");

      const data = await res.json();
      
      // Map backend response to frontend key
      const mappedDisease = backendToFrontendMapping[data.predicted_class] || data.predicted_class;
      
      setPrediction({
        disease: mappedDisease,
        confidence: data.confidence || 100,
      });
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Gagal melakukan prediksi. Pastikan backend sudah jalan.");
    }

    setIsLoading(false);
  };

  const resetApp = () => {
    setSelectedImage(null);
    setPrediction(null);
    setImageData(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getSeverityColor = (severity) => {
    if (severity.includes('Tinggi') || severity.includes('Darurat')) return 'text-red-500';
    if (severity.includes('Sedang')) return 'text-yellow-500';
    return 'text-green-500';
  };

  const Particles = () => (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {[...Array(50)].map((_, i) => (
        <div
          key={i}
          className="absolute animate-pulse"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 3}s`,
            animationDuration: `${3 + Math.random() * 2}s`
          }}
        >
          <div className={`w-1 h-1 rounded-full ${
            Math.random() > 0.5 ? 'bg-blue-400' : 'bg-purple-400'
          } opacity-30`} />
        </div>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-2000"></div>
      </div>

      {/* Particles */}
      {showParticles && <Particles />}

      {/* Glassmorphism Header */}
      <div className="relative z-10 backdrop-blur-xl bg-white/10 border-b border-white/20 shadow-2xl">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center justify-center space-x-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-full animate-ping opacity-75"></div>
              <div className="relative bg-gradient-to-r from-cyan-500 to-purple-600 p-4 rounded-full shadow-2xl">
                <Brain className="h-10 w-10 text-white" />
              </div>
            </div>
            <div className="text-center">
              <h1 className="text-5xl font-black bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
                AI SKIN DETECTOR
              </h1>
              <p className="text-xl text-gray-300 mt-2 font-medium">
                Powered by <span className="text-cyan-400">Huntrix Group</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-12">
        <div className="grid xl:grid-cols-2 gap-8">
          {/* Panel Upload & Prediksi */}
          <div className="space-y-8">
            {/* Upload Area dengan Glassmorphism */}
            <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-500 transform hover:scale-105">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <div className="bg-gradient-to-r from-cyan-500 to-purple-500 p-2 rounded-xl mr-3">
                  <Upload className="h-6 w-6 text-white" />
                </div>
                Upload Gambar Kulit
              </h2>
              
              <div className="relative">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="relative group cursor-pointer"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-2xl blur opacity-75 group-hover:opacity-100 transition-opacity duration-500 animate-pulse"></div>
                  <div className="relative border-4 border-dashed border-white/30 rounded-2xl p-12 text-center backdrop-blur-sm bg-white/5 group-hover:bg-white/10 transition-all duration-300">
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-full blur-lg opacity-50 animate-ping"></div>
                      <Camera className="relative h-20 w-20 mx-auto text-white mb-6" />
                    </div>
                    <p className="text-2xl font-bold text-white mb-2">Drag & Drop atau Klik</p>
                    <p className="text-lg text-gray-300">Format: JPG, PNG • Max: 10MB</p>
                    <div className="flex justify-center mt-4 space-x-2">
                      <Sparkles className="h-5 w-5 text-cyan-400 animate-pulse" />
                      <Sparkles className="h-5 w-5 text-purple-400 animate-pulse delay-100" />
                      <Sparkles className="h-5 w-5 text-pink-400 animate-pulse delay-200" />
                    </div>
                  </div>
                </div>
              </div>

              {selectedImage && (
                <div className="mt-8 relative">
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-pink-500 to-violet-600 rounded-2xl blur opacity-50 group-hover:opacity-75 transition-opacity duration-300"></div>
                    <div className="relative">
                      <img
                        src={selectedImage}
                        alt="Uploaded skin"
                        className="w-full h-80 object-cover rounded-2xl shadow-2xl border-4 border-white/20"
                      />
                      <div className="absolute top-4 right-4">
                        <button
                          onClick={resetApp}
                          className="bg-gradient-to-r from-red-500 to-pink-600 text-white px-6 py-2 rounded-full font-bold hover:from-red-600 hover:to-pink-700 transition-all duration-300 transform hover:scale-110 shadow-xl"
                        >
                          Reset
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  {!prediction && (
                    <div className="mt-6 relative">
                      <button
                        onClick={simulatePrediction}
                        disabled={isLoading}
                        className="w-full relative group"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 via-cyan-500 to-purple-600 rounded-2xl blur opacity-75 group-hover:opacity-100 transition-opacity duration-300 animate-pulse"></div>
                        <div className="relative bg-gradient-to-r from-emerald-600 via-cyan-600 to-purple-700 text-white py-6 px-8 rounded-2xl font-bold text-xl hover:from-emerald-700 hover:via-cyan-700 hover:to-purple-800 transition-all duration-300 transform hover:scale-105 shadow-2xl">
                          {isLoading ? (
                            <div className="flex items-center justify-center">
                              <div className="relative">
                                <div className="animate-spin rounded-full h-8 w-8 border-4 border-white/30 border-t-white mr-4"></div>
                                <div className="absolute inset-0 animate-ping rounded-full h-8 w-8 border-4 border-white/50"></div>
                              </div>
                              <span className="animate-pulse">AI Sedang Menganalisis Gambar...</span>
                            </div>
                          ) : (
                            <div className="flex items-center justify-center">
                              <Zap className="h-6 w-6 mr-3 animate-pulse" />
                              ANALISIS DENGAN AI
                              <ArrowRight className="h-6 w-6 ml-3 animate-bounce" />
                            </div>
                          )}
                        </div>
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Hasil Prediksi */}
            {prediction && (
              <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-8 border border-white/20 animate-fadeIn">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                  <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-2 rounded-xl mr-3 animate-pulse">
                    <CheckCircle2 className="h-6 w-6 text-white" />
                  </div>
                  Hasil Analisis AI
                </h2>
                
                <div className={`bg-gradient-to-br ${diseaseInfo[prediction.disease]?.bgColor} rounded-2xl p-8 border-2 border-white/30 shadow-2xl relative overflow-hidden`}>
                  <div className="absolute top-0 right-0 p-4">
                    <div className="flex space-x-1">
                      <Star className="h-4 w-4 text-yellow-400 animate-pulse" />
                      <Star className="h-4 w-4 text-yellow-400 animate-pulse delay-100" />
                      <Star className="h-4 w-4 text-yellow-400 animate-pulse delay-200" />
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-2xl font-black text-gray-800">DIAGNOSIS:</h3>
                    <div className={`px-4 py-2 rounded-full font-bold text-sm ${getSeverityColor(diseaseInfo[prediction.disease]?.severity || '')} bg-white/50`}>
                      <Shield className="inline h-4 w-4 mr-1" />
                      {diseaseInfo[prediction.disease]?.severity || 'Unknown'}
                    </div>
                  </div>
                  
                  <p className={`text-3xl font-black bg-gradient-to-r ${diseaseInfo[prediction.disease]?.color} bg-clip-text text-transparent mb-6`}>
                    {diseaseInfo[prediction.disease]?.name || prediction.disease}
                  </p>
                  
                  <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 mb-6 shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-gray-800 font-bold text-lg">Tingkat Keyakinan AI:</span>
                      <span className="text-4xl font-black text-emerald-600 animate-pulse">{prediction.confidence.toFixed(1)}%</span>
                    </div>
                    <div className="relative w-full bg-gray-200 rounded-full h-6 overflow-hidden">
                      <div
                        className={`h-6 rounded-full transition-all duration-2000 bg-gradient-to-r from-emerald-400 via-cyan-500 to-purple-500 relative`}
                        style={{ width: `${prediction.confidence}%` }}
                      >
                        <div className="absolute inset-0 bg-white/30 animate-pulse"></div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-yellow-100 to-orange-100 border-2 border-yellow-300 rounded-2xl p-6 shadow-xl">
                    <div className="flex items-start">
                      <div className="bg-yellow-500 p-2 rounded-full mr-4">
                        <AlertCircle className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <p className="text-yellow-800 font-bold text-lg mb-2">Penting!</p>
                        <p className="text-yellow-700 font-medium">Hasil ini adalah estimasi AI berdasarkan analisis visual. Untuk diagnosis yang akurat, konsultasikan dengan dokter spesialis kulit!</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Panel Informasi Penyakit */}
          {prediction && diseaseInfo[prediction.disease] && (
            <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-8 border border-white/20 animate-slideIn">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl mr-3">
                  <Info className="h-6 w-6 text-white" />
                </div>
                Informasi Lengkap
              </h2>
              
              <div className="space-y-8">
                {/* Deskripsi */}
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                  <h3 className="font-black text-white mb-4 text-xl flex items-center">
                    Deskripsi
                  </h3>
                  <p className="text-gray-200 leading-relaxed bg-white/5 p-4 rounded-xl border border-white/10">
                    {diseaseInfo[prediction.disease].description}
                  </p>
                </div>

                {/* Gejala */}
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                  <h3 className="font-black text-white mb-4 text-xl flex items-center">
                    Gejala Umum
                  </h3>
                  <div className="space-y-3">
                    {diseaseInfo[prediction.disease].symptoms.map((symptom, index) => (
                      <div key={index} className="flex items-center bg-red-500/20 backdrop-blur-sm p-4 rounded-xl border border-red-400/30 hover:bg-red-500/30 transition-all duration-300 transform hover:scale-105">
                        <div className="w-3 h-3 bg-gradient-to-r from-red-400 to-pink-500 rounded-full mr-4 animate-pulse"></div>
                        <span className="text-white font-medium">{symptom}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Penyebab */}
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                  <h3 className="font-black text-white mb-4 text-xl flex items-center">
                    Penyebab
                  </h3>
                  <div className="space-y-3">
                    {diseaseInfo[prediction.disease].causes.map((cause, index) => (
                      <div key={index} className="flex items-center bg-yellow-500/20 backdrop-blur-sm p-4 rounded-xl border border-yellow-400/30 hover:bg-yellow-500/30 transition-all duration-300 transform hover:scale-105">
                        <div className="w-3 h-3 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full mr-4 animate-pulse"></div>
                        <span className="text-white font-medium">{cause}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pengobatan */}
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                  <h3 className="font-black text-white mb-4 text-xl flex items-center">
                    Saran Pengobatan
                  </h3>
                  <div className="space-y-3">
                    {diseaseInfo[prediction.disease].treatment.map((treatment, index) => (
                      <div key={index} className="flex items-center bg-green-500/20 backdrop-blur-sm p-4 rounded-xl border border-green-400/30 hover:bg-green-500/30 transition-all duration-300 transform hover:scale-105">
                        <div className="w-3 h-3 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full mr-4 animate-pulse"></div>
                        <span className="text-white font-medium">{treatment}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Call to Action */}
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-2xl blur opacity-75 group-hover:opacity-100 transition-opacity duration-300 animate-pulse"></div>
                  <div className="relative bg-gradient-to-r from-cyan-600 via-purple-600 to-pink-600 text-white p-8 rounded-2xl shadow-2xl">
                    <h3 className="font-black text-2xl mb-4 flex items-center">
                      <Heart className="h-6 w-6 mr-2 animate-pulse" />
                      Langkah Selanjutnya
                    </h3>
                    <p className="mb-6 text-lg font-medium">Untuk diagnosis yang akurat dan pengobatan yang tepat, segera konsultasikan kondisi Anda dengan dokter spesialis kulit.</p>
                    <button className="bg-white text-purple-600 px-8 py-4 rounded-xl font-bold text-lg hover:bg-gray-100 transition-all duration-300 transform hover:scale-110 shadow-2xl flex items-center">
                      Cari Dokter Terdekat
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Info Panel ketika belum ada prediksi */}
          {!prediction && (
            <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl mr-3">
                  <Info className="h-6 w-6 text-white" />
                </div>
                Cara Penggunaan & Teknologi AI
              </h2>
              
              <div className="space-y-6">
                {[
                  { icon: '1', title: 'Upload Gambar', desc: 'Pilih foto kulit yang jelas dan berkualitas baik dengan pencahayaan yang cukup' },
                  { icon: '2', title: 'Analisis AI', desc: 'Sistem menganalisis warna, tekstur, dan pola menggunakan computer vision' },
                  { icon: '3', title: 'Hasil & Info', desc: 'Dapatkan prediksi berdasarkan karakteristik visual dan database medis' }
                ].map((step, index) => (
                  <div key={index} className="flex items-start space-x-4 bg-white/5 p-6 rounded-2xl border border-white/10 hover:bg-white/10 transition-all duration-300 transform hover:scale-105">
                    <div className="bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full p-3 text-white font-bold text-lg min-w-12 h-12 flex items-center justify-center">
                      {step.icon}
                    </div>
                    <div>
                      <h4 className="font-bold text-white text-lg flex items-center mb-2">
                        {step.title}
                      </h4>
                      <p className="text-gray-300 font-medium">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-6 rounded-2xl border border-purple-400/30 backdrop-blur-sm">
                <h4 className="font-black text-white mb-4 text-xl flex items-center">
                  <Sparkles className="h-6 w-6 mr-2 text-purple-400" />
                  Teknologi Analisis:
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  {[
                    'Analisis Warna RGB',
                    'Deteksi Tekstur',
                    'Pengukuran Bentuk',
                    'Edge Detection',
                    'Pattern Recognition',
                    'Statistical Analysis'
                  ].map((tech, index) => (
                    <div key={index} className="text-white font-medium bg-white/10 p-2 rounded-lg">
                      {tech}
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-6 bg-gradient-to-r from-green-500/20 to-emerald-500/20 p-6 rounded-2xl border border-green-400/30 backdrop-blur-sm">
                <h4 className="font-black text-white mb-4 text-xl flex items-center">
                  Dapat Mendeteksi:
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    'Jerawat dan Rosacea',
                    'Dermatitis Atopik (Eksim)',
                    'Keratosis Aktinik',
                    'Melanoma dan Tahi Lalat',
                    'Psoriasis dan Lichen Planus',
                    'Keratosis Seboroik'
                  ].map((item, index) => (
                    <div key={index} className="text-white font-medium bg-white/10 p-2 rounded-lg">
                      • {item}
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-6 bg-gradient-to-r from-red-500/20 to-orange-500/20 p-6 rounded-2xl border border-red-400/30 backdrop-blur-sm">
                <h4 className="font-black text-white mb-3 text-lg flex items-center">
                  <AlertCircle className="h-5 w-5 mr-2 text-red-400" />
                  Tips untuk Hasil Terbaik:
                </h4>
                <ul className="text-gray-200 space-y-2">
                  <li>• Gunakan pencahayaan yang baik dan natural</li>
                  <li>• Pastikan area kulit terlihat jelas tanpa bayangan</li>
                  <li>• Hindari flash yang terlalu terang</li>
                  <li>• Fokus pada satu area masalah kulit</li>
                  <li>• Gunakan resolusi gambar yang cukup tinggi</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Canvas tersembunyi untuk analisis */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.8s ease-out; }
        .animate-slideIn { animation: slideIn 0.8s ease-out; }
      `}</style>
    </div>
  );
};

export default SkinDiseasePredictor;