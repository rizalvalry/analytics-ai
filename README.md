# GAIKINDO Data Analytics Platform

Platform analitik data untuk industri otomotif Indonesia dengan integrasi AI untuk query natural language dan visualisasi data penjualan kendaraan.

## 📋 **Fitur Utama**

- 🔍 **Natural Language Query** - Tanyakan data dalam bahasa Indonesia
- 📊 **Data Visualization** - Charts dan grafik interaktif
- 🔄 **Real-time Processing** - Update data dari Azure Blob Storage
- 🤖 **AI Integration** - Generative AI untuk memahami query
- 📈 **Analytics Dashboard** - Summary dan trend analysis

## 🏗️ **Arsitektur Sistem**

```
Azure Blob Storage → Data Processor → Parquet Files → Streamlit UI
       ↓                    ↓              ↓              ↓
   Excel Files      Processing       Analytics     User Interface
   (Raw Data)       Pipeline        Views         (Web App)
```

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- Azure Storage Account dengan Blob Container
- Environment variables untuk Azure credentials

### **1. Clone Repository**

```bash
git clone <repository-url>
cd gaikindo-analytics
```

### **2. Setup Environment**

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.temp .env

# Edit .env file dengan Azure credentials Anda
```

### **3. Konfigurasi Azure Credentials**

Edit file `.env` dan isi dengan informasi Azure Anda:

```env
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
AZURE_CONTAINER_NAME=gaikindo
```

### **4. Jalankan Data Processor**

**PENTING:** Jalankan processor terlebih dahulu untuk memproses data!

```bash
python gaikindo_processor_updated.py
```

Processor akan:
- ✅ Mengunduh Excel files dari Azure Blob Storage
- ✅ Memproses data kendaraan dan penjualan
- ✅ Membuat summary/total rows untuk queries
- ✅ Generate analytics views (monthly, brand comparison, dll)

### **5. Jalankan Streamlit Application**

```bash
streamlit run gaikindo_app.py
```

Aplikasi akan tersedia di: `http://localhost:8501`

## 📁 **Struktur File**

```
gaikindo-analytics/
├── gaikindo_processor_updated.py    # Data processing pipeline
├── gaikindo_app.py                  # Streamlit web application
├── config/
│   └── app_config.py               # Application configuration
├── data/
│   ├── gaikindo_master_ACCURATE.parquet    # Complete dataset
│   ├── gaikindo_summary.parquet            # Summary/total rows
│   ├── gaikindo_detailed.parquet           # Individual records
│   ├── gaikindo_monthly.parquet            # Monthly aggregations
│   ├── gaikindo_brand_comparison.parquet   # Brand comparisons
│   ├── gaikindo_specs_only.parquet         # Specifications only
│   ├── gaikindo_sales_only.parquet         # Sales data only
│   ├── processing_metadata_enhanced.json   # Processing metadata
│   └── data_separation_metadata.json       # Validation metadata
├── .env                                # Environment variables
├── .env.temp                          # Environment template
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🔧 **Konfigurasi Detail**

### **Azure Blob Storage Setup**

1. Buat Azure Storage Account
2. Buat Blob Container bernama `gaikindo`
3. Upload Excel files dengan struktur:
   ```
   gaikindo/
   ├── 2018/
   │   ├── TOYOTA.xlsx
   │   ├── HINO.xlsx
   │   └── ...
   ├── 2019/
   │   └── ...
   └── ...
   ```

### **Environment Variables**

```env
# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=gaikindo

# Optional: AI Integration
OPENAI_API_KEY=your_openai_key_here
```

## 📊 **Cara Penggunaan**

### **Query Examples**

1. **Total Penjualan per Brand:**
   ```
   Berapa total penjualan Toyota tahun 2023?
   ```

2. **Perbandingan Antar Brand:**
   ```
   Bandingkan penjualan wholesale Toyota vs Mercedes Benz tahun 2023
   ```

3. **Trend Analysis:**
   ```
   Tunjukkan trend penjualan HINO dari 2018 sampai 2023
   ```

4. **Monthly Data:**
   ```
   Penjualan bulanan Isuzu tahun 2022
   ```

### **Filter Options**

- **Brand:** TOYOTA, HINO, ISUZU, MERCEDES_BENZ, SCANIA, TATA_MOTORS, FAW, UD_TRUCKS
- **Sales Type:** wholesales, retail_sales, production_ckd, import_cbu
- **Year:** 2018-2025
- **Segment:** TOTAL (untuk summary) atau nama segment spesifik

## 🔄 **Update Data Rutin**

### **Monthly Update Process:**

1. **Upload Excel baru** ke Azure Blob Storage
2. **Jalankan processor:**
   ```bash
   python gaikindo_processor_updated.py
   ```
3. **Restart aplikasi** jika sedang running:
   ```bash
   # Stop aplikasi (Ctrl+C)
   streamlit run gaikindo_app.py
   ```

## 🐛 **Troubleshooting**

### **Error: No data found**

**Solusi:**
1. Pastikan processor sudah dijalankan: `python gaikindo_processor_updated.py`
2. Periksa file `.env` sudah benar
3. Verifikasi Excel files ada di Azure Blob Storage

### **Error: Azure connection failed**

**Solusi:**
1. Periksa `AZURE_STORAGE_CONNECTION_STRING` di `.env`
2. Pastikan Azure Storage Account aktif
3. Verifikasi network connectivity

### **Error: Module not found**

**Solusi:**
```bash
pip install -r requirements.txt
```

### **App tidak loading**

**Solusi:**
1. Restart processor: `python gaikindo_processor_updated.py`
2. Restart Streamlit: `streamlit run gaikindo_app.py`
3. Clear browser cache

## 📈 **Performance Tips**

- **Data Processing:** Processor membutuhkan waktu 2-5 menit tergantung jumlah data
- **Memory Usage:** Pastikan RAM minimal 8GB untuk processing besar
- **Storage:** Siapkan space 2-3GB untuk file parquet
- **Network:** Koneksi internet stabil untuk download dari Azure

## 🤝 **Support**

Untuk pertanyaan atau issues:
1. Periksa bagian Troubleshooting di atas
2. Verifikasi semua prerequisites terpenuhi
3. Pastikan mengikuti urutan deployment yang benar

## 📝 **Catatan Penting**

- **Processor** hanya perlu dijalankan sekali atau saat ada data baru
- **App** bisa dijalankan berkali-kali setelah processor selesai
- Backup file `.env` dan jangan commit ke repository
- Monitor Azure costs untuk storage dan data transfer

---

**Dibuat dengan ❤️ untuk industri otomotif Indonesia**
