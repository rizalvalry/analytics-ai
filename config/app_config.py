import os
from dotenv import load_dotenv

load_dotenv("config/.env")

# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
INPUT_CONTAINER = "test-01"

# Google AI Configuration
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Data Configuration
LOCAL_PARQUET_FILE = "data/master_data_fixed.parquet"
DATA_FOLDER = "data"
TEMP_FOLDER = "temp"

# Streamlit Configuration
PAGE_TITLE = "GAIKINDO Sales Analysis Platform"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# Model Configuration
MODEL_NAME = "gemini-2.5-pro"
MODEL_JSON_CONFIG = {"response_mime_type": "application/json"}

# Available data metadata
AVAILABLE_BRANDS = [
    "TOYOTA", "HINO", "ISUZU", "MERCEDES_BENZ", 
    "UD_TRUCKS", "FAW", "TATA_MOTORS", "SCANIA"
]

AVAILABLE_SALES_TYPES = [
    "wholesales", "retail_sales", "production_ckd", "import_cbu"
]

MONTH_MAPPING = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 
    5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 
    9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

MONTH_NAMES_ID = {
    'jan': 'Januari', 'feb': 'Februari', 'mar': 'Maret', 'apr': 'April',
    'may': 'Mei', 'jun': 'Juni', 'jul': 'Juli', 'aug': 'Agustus',
    'sep': 'September', 'oct': 'Oktober', 'nov': 'November', 'dec': 'Desember'
}