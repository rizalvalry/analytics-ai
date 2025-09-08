# Fixed version - extract only the final TOTAL row values
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv
import traceback

load_dotenv()

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
INPUT_CONTAINER = "test-01"
LOCAL_PARQUET_FILE = "master_data_fixed.parquet"

def process_and_save_data():
    """
    Fixed ETL process - extract only TOTAL row values to avoid double counting
    """
    print("--- STARTING FIXED ETL PROCESS ---")
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        input_container_client = blob_service_client.get_container_client(INPUT_CONTAINER)
    except Exception as e:
        print(f"FATAL: Failed to connect to Azure Blob Storage. Error: {e}")
        return

    all_cleaned_data = []
    blobs = list(input_container_client.list_blobs())
    
    print(f"Found {len(blobs)} files in container '{INPUT_CONTAINER}'. Starting processing...")

    for blob in blobs:
        if not blob.name.endswith('.xlsx'):
            continue

        print(f"\n-> Processing file: {blob.name}...")
        try:
            path_parts = blob.name.split('/')
            if len(path_parts) < 2 or not path_parts[0].isdigit() or not path_parts[1].isdigit():
                print("   SKIPPED: Invalid folder structure.")
                continue
            
            year = int(path_parts[0])
            brand_name = os.path.splitext(os.path.basename(blob.name))[0].upper()

            blob_client = input_container_client.get_blob_client(blob)
            downloader = blob_client.download_blob()
            excel_content = io.BytesIO(downloader.readall())
            
            # Read raw data starting from row 10
            df_raw = pd.read_excel(excel_content, header=None, skiprows=10, engine='openpyxl')
            
            # Find the final TOTAL row (not SUBTOTAL)
            total_row_idx = None
            for i, row in df_raw.iterrows():
                segment_val = str(row.iloc[2]) if pd.notna(row.iloc[2]) else ''
                if segment_val.strip().upper() == 'TOTAL:':
                    total_row_idx = i
                    break
            
            if total_row_idx is None:
                print(f"   SKIPPED: No TOTAL row found in {brand_name}")
                continue
                
            total_row = df_raw.iloc[total_row_idx]
            print(f"   -> Found TOTAL row for {brand_name}")
            
            # Extract sales data from TOTAL row
            # WHOLESALES: cols 22-33 (JAN-DEC)
            # RETAIL SALES: cols 35-46 (JAN-DEC) 
            # PRODUCTION CKD: cols 48-59 (JAN-DEC)
            # IMPORT CBU: cols 61-72 (JAN-DEC)
            
            sales_sections = {
                'wholesales': list(range(22, 34)),     # 22-33
                'retail_sales': list(range(35, 47)),   # 35-46
                'production_ckd': list(range(48, 60)), # 48-59
                'import_cbu': list(range(61, 73))      # 61-72
            }
            
            month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            
            for sales_type, col_range in sales_sections.items():
                for i, col_idx in enumerate(col_range[:12]):  # Only first 12 months
                    if i < len(month_names) and col_idx < len(total_row):
                        sales_value = total_row.iloc[col_idx] if pd.notna(total_row.iloc[col_idx]) else 0
                        
                        if sales_value > 0:  # Only include positive sales
                            record = {
                                'brand': brand_name,
                                'segment': 'TOTAL',  # Mark as aggregated total
                                'model': 'ALL_MODELS',
                                'month': month_names[i],
                                'sales_type': sales_type,
                                'sales': float(sales_value),
                                'year': year
                            }
                            all_cleaned_data.append(record)

            print(f"   SUCCESS: {brand_name} processed with TOTAL row data")

        except Exception as e:
            print(f"   ERROR: Error processing {blob.name}")
            traceback.print_exc()
        
    if not all_cleaned_data:
        print("\n--- ETL FAILED: No data processed from any file. ---")
        return

    # Convert to DataFrame and save
    master_df = pd.DataFrame(all_cleaned_data)
    master_df.to_parquet(LOCAL_PARQUET_FILE, index=False)
    
    print(f"\n--- ETL COMPLETED ---")
    print(f"Total {len(master_df)} clean records saved to '{LOCAL_PARQUET_FILE}'")
    
    # Show summary by brand and sales type
    print("\nSummary by brand and sales type:")
    summary = master_df.groupby(['brand', 'sales_type']).size().unstack(fill_value=0)
    print(summary)

if __name__ == "__main__":
    process_and_save_data()