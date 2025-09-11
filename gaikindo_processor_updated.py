"""
GAIKINDO Master Data Processor - Updated for New Excel Structure
===============================================================

This updated processor handles the new Excel format with:
1. Dynamic month column detection
2. Proper header parsing for merged cells
3. Enhanced technical specification extraction
4. Separate sales and specifications data handling

Author: AI Assistant
Date: 2025-01-21
"""

import os
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv
import traceback
import json
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class GaikindoMasterProcessorUpdated:
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.input_container = "gaikindo"
        self.local_data_dir = "data/processed"
        self.master_file = "data/gaikindo_master_ACCURATE.parquet"
        self.metadata_file = "data/processing_metadata_enhanced.json"

        # Create directories
        os.makedirs(self.local_data_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Expected brands
        self.expected_brands = [
            'FAW', 'HINO', 'ISUZU', 'MERCEDES_BENZ',
            'SCANIA', 'TATA_MOTORS', 'TOYOTA', 'UD_TRUCKS'
        ]

        # Month names for detection
        self.month_names = [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]

        # Processing metadata
        self.metadata = {
            'processing_date': datetime.now().isoformat(),
            'brands_processed': [],
            'files_processed': [],
            'total_records': 0,
            'detailed_records': 0,
            'summary_records': 0,
            'errors': []
        }

    def connect_to_azure(self):
        """Establish Azure Blob Storage connection"""
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.input_container)
            print(f"[SUCCESS] Connected to Azure Blob Storage container: {self.input_container}")
            return True
        except Exception as e:
            print(f"[ERROR] FATAL: Failed to connect to Azure Blob Storage. Error: {e}")
            self.metadata['errors'].append(f"Azure connection failed: {str(e)}")
            return False

    def find_data_start_row(self, df_raw):
        """Find the row where actual data starts (after headers and metadata)"""
        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            # Look for rows that have VR-ID or actual data patterns
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                row_text = str(row.iloc[0]).strip().upper()
                if 'VR-ID' in row_text or any(char.isdigit() for char in row_text[:10]):
                    return i
        return 9  # Default fallback

    def parse_column_structure(self, df_raw, header_row_idx):
        """Parse the complex column structure with merged headers"""
        column_mapping = {}

        # Find month columns for each sales type
        sales_types = ['WHOLESALES', 'RETAIL SALES', 'PRODUCTION (CKD)', 'IMPORT (CBU)']

        for sales_type in sales_types:
            # Find columns containing this sales type
            sales_cols = []
            for col_idx in range(len(df_raw.columns)):
                col_values = df_raw.iloc[:20, col_idx].astype(str).str.upper()
                if any(sales_type in val for val in col_values if pd.notna(val)):
                    sales_cols.append(col_idx)

            if sales_cols:
                # Find month columns under this sales type
                month_cols = {}
                for col_idx in range(min(sales_cols), max(sales_cols) + 20):
                    if col_idx >= len(df_raw.columns):
                        break

                    col_header = str(df_raw.iloc[header_row_idx, col_idx]).strip().upper()
                    for month_name in self.month_names:
                        if month_name.upper() in col_header:
                            month_cols[month_name] = col_idx
                            break

                column_mapping[sales_type.lower().replace(' ', '_').replace('(', '').replace(')', '')] = {
                    'month_columns': month_cols,
                    'description': sales_type
                }

        return column_mapping

    def extract_vehicle_data(self, df_data, brand, year):
        """Extract vehicle specifications and sales data from the new format"""
        all_records = []

        try:
            # Based on the Excel structure analysis, use fixed column positions
            # Column mapping based on actual Excel structure:
            col_mapping = {
                'vr_id': 0,           # Column 0: VR-ID
                'no': 1,              # Column 1: NO.
                'segment': 2,         # Column 2: SEGMENT
                'model': 3,           # Column 3: MODEL
                'cc': 4,              # Column 4: CC
                'transmission': 5,    # Column 5: TRANS
                'fuel': 6,            # Column 6: FUEL
                'fc_kml': 7,          # Column 7: FC (KM/L)
                'co2': 9,             # Column 9: CO2
                'tank_cap': 10,       # Column 10: TANK CAP (L)
                'gvw': 11,            # Column 11: GVW (Kg)
                'gear_ratio': 12,     # Column 12: GEAR RATIO
                'tyre_size': 13,      # Column 13: TYRE SIZE
                'ps_hp': 14,          # Column 14: PS/HP
                'wheel_base': 15,     # Column 15: WHEEL BASE
                'dimension': 16,      # Column 16: DIMENSION
                'seater': 17,         # Column 17: SEATER
                'drive_system': 18,   # Column 18: DRIVE SYSTEM
                'speed': 20,          # Column 20: SPEED
                'door': 21,           # Column 21: DOOR
                # Sales data columns (JAN-DEC for each sales type)
                'wholesale_jan': 22, 'wholesale_feb': 23, 'wholesale_mar': 24, 'wholesale_apr': 25,
                'wholesale_may': 26, 'wholesale_jun': 27, 'wholesale_jul': 28, 'wholesale_aug': 29,
                'wholesale_sep': 30, 'wholesale_oct': 31, 'wholesale_nov': 32, 'wholesale_dec': 33,
                'retail_jan': 35, 'retail_feb': 36, 'retail_mar': 37, 'retail_apr': 38,
                'retail_may': 39, 'retail_jun': 40, 'retail_jul': 41, 'retail_aug': 42,
                'retail_sep': 43, 'retail_oct': 44, 'retail_nov': 45, 'retail_dec': 46,
                'ckd_jan': 48, 'ckd_feb': 49, 'ckd_mar': 50, 'ckd_apr': 51,
                'ckd_may': 52, 'ckd_jun': 53, 'ckd_jul': 54, 'ckd_aug': 55,
                'ckd_sep': 56, 'ckd_oct': 57, 'ckd_nov': 58, 'ckd_dec': 59,
                'cbu_jan': 61, 'cbu_feb': 62, 'cbu_mar': 63, 'cbu_apr': 64,
                'cbu_may': 65, 'cbu_jun': 66, 'cbu_jul': 67, 'cbu_aug': 68,
                'cbu_sep': 69, 'cbu_oct': 70, 'cbu_nov': 71, 'cbu_dec': 72
            }

            print(f"   [COLUMNS] Using fixed column mapping for {brand}")

            # Process each row starting from data rows (skip header rows)
            for idx, row in df_data.iterrows():
                try:
                    # Skip header/metadata rows - look for rows with VR-ID pattern
                    vr_id_cell = str(row.iloc[col_mapping['vr_id']]).strip() if col_mapping['vr_id'] < len(row) else ""
                    if not vr_id_cell or vr_id_cell == 'nan':
                        continue

                    # More flexible VR-ID pattern matching - look for alphanumeric strings that look like vehicle IDs
                    # Skip obvious header rows
                    if vr_id_cell.upper() in ['COMPANY', 'BRAND', 'DATA OF MONTH/YEAR', 'VR-ID', 'PIC']:
                        continue

                    # Check if it looks like a vehicle ID (alphanumeric, reasonable length)
                    if not (len(vr_id_cell) >= 8 and any(char.isalnum() for char in vr_id_cell)):
                        continue

                    # Extract basic vehicle information
                    vr_id = vr_id_cell
                    segment = str(row.iloc[col_mapping['segment']]).strip() if col_mapping['segment'] < len(row) and pd.notna(row.iloc[col_mapping['segment']]) else "Unknown"
                    model = str(row.iloc[col_mapping['model']]).strip() if col_mapping['model'] < len(row) and pd.notna(row.iloc[col_mapping['model']]) else "Unknown"

                    # Extract technical specifications
                    specs = {
                        'cc': str(row.iloc[col_mapping['cc']]).strip() if col_mapping['cc'] < len(row) and pd.notna(row.iloc[col_mapping['cc']]) else '',
                        'transmission': str(row.iloc[col_mapping['transmission']]).strip() if col_mapping['transmission'] < len(row) and pd.notna(row.iloc[col_mapping['transmission']]) else '',
                        'fuel': str(row.iloc[col_mapping['fuel']]).strip() if col_mapping['fuel'] < len(row) and pd.notna(row.iloc[col_mapping['fuel']]) else '',
                        'gvw': str(row.iloc[col_mapping['gvw']]).strip() if col_mapping['gvw'] < len(row) and pd.notna(row.iloc[col_mapping['gvw']]) else '',
                        'ps_hp': str(row.iloc[col_mapping['ps_hp']]).strip() if col_mapping['ps_hp'] < len(row) and pd.notna(row.iloc[col_mapping['ps_hp']]) else '',
                        'dimension': str(row.iloc[col_mapping['dimension']]).strip() if col_mapping['dimension'] < len(row) and pd.notna(row.iloc[col_mapping['dimension']]) else '',
                        'seater': str(row.iloc[col_mapping['seater']]).strip() if col_mapping['seater'] < len(row) and pd.notna(row.iloc[col_mapping['seater']]) else '',
                        'wheel': str(row.iloc[col_mapping['wheel_base']]).strip() if col_mapping['wheel_base'] < len(row) and pd.notna(row.iloc[col_mapping['wheel_base']]) else '',
                        'tank': str(row.iloc[col_mapping['tank_cap']]).strip() if col_mapping['tank_cap'] < len(row) and pd.notna(row.iloc[col_mapping['tank_cap']]) else '',
                        'gear': str(row.iloc[col_mapping['gear_ratio']]).strip() if col_mapping['gear_ratio'] < len(row) and pd.notna(row.iloc[col_mapping['gear_ratio']]) else '',
                        'system': str(row.iloc[col_mapping['drive_system']]).strip() if col_mapping['drive_system'] < len(row) and pd.notna(row.iloc[col_mapping['drive_system']]) else '',
                        'speed': str(row.iloc[col_mapping['speed']]).strip() if col_mapping['speed'] < len(row) and pd.notna(row.iloc[col_mapping['speed']]) else '',
                        'door': str(row.iloc[col_mapping['door']]).strip() if col_mapping['door'] < len(row) and pd.notna(row.iloc[col_mapping['door']]) else ''
                    }

                    # Extract sales data for each sales type and month
                    sales_types = {
                        'wholesales': 'wholesale',
                        'retail_sales': 'retail',
                        'production_ckd': 'ckd',
                        'import_cbu': 'cbu'
                    }

                    for sales_type, prefix in sales_types.items():
                        for month in self.month_names:
                            col_name = f"{prefix}_{month}"
                            if col_name in col_mapping:
                                col_idx = col_mapping[col_name]
                                if col_idx < len(row):
                                    sales_value = row.iloc[col_idx]
                                    if pd.notna(sales_value) and sales_value != 0:
                                        try:
                                            sales_numeric = float(sales_value)
                                            if sales_numeric > 0:
                                                record = {
                                                    'brand': brand,
                                                    'segment': segment,
                                                    'model': model,
                                                    'month': month,
                                                    'sales_type': sales_type,
                                                    'sales': int(sales_numeric),
                                                    'year': year,
                                                    'is_summary': False,  # Individual vehicle records
                                                    'row_index': idx,
                                                    'vr_id': vr_id,
                                                    **specs
                                                }
                                                all_records.append(record)
                                        except (ValueError, TypeError):
                                            continue

                    # If no sales data found for this vehicle, still create records with 0 sales
                    # This ensures we capture all vehicle specifications even if no sales
                    if not any(r['vr_id'] == vr_id for r in all_records[-48:]):  # Check last 48 records (4 sales types * 12 months)
                        for sales_type in ['wholesales', 'retail_sales', 'production_ckd', 'import_cbu']:
                            for month in self.month_names:
                                record = {
                                    'brand': brand,
                                    'segment': segment,
                                    'model': model,
                                    'month': month,
                                    'sales_type': sales_type,
                                    'sales': 0,
                                    'year': year,
                                    'is_summary': False,
                                    'row_index': idx,
                                    'vr_id': vr_id,
                                    **specs
                                }
                                all_records.append(record)

                except Exception as e:
                    print(f"   [WARNING] Error processing row {idx}: {e}")
                    continue

            print(f"   [SUCCESS] Extracted {len(all_records)} records from {brand}")
            return pd.DataFrame(all_records)

        except Exception as e:
            print(f"   [ERROR] Error extracting data from {brand}: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def process_brand_file(self, blob_name, brand, year):
        """Process individual brand Excel file with updated structure"""
        try:
            print(f"\n[PROCESSING] Processing: {blob_name}")
            print(f"   [YEAR] Year: {year}")
            print(f"   [BRAND] Brand: {brand}")

            # Download blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.input_container, blob=blob_name
            )
            downloader = blob_client.download_blob()
            excel_content = io.BytesIO(downloader.readall())

            # Read Excel file
            df_raw = pd.read_excel(excel_content, header=None)

            # Find data start row
            data_start_row = self.find_data_start_row(df_raw)
            print(f"   [STRUCTURE] Data starts at row: {data_start_row}")

            # Read data with proper headers
            df_data = pd.read_excel(excel_content, header=data_start_row)

            # Extract specifications and sales data
            processed_data = self.extract_vehicle_data(df_data, brand, year)

            if not processed_data.empty:
                # Save individual brand file
                output_file = f"data/processed/{brand}_{year}.parquet"
                processed_data.to_parquet(output_file, index=False)
                print(f"   [SUCCESS] Saved to: {output_file}")
                print(f"   [RECORDS] Records extracted: {len(processed_data)}")
            else:
                print(f"   [WARNING] No data extracted from {brand} {year}")

            return processed_data

        except Exception as e:
            print(f"   [ERROR] Error processing {blob_name}: {e}")
            return pd.DataFrame()

    def process_all_brands(self):
        """Process all brand files from Azure Blob Storage"""
        if not self.connect_to_azure():
            return False

        print("\n[START] Starting GAIKINDO Master Data Processing (Updated)")
        print("=" * 60)

        # Get all Excel files
        blobs = list(self.container_client.list_blobs())
        excel_files = [blob.name for blob in blobs if blob.name.endswith('.xlsx')]

        print(f"[FILES] Found {len(excel_files)} Excel files in container")

        if not excel_files:
            print("[ERROR] No Excel files found in container")
            return False

        # Process each file
        all_data = []
        for blob_name in excel_files:
            # Parse filename to get brand and year
            path_parts = blob_name.split('/')
            if len(path_parts) >= 2:
                year_str = path_parts[0]
                filename = path_parts[1]
                brand_name = filename.replace('.xlsx', '').upper()

                if year_str.isdigit() and brand_name in self.expected_brands:
                    year = int(year_str)
                    processed_data = self.process_brand_file(blob_name, brand_name, year)
                    if not processed_data.empty:
                        all_data.append(processed_data)

        if not all_data:
            print("\n[ERROR] No data processed from any file")
            return False

        # Combine all data
        master_df = pd.concat(all_data, ignore_index=True)

        # Save master file
        master_df.to_parquet(self.master_file, index=False)
        print(f"[SAVED] Master file saved: {self.master_file}")

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"[METADATA] Metadata saved: {self.metadata_file}")

        # Print summary
        self.print_processing_summary(master_df)

        return True

    def print_processing_summary(self, df):
        """Print comprehensive processing summary"""
        print("\n" + "="*60)
        print("[SUMMARY] PROCESSING SUMMARY")
        print("="*60)

        print(f"[SUCCESS] Total Records Processed: {len(df):,}")
        print(f"[BRANDS] Brands Processed: {df['brand'].nunique()}")
        print(f"[YEARS] Years Covered: {sorted(df['year'].unique())}")
        print(f"[MODELS] Unique Models: {df['model'].nunique()}")

        print(f"\n[BRANDS] Records by Brand:")
        brand_summary = df.groupby('brand').agg({
            'sales': ['count', 'sum'],
            'model': 'nunique'
        }).round(2)
        brand_summary.columns = ['Record_Count', 'Total_Sales', 'Unique_Models']
        print(brand_summary.to_string())

        if self.metadata['errors']:
            print(f"\n[ERRORS] Processing Errors ({len(self.metadata['errors'])}):")
            for error in self.metadata['errors'][:5]:
                print(f"   - {error}")

        print("\n[COMPLETE] Processing completed successfully!")
        print("="*60)

    def create_analytics_views(self):
        """Create pre-computed analytics views for fast querying"""
        if not os.path.exists(self.master_file):
            print("[ERROR] Master file not found. Run process_all_brands() first.")
            return False

        print("\n[VIEWS] Creating analytics views...")
        df = pd.read_parquet(self.master_file)

        # Create summary/total rows by aggregating sales data
        print("[AGGREGATION] Creating summary/total rows...")

        # Group by brand, year, sales_type to create TOTAL rows
        summary_aggregations = []
        for (brand, year, sales_type), group in df.groupby(['brand', 'year', 'sales_type']):
            total_sales = group['sales'].sum()

            # Create summary record for this brand/year/sales_type combination
            summary_record = {
                'brand': brand,
                'segment': f'TOTAL {brand}',
                'model': f'TOTAL {brand}',
                'month': 'all',  # Aggregate across all months
                'sales_type': sales_type,
                'sales': int(total_sales),
                'year': year,
                'is_summary': True,
                'row_index': -1,  # Special marker for aggregated rows
                'vr_id': f'TOTAL_{brand}_{year}_{sales_type}',
                # Set technical specs to empty/null for summary rows
                'cc': '',
                'transmission': '',
                'fuel': '',
                'gvw': '',
                'wheel': '',
                'ps_hp': '',
                'dimension': '',
                'seater': ''
            }
            summary_aggregations.append(summary_record)

        # Convert to DataFrame and add to master data
        if summary_aggregations:
            summary_df_new = pd.DataFrame(summary_aggregations)
            df_with_summary = pd.concat([df, summary_df_new], ignore_index=True)
            print(f"[SUCCESS] Added {len(summary_aggregations)} summary/total rows")

            # Save updated master file with summary rows
            df_with_summary.to_parquet(self.master_file, index=False)
            print(f"[SUCCESS] Updated master file with summary rows: {self.master_file}")
        else:
            df_with_summary = df
            print("[WARNING] No summary rows created")

        # Summary view (now contains the TOTAL rows we just created)
        summary_df = df_with_summary[df_with_summary['is_summary'] == True].copy()
        summary_df = summary_df.drop(columns=['is_summary', 'row_index'], errors='ignore')
        summary_df.to_parquet("data/gaikindo_summary.parquet", index=False)
        print(f"[SUCCESS] Created summary view: data/gaikindo_summary.parquet ({len(summary_df)} records)")

        # Detailed view (individual vehicle records)
        detailed_df = df_with_summary[df_with_summary['is_summary'] == False].copy()
        detailed_df = detailed_df.drop(columns=['is_summary', 'row_index'], errors='ignore')
        detailed_df.to_parquet("data/gaikindo_detailed.parquet", index=False)
        print(f"[SUCCESS] Created detailed view: data/gaikindo_detailed.parquet ({len(detailed_df)} records)")

        # Monthly aggregation view
        monthly_agg = df_with_summary.groupby(['brand', 'segment', 'month', 'sales_type', 'year']).agg({
            'sales': 'sum'
        }).reset_index()
        monthly_agg.to_parquet("data/gaikindo_monthly.parquet", index=False)
        print(f"[SUCCESS] Created monthly aggregation: data/gaikindo_monthly.parquet ({len(monthly_agg)} records)")

        # Brand comparison view
        brand_comparison = df_with_summary.groupby(['brand', 'sales_type', 'year']).agg({
            'sales': 'sum'
        }).reset_index()
        brand_comparison.to_parquet("data/gaikindo_brand_comparison.parquet", index=False)
        print(f"[SUCCESS] Created brand comparison: data/gaikindo_brand_comparison.parquet ({len(brand_comparison)} records)")

        return True

    def create_separate_data_files(self):
        """Create separate parquet files for sales and specifications data"""
        if not os.path.exists(self.master_file):
            print("[ERROR] Master file not found. Run process_all_brands() first.")
            return False

        print("\n[SEPARATION] Creating separate sales and specifications files...")
        df = pd.read_parquet(self.master_file)

        # Create specifications-only file
        specs_columns = [
            'brand', 'segment', 'model', 'year', 'month', 'vr_id',
            'cc', 'transmission', 'fuel', 'gvw', 'wheel', 'ps_hp',
            'dimension', 'seater'
        ]

        existing_specs_cols = [col for col in specs_columns if col in df.columns]
        specs_df = df[existing_specs_cols].drop_duplicates()
        specs_df.to_parquet("data/gaikindo_specs_only.parquet", index=False)
        print("[SUCCESS] Created specifications file: data/gaikindo_specs_only.parquet")
        print(f"   - Records: {len(specs_df)}")
        print(f"   - Unique models: {specs_df['model'].nunique()}")

        # Create sales-only file
        sales_columns = [
            'brand', 'segment', 'model', 'year', 'month',
            'sales_type', 'sales', 'is_summary', 'vr_id'
        ]
        existing_sales_cols = [col for col in sales_columns if col in df.columns]
        sales_df = df[existing_sales_cols]
        sales_df.to_parquet("data/gaikindo_sales_only.parquet", index=False)
        print("[SUCCESS] Created sales file: data/gaikindo_sales_only.parquet")
        print(f"   - Records: {len(sales_df)}")
        print(f"   - Total sales: {sales_df['sales'].sum():,.0f}")

        # Create validation metadata
        validation_metadata = {
            'created_at': datetime.now().isoformat(),
            'specs_records': len(specs_df),
            'sales_records': len(sales_df),
            'master_records': len(df),
            'data_integrity_check': len(df) == len(sales_df),
            'specs_coverage': len(specs_df) / len(df) * 100 if len(df) > 0 else 0,
            'brands_covered': sorted(df['brand'].unique().tolist()),
            'sales_types': sorted(df['sales_type'].unique().tolist()) if 'sales_type' in df.columns else []
        }

        with open("data/data_separation_metadata.json", 'w') as f:
            json.dump(validation_metadata, f, indent=2)
        print("[SUCCESS] Created validation metadata: data/data_separation_metadata.json")

        return True

def main():
    """Main execution function"""
    processor = GaikindoMasterProcessorUpdated()

    # Process all brands
    if processor.process_all_brands():
        # Create analytics views
        processor.create_analytics_views()
        # Create separate sales and specs files
        processor.create_separate_data_files()
        print("\n[COMPLETE] All processing completed successfully!")
        return True
    else:
        print("\n[FAILED] Processing failed!")
        return False

if __name__ == "__main__":
    main()
