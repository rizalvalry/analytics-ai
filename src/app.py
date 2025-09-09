# File: app.py
import os
import sys
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import streamlit as st

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.data_visualizer import GaikindoVisualizer
from analytics.insights_generator import GaikindoInsightsGenerator

# --- 1. KONFIGURASI DAN SETUP ---
load_dotenv("config/.env")

LOCAL_PARQUET_FILE = "data/master_data_fixed.parquet"
# --- PERUBAHAN PENTING: Gunakan nama environment variable yang baru ---
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google Generative AI with API key
try:
    # Try GOOGLE_API_KEY first, then GOOGLE_CLOUD_API_KEY as fallback
    api_key = GOOGLE_API_KEY or GOOGLE_CLOUD_API_KEY
    if not api_key:
        st.error("API key tidak ditemukan. Pastikan GOOGLE_API_KEY atau GOOGLE_CLOUD_API_KEY sudah diset di file .env")
        st.stop()
    
    genai.configure(api_key=api_key)
    st.success("Google Generative AI success configured!")
    
except Exception as e:
    st.error(f"failed initialize Google GenAI. Error: {e}")
    st.stop()

def get_structured_query(user_prompt, data_schema):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro', 
                                      generation_config={"response_mime_type": "application/json"})
        
        system_prompt = f"""
        You are an expert data analyst for GAIKINDO automotive sales data. Convert the user's question into a structured JSON query.

        Generate a comprehensive, data-driven analysis in a professional tone that delivers 3–5 specific and statistically supported insights, each with clear business relevance, quantified evidence (percentages, ranges, monetary values), and actionable intelligence. Explanations must connect findings directly to strategic, financial, or operational implications, using precise statistical and business language. Include context for numerical results, outline the analytical methodology (approach, statistical methods, assumptions, validation), assess data quality, and indicate confidence levels or statistical significance. Present measurable and time-bound recommendations with explicit next steps. Identify major risks with mitigation strategies and highlight high-potential opportunities with quantified potential impact. Where applicable, suggest visualizations that reveal meaningful patterns or correlations that reinforce the conclusions. Ensure the response avoids generic descriptions, focuses exclusively on insights that drive decision-making value, and maintains technical rigor while remaining accessible to business stakeholders.
        
        Available data columns: {data_schema}
        
        Available sales types:
        - 'wholesales': Penjualan grosir/wholesale
        - 'retail_sales': Penjualan ritel/retail  
        - 'production_ckd': Produksi CKD (Completely Knocked Down)
        - 'import_cbu': Import CBU (Complete Built Up)
        
        Available brands: TOYOTA, HINO, ISUZU, MERCEDES_BENZ, UD_TRUCKS, FAW, TATA_MOTORS, SCANIA
        
        Query structure:
        - 'operation': 'average' (rata-rata), 'total_sales' (total angka), 'summary' (ringkasan per brand/segment), 'detailed_list' (daftar detail), 'comparison' (perbandingan antar brand/bulan)
        - 'sales_type': If not specified, default to 'retail_sales'
        - 'brands': List of uppercase brand names, or empty array for all brands
        - 'year': 2022 (data year), or null if not mentioned
        - 'month': Integer (1-12) for single month, or array of integers for multiple months [1,2,3], or null if not mentioned
        - 'segment': Vehicle segment if mentioned (e.g. "Truck", "Bus")

        Examples:
        1. "rata-rata penjualan Isuzu jan-mar" → {{"operation": "average", "sales_type": "retail_sales", "brands": ["ISUZU"], "year": 2022, "month": [1,2,3], "segment": null}}
        2. "total penjualan retail toyota april 2022" → {{"operation": "total_sales", "sales_type": "retail_sales", "brands": ["TOYOTA"], "year": 2022, "month": 4, "segment": null}}
        3. "Compare sales performance between HINO and Isuzu" → {{"operation": "comparison", "sales_type": "retail_sales", "brands": ["HINO", "ISUZU"], "year": 2022, "month": null, "segment": null}}
        4. "ringkasan wholesale semua brand 2022" → {{"operation": "summary", "sales_type": "wholesales", "brands": [], "year": 2022, "month": null, "segment": null}}
        """
        
        full_prompt = system_prompt + "\nUser Question: " + user_prompt
        
        response = model.generate_content(full_prompt)
        return json.loads(response.text)
        
    except Exception as e:
        st.error(f"Error dalam get_structured_query: {e}")
        # Return default query as fallback
        return {
            "operation": "summary", 
            "sales_type": "retail_sales", 
            "brands": [], 
            "year": 2022, 
            "month": None,
            "segment": None
        }



def execute_query(query, df):
    try:
        filtered_df = df.copy()
        
        sales_type_query = query.get("sales_type", "retail_sales")
        filtered_df = filtered_df[filtered_df['sales_type'] == sales_type_query]

        if query.get("brands"):
            brands_upper = [b.upper() for b in query["brands"]]
            filtered_df = filtered_df[filtered_df['brand'].isin(brands_upper)]
            
        if query.get("year"):
            filtered_df = filtered_df[filtered_df['year'] == query["year"]]
            
        # Handle month filtering - can be single month or list of months
        if query.get("month"):
            month_map_3_letter = { 1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec' }
            
            # Handle both single month and list of months
            month_query = query["month"]
            if isinstance(month_query, list):
                month_strings = [month_map_3_letter.get(m) for m in month_query if month_map_3_letter.get(m)]
                if month_strings:
                    filtered_df = filtered_df[filtered_df['month'].isin(month_strings)]
            else:
                month_str_val = month_map_3_letter.get(month_query)
                if month_str_val:
                    filtered_df = filtered_df[filtered_df['month'] == month_str_val]
        
        if query.get("segment"):
            segment_filter = query["segment"].strip()
            filtered_df = filtered_df[filtered_df['segment'].str.contains(segment_filter, case=False, na=False)]

        if filtered_df.empty:
            return "Data tidak ditemukan untuk kriteria yang Anda berikan. Pastikan nama brand, bulan, dan tipe sales sudah benar."

        operation = query['operation'].lower()
        
        # Update display columns based on actual column names in the data
        display_cols = { 'segment': 'Segment', 'model': 'Model', 'brand': 'Brand', 'sales': 'Sales', 'month': 'Bulan', 'sales_type': 'Tipe Penjualan' }
        
        if operation == 'total_sales':
            total = filtered_df['sales'].sum()
            return f"Total penjualan **{sales_type_query.replace('_', ' ')}** untuk kriteria yang dipilih adalah: **{int(total)}** unit."
            
        elif operation == 'average':
            # For user's request "rata-rata penjualan", calculate MONTHLY AVERAGES by sales type
            if query.get("brands") and len(query["brands"]) == 1:
                brand_name = query["brands"][0]
                
                # Get all sales types for the brand with the specified filters
                brand_df = df[df['brand'].isin(query["brands"])]
                if query.get("month"):
                    if isinstance(query["month"], list):
                        month_strings = [month_map_3_letter.get(m) for m in query["month"] if month_map_3_letter.get(m)]
                        if month_strings:
                            brand_df = brand_df[brand_df['month'].isin(month_strings)]
                            num_months = len(month_strings)
                    else:
                        month_str_val = month_map_3_letter.get(query["month"])
                        if month_str_val:
                            brand_df = brand_df[brand_df['month'] == month_str_val]
                            num_months = 1
                else:
                    num_months = len(brand_df['month'].unique())
                
                # Calculate averages per month by sales type  
                averages_by_type = brand_df.groupby('sales_type')['sales'].sum().round(0) / num_months
                
                # Format for simple response like: "Wholesales 3,014 unit, Retail Sales 2,375 unit"
                type_names = {
                    'wholesales': 'Wholesales',
                    'retail_sales': 'Retail Sales', 
                    'production_ckd': 'Production CKD',
                    'import_cbu': 'Import CBU'
                }
                
                result_parts = []
                for sales_type, avg_val in averages_by_type.items():
                    type_name = type_names.get(sales_type, sales_type.replace('_', ' ').title())
                    result_parts.append(f"{type_name} {int(avg_val):,} unit")
                
                return f"Rata-rata penjualan **{brand_name}** per bulan: {', '.join(result_parts)}"
            else:
                avg = filtered_df['sales'].mean()
                return f"Rata-rata penjualan **{sales_type_query.replace('_', ' ')}** adalah: **{int(avg)}** unit per record."
            
        elif operation == 'summary':
            summary_cols = ['brand', 'segment']
            existing_summary_cols = [col for col in summary_cols if col in filtered_df.columns]
            if existing_summary_cols:
                summary = filtered_df.groupby(existing_summary_cols)['sales'].sum().reset_index()
                summary.rename(columns=display_cols, inplace=True)
                return f"Berikut adalah ringkasan penjualan **{sales_type_query.replace('_', ' ')}**:\n\n{summary.to_markdown(index=False)}"
            else:
                return "Kolom yang diperlukan untuk summary tidak tersedia."

        elif operation == 'detailed_list':
             detail_cols_to_show = ['brand', 'segment', 'model', 'month', 'sales']
             detail_cols_to_show_existing = [col for col in detail_cols_to_show if col in filtered_df.columns]
             if detail_cols_to_show_existing:
                 # Limit results to avoid overwhelming output
                 sample_df = filtered_df[detail_cols_to_show_existing].head(50)
                 filtered_df_renamed = sample_df.rename(columns=display_cols)
                 total_records = len(filtered_df)
                 return f"Berikut adalah daftar penjualan terperinci **{sales_type_query.replace('_', ' ')}** (menampilkan 50 dari {total_records} total records):\n\n{filtered_df_renamed.to_markdown(index=False)}"
             else:
                 return "Kolom yang diperlukan untuk detailed list tidak tersedia."

        elif operation == 'comparison':
            if query.get("brands") and len(query["brands"]) >= 2:
                comparison_df = filtered_df.groupby('brand')['sales'].agg(['sum', 'count']).reset_index()
                comparison_df.columns = ['Brand', 'Total Sales', 'Jumlah Model']
                comparison_df = comparison_df.sort_values('Total Sales', ascending=False)
                return f"Perbandingan penjualan **{sales_type_query.replace('_', ' ')}** antar brand:\n\n{comparison_df.to_markdown(index=False)}"
            else:
                # Monthly comparison if no specific brands
                comparison_df = filtered_df.groupby('month')['sales'].sum().reset_index()
                month_names = {'jan': 'Januari', 'feb': 'Februari', 'mar': 'Maret', 'apr': 'April', 
                              'may': 'Mei', 'jun': 'Juni', 'jul': 'Juli', 'aug': 'Agustus',
                              'sep': 'September', 'oct': 'Oktober', 'nov': 'November', 'dec': 'Desember'}
                comparison_df['month'] = comparison_df['month'].map(month_names)
                comparison_df.columns = ['Bulan', 'Total Sales']
                comparison_df = comparison_df.sort_values('Total Sales', ascending=False)
                return f"Perbandingan penjualan **{sales_type_query.replace('_', ' ')}** per bulan:\n\n{comparison_df.to_markdown(index=False)}"

        else:
            return "Operasi tidak dikenal. Gunakan: summary, total_sales, detailed_list, atau comparison."

    except Exception as e:
        return f"Terjadi kesalahan saat memproses permintaan: {e}"

def generate_final_response(user_prompt, calculated_data):
    """
    Generate simple, direct response
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        system_prompt = """
Generate comprehensive, evidence-based responses that are firmly grounded in the provided dataset. Structure your response using the following mandatory framework:

1. Key Insights

Present 3-5 critical findings that directly address the user's query
Ensure each insight is precise, actionable, and substantiated by the data
Focus on the most significant patterns, trends, or conclusions relevant to the request
Use clear, concise language that captures the essence of your analysis

2. Detailed Analysis

Provide an in-depth examination of each key insight identified above
Include specific data points, statistics, or evidence from the dataset to support your explanations.
        # 
        """
        
        full_prompt = f"{system_prompt}\n\nPertanyaan: '{user_prompt}'\n\nData: {calculated_data}\n\nJawaban ringkas:"
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error dalam generate_final_response: {e}")
        return calculated_data

# --- 3. UI UTAMA STREAMLIT ---
def main():
    st.set_page_config(page_title="GAIKINDO Sales Analysis Platform", layout="wide")
    
    # Sidebar for navigation
    st.sidebar.title("GAIKINDO Analytics")
    page = st.sidebar.selectbox("Choose Analysis Mode", 
                               ["Smart Q&A", "📊 Visual Analytics", "🔍 Business Insights", "📈 Predictions"])

    if not os.path.exists(LOCAL_PARQUET_FILE):
        st.error(f"File data '{LOCAL_PARQUET_FILE}' tidak ditemukan.")
        st.info("Jalankan proses ETL terlebih dahulu: python src/data_processor.py")
        st.stop()

    master_df = pd.read_parquet(LOCAL_PARQUET_FILE)
    api_key = GOOGLE_API_KEY or GOOGLE_CLOUD_API_KEY
    
    # Initialize analytics modules
    visualizer = GaikindoVisualizer(master_df)
    insights_generator = GaikindoInsightsGenerator(master_df, api_key)
    
    if page == "Smart Q&A":
        st.title("Smart Q&A")
        st.success(f"Data loaded: {len(master_df)} records")

        with st.expander("Data Preview"):
            st.dataframe(master_df.head())

        # Add information about available data
        st.info("""
        **Available Data:**
        - **Sales Types**: Wholesales, Retail Sales, Production CKD, Import CBU
        - **Brands**: Toyota, Hino, Isuzu, Mercedes-Benz, UD Trucks, FAW, Tata Motors, Scania  
        - **Period**: Jan-Apr 2022 | **Records**: 68 clean aggregated totals
        """)
        
        # Example queries
        with st.expander("💡 Example Questions"):
            st.write("""
            **Simple Queries:**
            - "Berapa rata-rata penjualan Isuzu jan-mar" 
            - "Total penjualan Toyota April"
            - "Compare sales performance between HINO and Isuzu"
            - "Ringkasan penjualan wholesale semua brand"
            """)
        
        prompt = st.text_input("Ask about GAIKINDO sales data:", 
                              placeholder="Example: What's the average retail sales for Toyota?")

        if st.button("💫 Ask AI"):
            if prompt:
                with st.spinner("Analyzing your request..."):
                    schema = ", ".join(master_df.columns)
                    structured_query = get_structured_query(prompt, schema)
                    with st.expander("🔍 Generated Query"):
                        st.json(structured_query)

                with st.spinner("Processing data..."):
                    calculation_result = execute_query(structured_query, master_df)
                    with st.expander("📊 Calculation Results"):
                        st.markdown(calculation_result)
                
                with st.spinner("Generating final answer..."):
                    final_answer = generate_final_response(prompt, calculation_result)
                    st.write("### 🤖 AI Answer:")
                    st.markdown(final_answer)
            else:
                st.warning("Please enter a question.")
    
    elif page == "📊 Visual Analytics":
        st.title("Visual Analytics Dashboard 📊")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_brands = st.multiselect("Select Brands", master_df['brand'].unique(), 
                                           default=["TOYOTA", "ISUZU", "HINO"])
        with col2:
            selected_sales_type = st.selectbox("Sales Type", master_df['sales_type'].unique())
        with col3:
            show_all_brands = st.checkbox("Show All Brands", value=False)
        
        if show_all_brands:
            selected_brands = master_df['brand'].unique().tolist()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(visualizer.sales_trend_chart(selected_brands, selected_sales_type), 
                          use_container_width=True)
            st.plotly_chart(visualizer.brand_comparison_chart(selected_sales_type), 
                          use_container_width=True)
        
        with col2:
            st.plotly_chart(visualizer.sales_type_distribution(selected_brands), 
                          use_container_width=True)
            st.plotly_chart(visualizer.monthly_heatmap(selected_brands), 
                          use_container_width=True)
        
        # Market share analysis
        st.subheader("📈 Market Analysis")
        market_fig, market_data = visualizer.market_share_analysis(selected_sales_type)
        st.plotly_chart(market_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Market Share Data:**")
            st.dataframe(market_data)
        
        with col2:
            growth_fig, growth_data = visualizer.growth_analysis()
            st.plotly_chart(growth_fig, use_container_width=True)
    
    elif page == "🔍 Business Insights":
        st.title("Business Insights & Analysis 🔍")
        
        with st.spinner("Generating comprehensive business insights..."):
            executive_summary = insights_generator.generate_executive_summary()
        
        # Executive Summary
        st.subheader("📋 Executive Summary")
        summary_data = executive_summary['summary_data']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Market Size", f"{summary_data['total_market_size']:,}", "units")
        with col2:
            st.metric("Active Brands", summary_data['number_of_brands'])
        with col3:
            st.metric("Peak Season", summary_data['peak_season'].title())
        with col4:
            st.metric("Analysis Period", "Jan-Apr 2022")
        
        # AI-Generated Insights
        st.subheader("🤖 AI-Powered Market Insights")
        st.markdown(executive_summary['ai_insights'])
        
        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏢 Brand Performance", "⚔️ Competition", "📅 Seasonal Trends", "⚠️ Risk & Opportunities"])
        
        with tab1:
            st.write("**Brand Performance Analysis:**")
            for brand, data in executive_summary['brand_analysis'].items():
                with st.expander(f"{brand} - Market Share: {data['market_share']}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sales", f"{data['total_sales']:,}")
                        st.metric("Avg Monthly Sales", f"{data['avg_monthly_sales']:,}")
                        st.write(f"**Dominant Sales Type:** {data['dominant_sales_type']}")
                    with col2:
                        st.write("**Sales by Type:**")
                        for sales_type, value in data['sales_by_type'].items():
                            st.write(f"- {sales_type}: {value:,}")
        
        with tab2:
            st.write("**Competitive Landscape:**")
            comp_data = executive_summary['competitive_landscape']
            st.write("**Overall Market Leaders:**")
            for i, (brand, sales) in enumerate(comp_data['market_leaders'].items(), 1):
                st.write(f"{i}. **{brand}**: {sales:,} units")
        
        with tab3:
            seasonal_data = executive_summary['seasonal_trends']
            st.write("**Seasonal Performance Index (100 = Average):**")
            for month, index in seasonal_data['seasonal_indices'].items():
                color = "🟢" if index > 100 else "🔴"
                st.write(f"{color} **{month.title()}**: {index}%")
            
            st.info(f"📈 **Peak Season**: {seasonal_data['peak_month'].title()} | 📉 **Low Season**: {seasonal_data['low_month'].title()}")
        
        with tab4:
            risk_data = executive_summary['risk_opportunities']
            
            if risk_data['risks']:
                st.write("⚠️ **Identified Risks:**")
                for risk in risk_data['risks']:
                    st.warning(risk)
            
            if risk_data['opportunities']:
                st.write("🌟 **Growth Opportunities:**")
                for opportunity in risk_data['opportunities']:
                    st.success(opportunity)
            
            st.info(f"**Overall Risk Level**: {risk_data['overall_risk_level'].title()}")
    
    elif page == "📈 Predictions":
        st.title("Sales Predictions & Forecasting 📈")
        
        # Prediction controls
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_brand = st.selectbox("Select Brand for Prediction", 
                                    ["All Brands"] + list(master_df['brand'].unique()))
        with col2:
            pred_sales_type = st.selectbox("Sales Type for Prediction", 
                                         master_df['sales_type'].unique())
        with col3:
            pred_periods = st.slider("Forecast Periods", 1, 6, 3)
        
        if st.button("🔮 Generate Predictions"):
            selected_brand = None if pred_brand == "All Brands" else pred_brand
            
            with st.spinner("Generating predictions..."):
                predictions = insights_generator.predict_future_sales(
                    brand=selected_brand, 
                    sales_type=pred_sales_type, 
                    periods=pred_periods
                )
            
            st.subheader(f"📊 Sales Forecast - {pred_sales_type.replace('_', ' ').title()}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Predicted Sales:**")
                for month, value in predictions['predictions'].items():
                    st.metric(month.title(), f"{value:,} units")
            
            with col2:
                trend_emoji = "📈" if predictions['trend'] == "increasing" else "📉"
                st.metric("Trend Direction", f"{trend_emoji} {predictions['trend'].title()}")
                st.metric("Trend Strength", f"{predictions['trend_strength']:.0f} units/month")
                st.metric("Confidence Level", predictions['confidence'].title())
            
            # Create prediction visualization
            current_months = ['jan', 'feb', 'mar', 'apr']
            future_months = list(predictions['predictions'].keys())
            
            # Get current data for visualization
            df_filtered = master_df[master_df['sales_type'] == pred_sales_type].copy()
            if selected_brand:
                df_filtered = df_filtered[df_filtered['brand'] == selected_brand]
            
            current_data = df_filtered.groupby('month')['sales'].sum().reindex(current_months, fill_value=0)
            
            # Combine current and predicted data
            all_months = current_months + future_months
            all_values = list(current_data.values) + list(predictions['predictions'].values())
            colors = ['blue'] * len(current_months) + ['orange'] * len(future_months)
            
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=current_months, y=current_data.values,
                mode='lines+markers', name='Historical',
                line=dict(color='blue', width=3)
            ))
            
            # Predicted data
            fig.add_trace(go.Scatter(
                x=future_months, y=list(predictions['predictions'].values()),
                mode='lines+markers', name='Predicted',
                line=dict(color='orange', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Sales Forecast: {pred_brand if pred_brand != 'All Brands' else 'All Brands'} - {pred_sales_type.replace('_', ' ').title()}",
                xaxis_title="Month",
                yaxis_title="Sales Units",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction insights
            st.info(f"""
            **Prediction Summary:**
            - Trend: {predictions['trend'].title()} trend detected
            - Monthly change: {predictions['trend_strength']:.0f} units per month
            - Confidence: {predictions['confidence'].title()} (based on historical data consistency)
            - Note: Predictions are based on linear trend analysis of Jan-Apr 2022 data
            """)

if __name__ == "__main__":
    main()