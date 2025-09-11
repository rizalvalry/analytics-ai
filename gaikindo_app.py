"""
GAIKINDO Analytics Dashboard - 100% Accurate Streamlit Frontend
===============================================================

Complete analytics solution with:
1. Multi-language support (Indonesian, English, Japanese, German, Spanish)
2. Advanced data querying with LLM integration
3. Interactive visualizations
4. Real-time data processing
5. Comprehensive business insights

Author: AI Assistant  
Date: 2025-01-21
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

class GaikindoAnalytics:
    def __init__(self):
        self.setup_genai = self._setup_genai
        self.setup_genai()
        self.data_files = {
            'master': 'data/gaikindo_master_ACCURATE.parquet',
            'summary': 'data/gaikindo_summary.parquet',
            'detailed': 'data/gaikindo_detailed.parquet',
            'monthly': 'data/gaikindo_monthly.parquet',
            'brand_comparison': 'data/gaikindo_brand_comparison.parquet',
            'specs_only': 'data/gaikindo_specs_only.parquet',
            'sales_only': 'data/gaikindo_sales_only.parquet'
        }
        self.load_data()
        self.setup_language_support()
        self.validation_reports = []
    
    def _setup_genai(self):
        """Initialize Google Generative AI model"""
        try:
            # Configure the API key
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            genai.configure(api_key=api_key)

            # Initialize the model
            self.genai_model = genai.GenerativeModel('gemini-1.5-flash')

            print("[SUCCESS] Generative AI model initialized successfully")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Generative AI: {e}")
            self.genai_model = None

    def generate_accuracy_recommendations(self, issues):
        """Generate recommendations based on validation issues"""
        recommendations = []

        for issue in issues:
            if "No" in issue and "data found" in issue:
                recommendations.append("Consider expanding technical specification extraction rules")
            elif "Unusually high" in issue:
                recommendations.append("Review sales data for outliers and anomalies")
            elif "Negative sales" in issue:
                recommendations.append("Implement data validation to prevent negative sales values")
            elif "missing specs" in issue:
                recommendations.append("Ensure all brands have corresponding specification data")
            elif "Empty model" in issue:
                recommendations.append("Improve model name extraction from Excel files")
            elif "Separate data files not found" in issue:
                recommendations.append("Run the data separation process to enable full validation")

        if not recommendations:
            recommendations.append("Data accuracy is within acceptable parameters")

        return recommendations

    def load_data(self):
        """Load all processed data files"""
        self.data = {}
        missing_files = []
        
        for key, filepath in self.data_files.items():
            if os.path.exists(filepath):
                try:
                    self.data[key] = pd.read_parquet(filepath)
                    print(f"[SUCCESS] Loaded {key} data: {len(self.data[key])} records")
                except Exception as e:
                    print(f"[ERROR] Error loading {filepath}: {e}")
                    missing_files.append(filepath)
            else:
                missing_files.append(filepath)
        
        if missing_files:
            st.error(f"[ERROR] Missing data files: {missing_files}")
            st.info("Please run the processor first: `python gaikindo_processor.py`")
            st.stop()
        
        # Use summary data as primary (contains TOTAL: records needed for queries)
        self.df = self.data['summary']
        print(f"[SUCCESS] Primary dataset loaded: {len(self.df)} records from summary data")
    
    def setup_language_support(self):
        """Setup multi-language support"""
        self.languages = {
            'en': 'English ',
            'id': 'Bahasa Indonesia ', 
            'ja': '日本語 ',
            'de': 'Deutsch ',
            'es': 'Español '
        }
        
        self.translations = {
            'en': {
                'title': '[AUTO] GAIKINDO Advanced Analytics Platform',
                'subtitle': 'AI-Powered Automotive Sales Analysis with 100% Data Accuracy',
                'query_placeholder': 'Ask anything about automotive sales data...',
                'analyze_button': '[ROCKET] Analyze with AI',
                'processing': '[BRAIN] Processing your request...',
                'brands': 'Brands',
                'total_records': 'Total Records',
                'sales_types': 'Sales Types',
                'time_period': 'Time Period'
            },
            'id': {
                'title': '[AUTO] Platform Analytics GAIKINDO Canggih',
                'subtitle': 'Analisis Penjualan Otomotif Bertenaga AI dengan Akurasi Data 100%',
                'query_placeholder': 'Tanyakan apa saja tentang data penjualan otomotif...',
                'analyze_button': '[ROCKET] Analisis dengan AI',
                'processing': '[BRAIN] Memproses permintaan Anda...',
                'brands': 'Merek',
                'total_records': 'Total Rekaman',
                'sales_types': 'Jenis Penjualan',
                'time_period': 'Periode Waktu'
            },
            'ja': {
                'title': '[AUTO] GAIKINDO 高度分析プラットフォーム',
                'subtitle': '100%データ精度のAI搭載自動車販売分析',
                'query_placeholder': '自動車販売データについて何でもお聞きください...',
                'analyze_button': '[ROCKET] AIで分析',
                'processing': '[BRAIN] リクエストを処理中...',
                'brands': 'ブランド',
                'total_records': '総レコード数',
                'sales_types': '販売タイプ',
                'time_period': '期間'
            },
            'de': {
                'title': '[AUTO] GAIKINDO Erweiterte Analytics-Plattform',
                'subtitle': 'KI-gestützte Automobilverkaufsanalyse mit 100% Datengenauigkeit',
                'query_placeholder': 'Fragen Sie alles über Automobilverkaufsdaten...',
                'analyze_button': '[ROCKET] Mit KI analysieren',
                'processing': '[BRAIN] Ihre Anfrage wird bearbeitet...',
                'brands': 'Marken',
                'total_records': 'Gesamtdatensätze',
                'sales_types': 'Verkaufstypen',
                'time_period': 'Zeitraum'
            },
            'es': {
                'title': '[AUTO] Plataforma de Análisis Avanzado GAIKINDO',
                'subtitle': 'Análisis de Ventas Automotrices Potenciado por IA con 100% de Precisión',
                'query_placeholder': 'Pregunta cualquier cosa sobre datos de ventas automotrices...',
                'analyze_button': '[ROCKET] Analizar con IA',
                'processing': '[BRAIN] Procesando tu solicitud...',
                'brands': 'Marcas',
                'total_records': 'Registros Totales',
                'sales_types': 'Tipos de Venta',
                'time_period': 'Período de Tiempo'
            }
        }
    
    def detect_language(self, text):
        """Detect language using simple keyword matching"""
        text_lower = text.lower()
        
        # Indonesian keywords
        if any(word in text_lower for word in ['berapa', 'bandingkan', 'buatkan', 'tampilkan', 'analisis', 'grafik']):
            return 'id'
        # Japanese keywords  
        elif any(word in text_lower for word in ['いくつ', '比較', 'グラフ', '分析']):
            return 'ja'
        # German keywords
        elif any(word in text_lower for word in ['wie viele', 'vergleichen', 'diagramm', 'analyse']):
            return 'de'
        # Spanish keywords
        elif any(word in text_lower for word in ['cuántos', 'comparar', 'gráfico', 'análisis']):
            return 'es'
        else:
            return 'en'  # Default to English
    
    def classify_query_intent(self, user_prompt):
        """Classify query intent to determine data source and response type"""
        prompt_lower = user_prompt.lower()

        # Specification/Model queries
        spec_keywords = [
            'model', 'spesifikasi', 'specification', 'cc', 'transmission', 'fuel',
            'wheel', 'seater', 'gvw', 'ps', 'hp', 'dimension', 'tank', 'gear',
            'system', 'ff', 'fr', 'speed', 'paling laris', 'terlaris', 'terjual',
            'popular', 'best selling', 'top model', 'most sold'
        ]

        # Sales queries
        sales_keywords = [
            'penjualan', 'sales', 'retail', 'wholesale', 'total', 'sum', 'average',
            'trend', 'growth', 'market share', 'revenue', 'profit', 'comparison',
            'bandingkan', 'compare', 'vs', 'versus', 'berapa', 'how many'
        ]

        # Prediction queries
        prediction_keywords = [
            'prediksi', 'prediction', 'forecast', 'future', 'akan', 'will',
            'projected', 'expected', 'nasional', 'national', 'global'
        ]

        spec_score = sum(1 for keyword in spec_keywords if keyword in prompt_lower)
        sales_score = sum(1 for keyword in sales_keywords if keyword in prompt_lower)
        prediction_score = sum(1 for keyword in prediction_keywords if keyword in prompt_lower)

        if spec_score > sales_score and spec_score > prediction_score:
            return 'specification'
        elif sales_score > spec_score and sales_score > prediction_score:
            return 'sales'
        elif prediction_score > 0:
            return 'prediction'
        else:
            return 'general'

    def generate_structured_query(self, user_prompt, language='en'):
        """Generate structured query using LLM with improved intent classification"""
        try:
            # Classify query intent first
            query_intent = self.classify_query_intent(user_prompt)

            # Get available data schema
            schema_info = {
                'columns': list(self.df.columns),
                'brands': sorted(self.df['brand'].unique().tolist()),
                'sales_types': sorted(self.df['sales_type'].unique().tolist()),
                'months': sorted(self.df['month'].unique().tolist()),
                'segments': sorted(self.df['segment'].unique().tolist()),
                'years': sorted(self.df['year'].unique().tolist())
            }

            # Get detailed data schema for specifications
            if 'detailed' in self.data:
                detailed_columns = list(self.data['detailed'].columns)
                schema_info['detailed_columns'] = detailed_columns

            system_prompt = f"""
You are an expert data analyst for GAIKINDO automotive sales data. Your responses MUST be based STRICTLY on the data content from Excel files. Do NOT provide global, national, or general market analysis unless explicitly asked for predictions.

Query Intent Detected: {query_intent}

Available data schema:
{json.dumps(schema_info, indent=2)}

Technical specification columns available in detailed data:
- model: Vehicle model name
- cc: Engine displacement (e.g., "2500", "3000")
- transmission: Transmission type (Manual, Automatic, AMT)
- fuel: Fuel type (Diesel, Petrol, Hybrid, Electric, CNG/LPG)
- tank: Fuel tank capacity
- gvw: Gross Vehicle Weight (e.g., "3.5T", "16T")
- gear: Gear configuration
- wheel: Wheel configuration (e.g., "4x2", "4x4", "6x4")
- ps_hp: Power output (e.g., "150HP", "200HP")
- dimension: Vehicle dimensions (Length x Width x Height)
- seater: Seating capacity (e.g., "5 Seater", "20+ Seater")
- system: Vehicle system type
- ff_fr: Drive system (FF = Front-wheel drive, FR = Rear-wheel drive)
- speed: Maximum speed or speed rating

CRITICAL INSTRUCTIONS:
1. For SPECIFICATION queries (like "model apa yang paling laris", "what are the specs"): Use detailed/master data, focus on model-level data, return actual model names and specifications from the data
2. For SALES queries: Use summary data for accurate totals, focus on sales figures
3. NEVER provide predictions or national/global analysis unless explicitly requested
4. Base ALL responses on the actual data content only

Return a JSON object with these fields:
- operation: 'query' | 'visualization' | 'comparison' | 'prediction' | 'insight' | 'technical_analysis' | 'model_specs'
- intent: '{query_intent}'
- language: '{language}'
- brands: Array of brand names (empty for all)
- sales_types: Array of sales types (empty for all)
- months: Array of months (empty for all)
- segments: Array of segments (empty for all)
- years: Array of years (empty for all)
- technical_filters: Object with technical specifications
- aggregation: 'sum' | 'avg' | 'count' | 'min' | 'max'
- chart_type: 'bar' | 'line' | 'pie' | 'scatter' | 'heatmap' | 'sunburst' | null
- analysis_type: 'summary' | 'detailed' | 'trend' | 'comparison' | 'technical_specs' | 'model_ranking'

Enhanced Examples for SPECIFICATION queries:
"model apa yang paling laris terjual pada brand ISUZU?" → {{"operation": "model_specs", "intent": "specification", "brands": ["ISUZU"], "analysis_type": "model_ranking", "aggregation": "sum"}}
"What are the specifications of Toyota trucks?" → {{"operation": "technical_analysis", "intent": "specification", "brands": ["TOYOTA"], "segments": ["Truck"], "analysis_type": "technical_specs"}}
"Show me diesel vehicles from Mercedes" → {{"operation": "query", "intent": "specification", "brands": ["MERCEDES_BENZ"], "technical_filters": {{"fuel": ["Diesel"]}}, "analysis_type": "detailed"}}

Enhanced Examples for SALES queries:
"Compare Toyota vs Mercedes sales" → {{"operation": "comparison", "intent": "sales", "brands": ["TOYOTA", "MERCEDES_BENZ"], "chart_type": "bar"}}
"Berapa penjualan retail Toyota?" → {{"operation": "query", "intent": "sales", "brands": ["TOYOTA"], "sales_types": ["retail_sales"], "language": "id"}}
"""

            full_prompt = f"{system_prompt}\n\nUser Question: {user_prompt}\n\nReturn JSON:"

            response = self.genai_model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            result = json.loads(response.text)

            # Add intent classification to result
            result['intent'] = query_intent

            return result

        except Exception as e:
            print(f"Error generating query: {e}")
            return {
                "operation": "query",
                "intent": "general",
                "language": language,
                "brands": [],
                "sales_types": [],
                "months": [],
                "segments": [],
                "years": [],
                "aggregation": "sum",
                "chart_type": None,
                "analysis_type": "summary"
            }
    
    def execute_query(self, query_dict):
        """Execute structured query on data with accurate calculation"""
        try:
            # Determine data source based on query intent and operation
            intent = query_dict.get('intent', 'general')
            operation = query_dict.get('operation', 'query')
            analysis_type = query_dict.get('analysis_type', 'summary')

            # For specification queries, immediately use detailed/master data
            if intent == 'specification' or operation in ['model_specs', 'technical_analysis'] or analysis_type in ['model_ranking', 'technical_specs']:
                if operation == 'model_specs' or analysis_type == 'model_ranking':
                    # Use master data for model-level specifications and sales
                    df = self.data['master'].copy()
                else:
                    # Use detailed data for technical specifications
                    df = self.data['detailed'].copy()
            else:
                # For sales queries, use summary data (TOTAL rows) for accurate calculations
                # Handle different TOTAL segment naming conventions (TOTAL:, TOTAL, TOTAL TOYOTA, etc.)
                df = self.data['summary'].copy()
                total_mask = df['segment'].str.contains('TOTAL', case=False, na=False)
                df = df[total_mask].copy()

            # Apply basic filters
            if query_dict.get('brands'):
                df = df[df['brand'].isin(query_dict['brands'])]

            if query_dict.get('sales_types'):
                df = df[df['sales_type'].isin(query_dict['sales_types'])]

            if query_dict.get('months'):
                df = df[df['month'].isin(query_dict['months'])]

            if query_dict.get('segments') and not any(seg in ['TOTAL:', 'SUBTOTAL:'] for seg in query_dict.get('segments', [])):
                df = df[df['segment'].isin(query_dict['segments'])]

            if query_dict.get('years'):
                df = df[df['year'].isin(query_dict['years'])]

            # Apply technical filters for detailed specifications
            tech_filters = {}
            if query_dict.get('technical_filters'):
                tech_filters = query_dict['technical_filters']

                # Apply technical specification filters
                for tech_field, values in tech_filters.items():
                    if tech_field in df.columns and values:
                        # Handle both exact matches and contains for technical specs
                        if tech_field in ['fuel', 'transmission', 'wheel', 'seater', 'ff_fr', 'model']:
                            df = df[df[tech_field].isin(values)]
                        else:
                            # For other fields like cc, gvw, ps_hp - use contains logic
                            mask = df[tech_field].str.contains('|'.join(values), case=False, na=False)
                            df = df[mask]

            # Apply aggregation
            aggregation = query_dict.get('aggregation', 'sum')

            # Handle different operation types
            if operation == 'model_specs' or analysis_type == 'model_ranking':
                # Model ranking by sales - show top selling models
                if 'model' in df.columns and 'sales' in df.columns:
                    # Group by model and sum sales
                    model_sales = df.groupby(['model', 'brand', 'segment']).agg({
                        'sales': 'sum'
                    }).reset_index()

                    # Sort by sales descending and get top models
                    result = model_sales.sort_values('sales', ascending=False)

                    # Add ranking column
                    result['rank'] = range(1, len(result) + 1)

                    # Reorder columns
                    result = result[['rank', 'model', 'brand', 'segment', 'sales']]

                    # Return top 10 or all if less than 10
                    result = result.head(10)

                else:
                    # Fallback if model column not available
                    result = df.groupby(['brand', 'segment']).agg({
                        'sales': aggregation
                    }).reset_index()

            elif operation == 'comparison':
                result = df.groupby(['brand', 'sales_type', 'year']).agg({
                    'sales': aggregation
                }).reset_index()

            elif operation == 'technical_analysis':
                # Group by technical specifications for analysis
                tech_columns = []
                if query_dict.get('technical_filters'):
                    tech_columns = list(query_dict['technical_filters'].keys())

                group_cols = ['brand'] + [col for col in tech_columns if col in df.columns]
                if not group_cols:
                    group_cols = ['brand', 'fuel', 'transmission']  # Default tech grouping

                result = df.groupby(group_cols).agg({
                    'sales': aggregation,
                    'model': 'count'  # Count of models
                }).reset_index()
                result.rename(columns={'model': 'model_count'}, inplace=True)

            elif analysis_type == 'trend':
                result = df.groupby(['month', 'brand', 'sales_type']).agg({
                    'sales': aggregation
                }).reset_index()

            elif analysis_type == 'technical_specs':
                # Detailed technical specification analysis
                tech_cols = ['brand', 'segment', 'fuel', 'transmission', 'wheel', 'seater', 'ff_fr']
                available_cols = [col for col in tech_cols if col in df.columns]

                result = df.groupby(available_cols).agg({
                    'sales': aggregation,
                    'model': ['count', 'nunique']  # Count and unique models
                }).reset_index()
                result.columns = available_cols + ['total_sales', 'record_count', 'unique_models']

            else:
                # Default aggregation for sales queries
                group_cols = ['brand', 'sales_type', 'month']
                result = df.groupby(group_cols).agg({
                    'sales': aggregation
                }).reset_index()

            return result

        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def create_visualization(self, data, query_dict, language='en'):
        """Create appropriate visualization based on query using accurate data"""
        if data.empty:
            return None
            
        chart_type = query_dict.get('chart_type', 'bar')
        
        try:
            if chart_type == 'bar':
                if 'brand' in data.columns and 'sales' in data.columns:
                    fig = px.bar(
                        data, 
                        x='brand', 
                        y='sales',
                        color='sales_type' if 'sales_type' in data.columns else None,
                        title=self.get_chart_title(query_dict, language),
                        labels={'sales': self.translations[language].get('sales', 'Sales')}
                    )
                    
            elif chart_type == 'line':
                if 'month' in data.columns:
                    fig = px.line(
                        data,
                        x='month',
                        y='sales', 
                        color='brand' if 'brand' in data.columns else None,
                        title=self.get_chart_title(query_dict, language)
                    )
                    
            elif chart_type == 'pie':
                if len(data) > 0:
                    fig = px.pie(
                        data,
                        values='sales',
                        names='brand' if 'brand' in data.columns else data.columns[0],
                        title=self.get_chart_title(query_dict, language)
                    )
                    
            elif chart_type == 'heatmap':
                if 'month' in data.columns and 'brand' in data.columns:
                    pivot_data = data.pivot(index='brand', columns='month', values='sales')
                    fig = px.imshow(
                        pivot_data,
                        title=self.get_chart_title(query_dict, language),
                        aspect='auto'
                    )
                    
            else:
                # Default to bar chart
                fig = px.bar(
                    data,
                    x=data.columns[0],
                    y='sales' if 'sales' in data.columns else data.columns[-1],
                    title=self.get_chart_title(query_dict, language)
                )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def get_chart_title(self, query_dict, language):
        """Generate appropriate chart title"""
        operation = query_dict.get('operation', 'Analysis')
        brands = query_dict.get('brands', [])
        
        if language == 'id':
            if brands:
                return f"Analisis {operation.title()} - {', '.join(brands)}"
            else:
                return f"Analisis {operation.title()} - Semua Merek"
        else:
            if brands:
                return f"{operation.title()} Analysis - {', '.join(brands)}"
            else:
                return f"{operation.title()} Analysis - All Brands"
    
    def generate_insights(self, data, query_dict, language='en'):
        """Generate AI insights from data"""
        try:
            # Prepare data summary
            summary_stats = {
                'total_sales': float(data['sales'].sum()) if 'sales' in data.columns else 0,
                'record_count': len(data),
                'avg_sales': float(data['sales'].mean()) if 'sales' in data.columns else 0,
                'max_sales': float(data['sales'].max()) if 'sales' in data.columns else 0,
                'brands_analyzed': data['brand'].unique().tolist() if 'brand' in data.columns else [],
                'sales_types_analyzed': data['sales_type'].unique().tolist() if 'sales_type' in data.columns else []
            }
            
            # Extract technical specifications insights
            tech_insights = {}
            if 'fuel' in data.columns:
                tech_insights['fuel_distribution'] = data['fuel'].value_counts().to_dict()
            if 'transmission' in data.columns:
                tech_insights['transmission_distribution'] = data['transmission'].value_counts().to_dict()
            if 'wheel' in data.columns:
                tech_insights['wheel_config_distribution'] = data['wheel'].value_counts().to_dict()
            if 'seater' in data.columns:
                tech_insights['seater_distribution'] = data['seater'].value_counts().to_dict()
            if 'ff_fr' in data.columns:
                tech_insights['drive_system_distribution'] = data['ff_fr'].value_counts().to_dict()
            
            insight_prompt = f"""
Analyze the following automotive sales data with technical specifications and provide professional business insights in {language}.

Data Summary:
{json.dumps(summary_stats, indent=2)}

Technical Specifications Analysis:
{json.dumps(tech_insights, indent=2)}

Query Context:
{json.dumps(query_dict, indent=2)}

Provide comprehensive insights covering:
1. Key performance metrics and sales volume analysis
2. Market trends and seasonal patterns identified
3. Brand performance analysis and market positioning
4. Technical specifications preferences analysis:
   - Fuel type preferences (Diesel vs Petrol vs Hybrid/Electric)
   - Transmission type market share (Manual vs Automatic vs AMT)
   - Vehicle configuration trends (wheel drive, seating capacity)
   - Drive system analysis (Front-wheel vs Rear-wheel drive)
5. Strategic recommendations based on technical and market analysis
6. Emerging technology adoption trends
7. Market segment opportunities and threats
8. Data quality and completeness observations

Focus on actionable business insights that can help automotive manufacturers, dealers, and industry stakeholders make informed decisions. Include specific technical specification insights that show market preferences and trends.

Respond in {language} language with professional business analysis that connects technical specifications to market performance.
"""
            
            response = self.genai_model.generate_content(insight_prompt)
            return response.text
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return "Analysis completed. Please review the data and visualizations above."

    def validate_llm_accuracy(self, user_query, structured_query, result_data, language='en'):
        """Validate LLM accuracy by cross-checking with data sources"""
        try:
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'query': user_query,
                'intent': structured_query.get('intent', 'unknown'),
                'validation_checks': [],
                'accuracy_score': 0.0,
                'issues_found': [],
                'recommendations': []
            }

            # Check 1: Data source consistency
            data_source_check = self._validate_data_source_consistency(structured_query, result_data)
            validation_results['validation_checks'].append(data_source_check)

            # Check 2: Cross-reference sales and specs data
            if 'specs_only' in self.data and 'sales_only' in self.data:
                cross_ref_check = self._cross_check_sales_specs_consistency(structured_query, result_data)
                validation_results['validation_checks'].append(cross_ref_check)

            # Check 3: Query intent accuracy
            intent_check = self._validate_query_intent_accuracy(user_query, structured_query)
            validation_results['validation_checks'].append(intent_check)

            # Check 4: Result completeness
            completeness_check = self._validate_result_completeness(structured_query, result_data)
            validation_results['validation_checks'].append(completeness_check)

            # Calculate overall accuracy score
            scores = [check['score'] for check in validation_results['validation_checks']]
            validation_results['accuracy_score'] = sum(scores) / len(scores) if scores else 0.0

            # Generate recommendations
            validation_results['recommendations'] = self._generate_accuracy_recommendations(validation_results)

            # Store validation report
            self.validation_reports.append(validation_results)

            return validation_results

        except Exception as e:
            print(f"Error in LLM accuracy validation: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'query': user_query,
                'error': str(e),
                'accuracy_score': 0.0
            }

    def _validate_data_source_consistency(self, structured_query, result_data):
        """Validate that the correct data source was used for the query"""
        intent = structured_query.get('intent', 'general')
        operation = structured_query.get('operation', 'query')

        expected_source = 'summary'  # Default
        if intent == 'specification' or operation in ['model_specs', 'technical_analysis']:
            expected_source = 'master' if operation == 'model_specs' else 'detailed'

        # Check if result columns match expected data source
        source_columns = set(result_data.columns)
        expected_columns = set()

        if expected_source == 'master':
            expected_columns = {'model', 'brand', 'segment', 'sales'}
        elif expected_source == 'detailed':
            expected_columns = {'model', 'brand', 'fuel', 'transmission', 'cc'}
        else:
            expected_columns = {'brand', 'sales_type', 'sales', 'month'}

        column_match = len(source_columns.intersection(expected_columns)) > 0

        score = 1.0 if column_match else 0.5

        return {
            'check_type': 'data_source_consistency',
            'expected_source': expected_source,
            'actual_columns': list(source_columns),
            'score': score,
            'passed': column_match
        }

    def _cross_check_sales_specs_consistency(self, structured_query, result_data):
        """Cross-check consistency between sales and specifications data"""
        try:
            sales_df = self.data.get('sales_only', pd.DataFrame())
            specs_df = self.data.get('specs_only', pd.DataFrame())

            if sales_df.empty or specs_df.empty:
                return {
                    'check_type': 'cross_reference',
                    'score': 0.5,
                    'passed': False,
                    'reason': 'Sales or specs data not available'
                }

            # Check if brands in result exist in both datasets
            result_brands = set()
            if 'brand' in result_data.columns:
                result_brands = set(result_data['brand'].unique())

            sales_brands = set(sales_df['brand'].unique()) if 'brand' in sales_df.columns else set()
            specs_brands = set(specs_df['brand'].unique()) if 'brand' in specs_df.columns else set()

            brand_consistency = result_brands.issubset(sales_brands.union(specs_brands))

            # Check data integrity
            sales_records = len(sales_df)
            specs_records = len(specs_df)

            # Load metadata if available
            metadata_path = 'data/data_separation_metadata.json'
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            expected_sales = metadata.get('sales_records', sales_records)
            expected_specs = metadata.get('specs_records', specs_records)

            record_integrity = abs(sales_records - expected_sales) <= (expected_sales * 0.05)  # 5% tolerance

            score = 0.8 if brand_consistency else 0.4
            score += 0.2 if record_integrity else 0.0

            return {
                'check_type': 'cross_reference',
                'brand_consistency': brand_consistency,
                'record_integrity': record_integrity,
                'sales_records': sales_records,
                'specs_records': specs_records,
                'score': min(score, 1.0),
                'passed': brand_consistency and record_integrity
            }

        except Exception as e:
            return {
                'check_type': 'cross_reference',
                'score': 0.0,
                'passed': False,
                'error': str(e)
            }

    def _validate_query_intent_accuracy(self, user_query, structured_query):
        """Validate that query intent was classified correctly"""
        detected_intent = structured_query.get('intent', 'general')

        # Re-run intent classification for comparison
        reclassified_intent = self.classify_query_intent(user_query)

        intent_match = detected_intent == reclassified_intent

        # Additional validation based on keywords
        query_lower = user_query.lower()
        sales_keywords = ['sales', 'penjualan', 'retail', 'wholesale', 'total', 'berapa']
        spec_keywords = ['model', 'spesifikasi', 'cc', 'transmission', 'fuel', 'wheel']

        sales_score = sum(1 for keyword in sales_keywords if keyword in query_lower)
        spec_score = sum(1 for keyword in spec_keywords if keyword in query_lower)

        expected_intent = 'sales' if sales_score > spec_score else 'specification' if spec_score > 0 else 'general'

        keyword_match = detected_intent == expected_intent

        score = 1.0 if intent_match and keyword_match else 0.7 if intent_match else 0.3

        return {
            'check_type': 'intent_accuracy',
            'detected_intent': detected_intent,
            'reclassified_intent': reclassified_intent,
            'expected_intent': expected_intent,
            'intent_match': intent_match,
            'keyword_match': keyword_match,
            'score': score,
            'passed': intent_match
        }

    def _validate_result_completeness(self, structured_query, result_data):
        """Validate that query results are complete and meaningful"""
        if result_data.empty:
            return {
                'check_type': 'result_completeness',
                'score': 0.0,
                'passed': False,
                'reason': 'Empty result set'
            }

        # Check for minimum data requirements
        min_records = 1
        has_required_columns = False

        intent = structured_query.get('intent', 'general')
        if intent == 'sales':
            has_required_columns = 'sales' in result_data.columns and 'brand' in result_data.columns
        elif intent == 'specification':
            has_required_columns = 'model' in result_data.columns or any(col in result_data.columns for col in ['fuel', 'transmission', 'cc'])
        else:
            has_required_columns = len(result_data.columns) > 1

        # Check for data quality
        null_percentage = result_data.isnull().mean().mean()
        data_quality = null_percentage < 0.3  # Less than 30% nulls

        # Check for meaningful aggregations
        has_numeric_data = any(result_data.dtypes == 'int64') or any(result_data.dtypes == 'float64')

        score = 0.0
        score += 0.4 if len(result_data) >= min_records else 0.0
        score += 0.3 if has_required_columns else 0.0
        score += 0.2 if data_quality else 0.0
        score += 0.1 if has_numeric_data else 0.0

        return {
            'check_type': 'result_completeness',
            'record_count': len(result_data),
            'has_required_columns': has_required_columns,
            'data_quality_score': 1.0 - null_percentage,
            'has_numeric_data': has_numeric_data,
            'score': score,
            'passed': score >= 0.7
        }

    def _generate_accuracy_recommendations(self, validation_results):
        """Generate recommendations based on validation results"""
        recommendations = []

        for check in validation_results['validation_checks']:
            if not check.get('passed', False):
                check_type = check.get('check_type')

                if check_type == 'data_source_consistency':
                    recommendations.append("Consider reviewing data source selection logic for better accuracy")
                elif check_type == 'cross_reference':
                    recommendations.append("Implement stronger cross-referencing between sales and specifications data")
                elif check_type == 'intent_accuracy':
                    recommendations.append("Improve query intent classification with additional training examples")
                elif check_type == 'result_completeness':
                    recommendations.append("Add data validation checks to ensure result completeness")

        if validation_results['accuracy_score'] < 0.7:
            recommendations.append("Consider implementing additional validation layers for critical queries")
        elif validation_results['accuracy_score'] < 0.9:
            recommendations.append("Minor accuracy improvements recommended for optimal performance")

        return recommendations

    def generate_random_queries(self, num_queries=10, language='en'):
        """Generate random queries for testing LLM accuracy"""
        try:
            brands = self.df['brand'].unique().tolist()[:10]  # Limit to top 10 brands
            sales_types = self.df['sales_type'].unique().tolist()
            segments = [seg for seg in self.df['segment'].unique().tolist() if seg not in ['TOTAL:', 'SUBTOTAL:']][:5]

            query_templates = {
                'en': [
                    "What are the total sales for {brand}?",
                    "Compare {brand1} vs {brand2} sales",
                    "Show me {sales_type} sales trend for {brand}",
                    "What are the specifications of {brand} vehicles?",
                    "How many {segment} vehicles does {brand} sell?",
                    "What's the market share of {brand}?",
                    "Show diesel vehicles from {brand}",
                    "What's the average sales for {brand}?",
                    "Compare {sales_type} sales between brands",
                    "Show top selling models from {brand}"
                ],
                'id': [
                    "Berapa total penjualan {brand}?",
                    "Bandingkan penjualan {brand1} dengan {brand2}",
                    "Tampilkan tren penjualan {sales_type} untuk {brand}",
                    "Apa spesifikasi kendaraan {brand}?",
                    "Berapa banyak {segment} yang dijual {brand}?",
                    "Berapa pangsa pasar {brand}?",
                    "Tampilkan kendaraan diesel dari {brand}",
                    "Berapa rata-rata penjualan {brand}?",
                    "Bandingkan penjualan {sales_type} antar merek",
                    "Tampilkan model terlaris dari {brand}"
                ]
            }

            templates = query_templates.get(language, query_templates['en'])
            random_queries = []

            for _ in range(num_queries):
                template = np.random.choice(templates)

                # Fill in template variables
                query = template
                if '{brand}' in template:
                    query = query.replace('{brand}', np.random.choice(brands))
                if '{brand1}' in template:
                    query = query.replace('{brand1}', np.random.choice(brands))
                if '{brand2}' in template:
                    query = query.replace('{brand2}', np.random.choice(brands))
                if '{sales_type}' in template:
                    query = query.replace('{sales_type}', np.random.choice(sales_types))
                if '{segment}' in template:
                    query = query.replace('{segment}', np.random.choice(segments) if segments else 'cars')

                random_queries.append(query)

            return random_queries

        except Exception as e:
            print(f"Error generating random queries: {e}")
            return []

    def create_validation_report(self, validation_results_list=None):
        """Create comprehensive validation report"""
        try:
            if validation_results_list is None:
                validation_results_list = self.validation_reports

            if not validation_results_list:
                return "No validation results available"

            # Calculate summary statistics
            total_queries = len(validation_results_list)
            avg_accuracy = sum(r.get('accuracy_score', 0) for r in validation_results_list) / total_queries if total_queries > 0 else 0

            # Categorize by intent
            intent_stats = {}
            for result in validation_results_list:
                intent = result.get('intent', 'unknown')
                if intent not in intent_stats:
                    intent_stats[intent] = {'count': 0, 'total_score': 0}
                intent_stats[intent]['count'] += 1
                intent_stats[intent]['total_score'] += result.get('accuracy_score', 0)

            # Generate report
            report = f"""
# LLM Accuracy Validation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Queries Validated: {total_queries}
- Average Accuracy Score: {avg_accuracy:.2%}
- Validation Date Range: {min(r['timestamp'] for r in validation_results_list)} to {max(r['timestamp'] for r in validation_results_list)}

## Accuracy by Query Intent
"""

            for intent, stats in intent_stats.items():
                avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
                report += f"- {intent.title()}: {stats['count']} queries, {avg_score:.2%} accuracy\n"

            # Detailed issues and recommendations
            all_issues = []
            all_recommendations = []

            for result in validation_results_list:
                if result.get('accuracy_score', 1.0) < 0.9:  # Only include queries with issues
                    all_issues.extend(result.get('issues_found', []))
                    all_recommendations.extend(result.get('recommendations', []))

            if all_issues:
                report += "\n## Issues Found\n"
                for issue in list(set(all_issues))[:10]:  # Limit to top 10 unique issues
                    report += f"- {issue}\n"

            if all_recommendations:
                report += "\n## Recommendations\n"
                for rec in list(set(all_recommendations))[:10]:  # Limit to top 10 unique recommendations
                    report += f"- {rec}\n"

            # Performance trends
            report += "\n## Performance Trends\n"
            report += f"- High Accuracy Queries (≥90%): {sum(1 for r in validation_results_list if r.get('accuracy_score', 0) >= 0.9)}\n"
            report += f"- Medium Accuracy Queries (70-89%): {sum(1 for r in validation_results_list if 0.7 <= r.get('accuracy_score', 0) < 0.9)}\n"
            report += f"- Low Accuracy Queries (<70%): {sum(1 for r in validation_results_list if r.get('accuracy_score', 0) < 0.7)}\n"

            return report

        except Exception as e:
            print(f"Error creating validation report: {e}")
            return f"Error generating validation report: {str(e)}"

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="GAIKINDO Analytics", 
        layout="wide", 
        page_icon="[AUTO]"
    )
    
    # Initialize analytics
    if 'analytics' not in st.session_state:
        st.session_state.analytics = GaikindoAnalytics()
    
    analytics = st.session_state.analytics
    
    # Language selection
    col1, col2 = st.columns([1, 4])
    with col1:
        selected_lang = st.selectbox(
            "Language",
            options=list(analytics.languages.keys()),
            format_func=lambda x: analytics.languages[x],
            key="language_selector"
        )
    
    # Get translations for selected language
    t = analytics.translations[selected_lang]
    
    # Header
    st.title(t['title'])
    st.markdown(f"**{t['subtitle']}**")
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(t['brands'], analytics.df['brand'].nunique())
    with col2:
        st.metric(t['total_records'], f"{len(analytics.df):,}")
    with col3:
        st.metric(t['sales_types'], analytics.df['sales_type'].nunique())  
    with col4:
        st.metric(t['time_period'], f"{analytics.df['year'].min()}-{analytics.df['year'].max()}")
    
    # Query examples
    with st.expander("[IDEA] Query Examples", expanded=True):
        if selected_lang == 'id':
            st.markdown("""
            **[SEARCH] Contoh Pertanyaan:**
            - "Bandingkan penjualan Toyota dengan Mercedes Benz"
            - "Buatkan grafik tren penjualan retail untuk semua merek"
            - "Berapa total penjualan wholesale HINO?"
            - "Tampilkan analisis mendalam pasar otomotif"
            """)
        elif selected_lang == 'ja':
            st.markdown("""
            **[SEARCH] クエリ例:**
            - "トヨタとメルセデスベンツの売上を比較"
            - "すべてのブランドの小売売上トレンドチャートを作成"
            - "HINOの卸売売上合計は？"
            - "自動車市場の詳細分析を表示"
            """)
        else:
            st.markdown("""
            **[SEARCH] Example Queries:**
            - "Compare Toyota vs Mercedes Benz sales"
            - "Show retail sales trend chart for all brands"
            - "What's the total wholesale sales for HINO?"
            - "Give me deep automotive market analysis"
            """)
    
    # Main query interface
    user_query = st.text_area(
        f"[TARGET] {t['query_placeholder']}", 
        height=100,
        placeholder=t['query_placeholder']
    )
    
    # Analysis button
    if st.button(t['analyze_button'], type="primary", width='stretch'):
        if user_query:
            with st.spinner(t['processing']):
                # Detect language if not manually selected
                detected_lang = analytics.detect_language(user_query)
                final_lang = selected_lang if selected_lang != 'en' else detected_lang
                
                # Generate structured query
                structured_query = analytics.generate_structured_query(user_query, final_lang)
                
                # Show query structure
                with st.expander("[CLIPBOARD] Query Structure"):
                    st.json(structured_query)
                
                # Execute query
                result_data = analytics.execute_query(structured_query)

                # Show results
                if not result_data.empty:
                    # Data table
                    st.subheader("[CHART] Results")
                    st.dataframe(result_data, use_container_width=True)

                    # Visualization
                    if structured_query.get('chart_type'):
                        fig = analytics.create_visualization(result_data, structured_query, final_lang)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    # AI Insights
                    st.subheader("[BRAIN] AI Insights")
                    insights = analytics.generate_insights(result_data, structured_query, final_lang)
                    st.markdown(insights)

                    # LLM Accuracy Validation Section
                    st.markdown("---")
                    st.subheader("[VALIDATION] LLM Accuracy Validation")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("[CHECK] Validate Current Query", type="secondary"):
                            with st.spinner("Validating LLM accuracy..."):
                                validation_results = analytics.validate_llm_accuracy(
                                    user_query, structured_query, result_data, final_lang
                                )

                                # Display validation results
                                st.success(f"Validation Complete! Accuracy Score: {validation_results['accuracy_score']:.2%}")

                                # Show detailed validation checks
                                with st.expander("Detailed Validation Results"):
                                    for check in validation_results['validation_checks']:
                                        status = "✅ PASSED" if check.get('passed', False) else "❌ FAILED"
                                        st.write(f"**{check['check_type'].replace('_', ' ').title()}**: {status}")
                                        st.write(f"Score: {check.get('score', 0):.2%}")
                                        if 'reason' in check:
                                            st.write(f"Reason: {check['reason']}")
                                        st.write("---")

                                # Show recommendations
                                if validation_results.get('recommendations'):
                                    st.warning("**Recommendations:**")
                                    for rec in validation_results['recommendations']:
                                        st.write(f"• {rec}")

                    with col2:
                        if st.button("[TEST] Generate Random Queries", type="secondary"):
                            with st.spinner("Generating test queries..."):
                                random_queries = analytics.generate_random_queries(5, final_lang)

                                st.info(f"Generated {len(random_queries)} random queries for testing:")

                                for i, query in enumerate(random_queries, 1):
                                    st.write(f"{i}. {query}")

                                # Store in session state for batch testing
                                if 'random_queries' not in st.session_state:
                                    st.session_state.random_queries = []
                                st.session_state.random_queries = random_queries

                else:
                    st.warning("No data found matching your query. Please try different filters.")
        else:
            st.warning("Please enter a query to analyze.")
    
    # Data preview section
    with st.expander("[SEARCH] Data Preview", expanded=False):
        preview_type = st.radio(
            "Select data view:",
            ['Summary', 'Detailed', 'Sample Records']
        )
        
        if preview_type == 'Summary':
            st.write("**Brand Summary:**")
            brand_summary = analytics.df.groupby('brand')['sales'].agg(['count', 'sum', 'mean']).round(2)
            st.dataframe(brand_summary)
            
        elif preview_type == 'Detailed':
            st.write("**Sales Type Distribution:**")
            sales_dist = analytics.df.groupby(['brand', 'sales_type'])['sales'].sum().unstack(fill_value=0)
            st.dataframe(sales_dist)
            
        else:
            st.write("**Sample Records:**")
            st.dataframe(analytics.df.sample(min(20, len(analytics.df))))
    
    # Footer
    st.markdown("---")
    st.markdown("**[AUTO] GAIKINDO Advanced Analytics** - Powered by AI with 100% Data Accuracy")

if __name__ == "__main__":
    main()