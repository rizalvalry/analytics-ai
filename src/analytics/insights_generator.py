import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import google.generativeai as genai
import json
import streamlit as st

class GaikindoInsightsGenerator:
    """Generate business insights and predictions from GAIKINDO sales data"""
    
    def __init__(self, df, api_key):
        self.df = df
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def generate_market_insights(self, sales_data_summary):
        """Generate AI-powered market insights"""
        prompt = f"""
        Analisis data penjualan otomotif GAIKINDO berikut dan berikan 3-4 insight bisnis yang actionable:
        
        Data Summary:
        {sales_data_summary}
        
        Fokus pada:
        1. Tren pasar dan performa brand
        2. Peluang pertumbuhan
        3. Segmen yang underperform
        4. Rekomendasi strategis
        
        Format: Bullet points yang ringkas dan mudah dipahami.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insights: {e}"
    
    def brand_performance_analysis(self):
        """Analyze brand performance with detailed metrics"""
        analysis = {}
        
        for brand in self.df['brand'].unique():
            brand_data = self.df[self.df['brand'] == brand]
            
            # Calculate key metrics
            total_sales = brand_data['sales'].sum()
            avg_monthly_sales = brand_data.groupby('month')['sales'].sum().mean()
            
            # Sales type breakdown
            sales_by_type = brand_data.groupby('sales_type')['sales'].sum().to_dict()
            
            # Monthly trend
            monthly_trend = brand_data.groupby('month')['sales'].sum().to_dict()
            
            # Market position
            market_share = (total_sales / self.df['sales'].sum() * 100)
            
            analysis[brand] = {
                'total_sales': int(total_sales),
                'avg_monthly_sales': int(avg_monthly_sales),
                'market_share': round(market_share, 2),
                'sales_by_type': {k: int(v) for k, v in sales_by_type.items()},
                'monthly_trend': {k: int(v) for k, v in monthly_trend.items()},
                'dominant_sales_type': max(sales_by_type, key=sales_by_type.get)
            }
            
        return analysis
    
    def predict_future_sales(self, brand=None, sales_type='retail_sales', periods=3):
        """Simple linear prediction for future sales"""
        df_filtered = self.df[self.df['sales_type'] == sales_type].copy()
        
        if brand:
            df_filtered = df_filtered[df_filtered['brand'] == brand]
        
        # Aggregate monthly data
        month_order = ['jan', 'feb', 'mar', 'apr']
        monthly_data = df_filtered.groupby('month')['sales'].sum().reindex(month_order)
        
        # Prepare data for prediction
        X = np.array(range(len(monthly_data))).reshape(-1, 1)
        y = monthly_data.values
        
        # Simple linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future periods
        future_X = np.array(range(len(monthly_data), len(monthly_data) + periods)).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Calculate trend
        trend = "increasing" if model.coef_[0] > 0 else "decreasing"
        trend_strength = abs(model.coef_[0])
        
        future_months = ['may', 'jun', 'jul'][:periods]
        prediction_dict = {month: max(0, int(pred)) for month, pred in zip(future_months, predictions)}
        
        return {
            'predictions': prediction_dict,
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'confidence': 'medium'  # Simple heuristic
        }
    
    def competitive_analysis(self):
        """Analyze competitive landscape"""
        analysis = {}
        
        # Overall market leaders
        brand_totals = self.df.groupby('brand')['sales'].sum().sort_values(ascending=False)
        analysis['market_leaders'] = brand_totals.head(3).to_dict()
        
        # Sales type dominance
        for sales_type in self.df['sales_type'].unique():
            type_data = self.df[self.df['sales_type'] == sales_type]
            type_leaders = type_data.groupby('brand')['sales'].sum().sort_values(ascending=False)
            analysis[f'{sales_type}_leaders'] = type_leaders.head(3).to_dict()
        
        # Growth potential (based on variance)
        brand_variance = self.df.groupby('brand')['sales'].std().sort_values(ascending=False)
        analysis['high_variance_brands'] = brand_variance.head(3).to_dict()
        
        return analysis
    
    def seasonal_patterns(self):
        """Identify seasonal patterns in sales"""
        monthly_totals = self.df.groupby('month')['sales'].sum()
        
        # Calculate seasonal indices
        average_monthly = monthly_totals.mean()
        seasonal_indices = (monthly_totals / average_monthly * 100).round(2)
        
        # Identify peak and low seasons
        peak_month = seasonal_indices.idxmax()
        low_month = seasonal_indices.idxmin()
        
        return {
            'seasonal_indices': seasonal_indices.to_dict(),
            'peak_month': peak_month,
            'low_month': low_month,
            'seasonality_strength': round(seasonal_indices.std(), 2)
        }
    
    def risk_assessment(self):
        """Assess market risks and opportunities"""
        risks = []
        opportunities = []
        
        # Brand concentration risk
        brand_concentration = self.df.groupby('brand')['sales'].sum()
        top_brand_share = (brand_concentration.max() / brand_concentration.sum() * 100)
        
        if top_brand_share > 40:
            risks.append(f"High market concentration: Top brand has {top_brand_share:.1f}% market share")
        
        # Sales type diversification
        sales_type_distribution = self.df.groupby('sales_type')['sales'].sum()
        dominant_type_share = (sales_type_distribution.max() / sales_type_distribution.sum() * 100)
        
        if dominant_type_share > 50:
            risks.append(f"Over-reliance on one sales type: {dominant_type_share:.1f}%")
        
        # Identify underperforming segments
        brand_performance = self.df.groupby('brand')['sales'].sum().sort_values()
        bottom_performers = brand_performance.head(2)
        
        for brand, sales in bottom_performers.items():
            market_share = sales / self.df['sales'].sum() * 100
            if market_share < 5:
                opportunities.append(f"{brand} has only {market_share:.1f}% market share - growth potential")
        
        return {
            'risks': risks,
            'opportunities': opportunities,
            'overall_risk_level': 'medium' if len(risks) > 2 else 'low'
        }
    
    def generate_executive_summary(self):
        """Generate comprehensive executive summary"""
        # Gather all analysis components
        brand_analysis = self.brand_performance_analysis()
        competitive_analysis = self.competitive_analysis()
        seasonal_patterns = self.seasonal_patterns()
        risk_assessment = self.risk_assessment()
        
        # Create summary data for AI analysis
        summary_data = {
            'total_market_size': int(self.df['sales'].sum()),
            'number_of_brands': len(self.df['brand'].unique()),
            'top_3_brands': list(competitive_analysis['market_leaders'].keys()),
            'peak_season': seasonal_patterns['peak_month'],
            'main_risks': risk_assessment['risks'][:2],
            'top_opportunities': risk_assessment['opportunities'][:2]
        }
        
        # Generate AI insights
        ai_insights = self.generate_market_insights(json.dumps(summary_data, indent=2))
        
        return {
            'summary_data': summary_data,
            'ai_insights': ai_insights,
            'brand_analysis': brand_analysis,
            'competitive_landscape': competitive_analysis,
            'seasonal_trends': seasonal_patterns,
            'risk_opportunities': risk_assessment
        }