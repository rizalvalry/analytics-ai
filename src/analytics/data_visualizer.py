import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

class GaikindoVisualizer:
    """Advanced data visualization for GAIKINDO automotive sales data"""
    
    def __init__(self, df):
        self.df = df
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800'
        }
    
    def sales_trend_chart(self, brands=None, sales_type='retail_sales'):
        """Create interactive sales trend chart"""
        df_filtered = self.df[self.df['sales_type'] == sales_type].copy()
        
        if brands:
            df_filtered = df_filtered[df_filtered['brand'].isin(brands)]
        
        # Convert month names to numbers for proper sorting
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        df_filtered['month_num'] = df_filtered['month'].map({m: i+1 for i, m in enumerate(month_order)})
        df_filtered = df_filtered.sort_values('month_num')
        
        # Aggregate by month and brand
        monthly_data = df_filtered.groupby(['month', 'brand'])['sales'].sum().reset_index()
        
        fig = px.line(monthly_data, x='month', y='sales', color='brand',
                     title=f'Sales Trend - {sales_type.replace("_", " ").title()}',
                     labels={'sales': 'Sales Units', 'month': 'Month'},
                     markers=True)
        
        fig.update_layout(
            hovermode='x unified',
            xaxis_title="Month",
            yaxis_title="Sales Units",
            legend_title="Brand"
        )
        
        return fig
    
    def brand_comparison_chart(self, sales_type='retail_sales'):
        """Create brand comparison chart"""
        df_filtered = self.df[self.df['sales_type'] == sales_type].copy()
        brand_totals = df_filtered.groupby('brand')['sales'].sum().reset_index()
        brand_totals = brand_totals.sort_values('sales', ascending=False)
        
        fig = px.bar(brand_totals, x='brand', y='sales',
                    title=f'Brand Comparison - {sales_type.replace("_", " ").title()}',
                    labels={'sales': 'Total Sales Units', 'brand': 'Brand'},
                    color='sales',
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            xaxis_title="Brand",
            yaxis_title="Total Sales Units",
            showlegend=False
        )
        
        return fig
    
    def sales_type_distribution(self, brands=None):
        """Create sales type distribution chart"""
        df_filtered = self.df.copy()
        
        if brands:
            df_filtered = df_filtered[df_filtered['brand'].isin(brands)]
        
        sales_type_totals = df_filtered.groupby('sales_type')['sales'].sum().reset_index()
        
        # Rename sales types for better display
        type_names = {
            'wholesales': 'Wholesales',
            'retail_sales': 'Retail Sales', 
            'production_ckd': 'Production CKD',
            'import_cbu': 'Import CBU'
        }
        sales_type_totals['sales_type_display'] = sales_type_totals['sales_type'].map(type_names)
        
        fig = px.pie(sales_type_totals, values='sales', names='sales_type_display',
                    title='Sales Distribution by Type',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def monthly_heatmap(self, brands=None):
        """Create monthly sales heatmap"""
        df_filtered = self.df.copy()
        
        if brands:
            df_filtered = df_filtered[df_filtered['brand'].isin(brands)]
        
        # Pivot table for heatmap
        heatmap_data = df_filtered.groupby(['brand', 'month'])['sales'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='brand', columns='month', values='sales').fillna(0)
        
        # Ensure proper month order
        month_order = ['jan', 'feb', 'mar', 'apr']
        heatmap_pivot = heatmap_pivot.reindex(columns=month_order, fill_value=0)
        
        fig = px.imshow(heatmap_pivot.values,
                       labels=dict(x="Month", y="Brand", color="Sales"),
                       x=heatmap_pivot.columns,
                       y=heatmap_pivot.index,
                       title="Sales Heatmap by Brand and Month",
                       color_continuous_scale='Blues')
        
        return fig
    
    def market_share_analysis(self, sales_type='retail_sales'):
        """Create market share analysis"""
        df_filtered = self.df[self.df['sales_type'] == sales_type].copy()
        
        # Calculate market share
        brand_totals = df_filtered.groupby('brand')['sales'].sum().reset_index()
        total_market = brand_totals['sales'].sum()
        brand_totals['market_share'] = (brand_totals['sales'] / total_market * 100).round(2)
        brand_totals = brand_totals.sort_values('market_share', ascending=False)
        
        # Create combined chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sales Volume', 'Market Share %'),
            specs=[[{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Bar chart for sales volume
        fig.add_trace(
            go.Bar(x=brand_totals['brand'], y=brand_totals['sales'], 
                   name='Sales Volume', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Pie chart for market share
        fig.add_trace(
            go.Pie(labels=brand_totals['brand'], values=brand_totals['market_share'],
                   name="Market Share", textinfo='label+percent'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Market Analysis - {sales_type.replace('_', ' ').title()}",
            showlegend=False
        )
        
        return fig, brand_totals
    
    def performance_metrics_dashboard(self):
        """Create comprehensive performance dashboard"""
        metrics = {}
        
        # Total sales by type
        for sales_type in self.df['sales_type'].unique():
            type_data = self.df[self.df['sales_type'] == sales_type]
            metrics[f'{sales_type}_total'] = type_data['sales'].sum()
            metrics[f'{sales_type}_avg'] = type_data['sales'].mean()
        
        # Top performers
        metrics['top_brand'] = self.df.groupby('brand')['sales'].sum().idxmax()
        metrics['top_brand_sales'] = self.df.groupby('brand')['sales'].sum().max()
        
        # Monthly performance
        monthly_totals = self.df.groupby('month')['sales'].sum()
        metrics['best_month'] = monthly_totals.idxmax()
        metrics['best_month_sales'] = monthly_totals.max()
        
        return metrics
    
    def growth_analysis(self):
        """Analyze month-over-month growth"""
        month_order = ['jan', 'feb', 'mar', 'apr']
        monthly_data = self.df.groupby('month')['sales'].sum().reindex(month_order)
        
        # Calculate growth rates
        growth_rates = monthly_data.pct_change() * 100
        growth_rates = growth_rates.dropna()
        
        fig = px.bar(x=growth_rates.index, y=growth_rates.values,
                    title='Month-over-Month Growth Rate (%)',
                    labels={'x': 'Month', 'y': 'Growth Rate (%)'},
                    color=growth_rates.values,
                    color_continuous_scale='RdYlGn')
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Growth Rate (%)",
            showlegend=False
        )
        
        return fig, growth_rates