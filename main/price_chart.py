"""
Price Chart Generator
Creates interactive price trend charts using Plotly
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict
import pandas as pd


def generate_price_trend_chart(price_data: List[Dict], title: str) -> go.Figure:
    """
    Generate an interactive price trend chart.
    
    Args:
        price_data: List of price history entries
        title: Deal title
    
    Returns:
        Plotly Figure object
    """
    if not price_data:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No price history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f"Price Trend: {title}",
            height=400
        )
        return fig
    
    # Extract data
    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in price_data]
    prices = [entry["price"] for entry in price_data]
    estimated_values = [entry.get("estimated_value", 0) for entry in price_data]
    
    # Create figure
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add estimated value line if available
    if any(estimated_values):
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=estimated_values,
            mode='lines',
            name='Estimated Value',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Est. Value:</b> $%{y:.2f}<extra></extra>'
        ))
    
    # Calculate statistics
    min_price = min(prices)
    max_price = max(prices)
    avg_price = sum(prices) / len(prices)
    
    # Update layout
    fig.update_layout(
        title=f"Price Trend: {title}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text=f"Min: ${min_price:.2f} | Max: ${max_price:.2f} | Avg: ${avg_price:.2f}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor='center'
            )
        ]
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig


def generate_multi_deal_comparison(deals_data: Dict[str, List[Dict]]) -> go.Figure:
    """
    Generate comparison chart for multiple deals.
    
    Args:
        deals_data: Dictionary mapping deal titles to price history
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (deal_title, price_data) in enumerate(deals_data.items()):
        if not price_data:
            continue
        
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in price_data]
        prices = [entry["price"] for entry in price_data]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines+markers',
            name=deal_title[:30] + "..." if len(deal_title) > 30 else deal_title,
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br><b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Price Comparison: Multiple Deals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig


def generate_discount_trend_chart(price_data: List[Dict], title: str) -> go.Figure:
    """
    Generate discount trend chart.
    
    Args:
        price_data: List of price history entries with discount info
        title: Deal title
    
    Returns:
        Plotly Figure object
    """
    if not price_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No discount history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in price_data]
    discounts = [entry.get("discount", 0) for entry in price_data]
    
    fig = go.Figure()
    
    # Add discount bar chart
    fig.add_trace(go.Bar(
        x=timestamps,
        y=discounts,
        name='Discount %',
        marker_color='lightgreen',
        hovertemplate='<b>Date:</b> %{x}<br><b>Discount:</b> %{y:.1f}%<extra></extra>'
    ))
    
    # Add average line
    avg_discount = sum(discounts) / len(discounts) if discounts else 0
    fig.add_hline(
        y=avg_discount,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_discount:.1f}%",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f"Discount Trend: {title}",
        xaxis_title="Date",
        yaxis_title="Discount (%)",
        hovermode='x',
        template='plotly_white',
        height=400
    )
    
    return fig


def generate_summary_statistics_table(deals_stats: List[Dict]) -> go.Figure:
    """
    Generate summary statistics table.
    
    Args:
        deals_stats: List of deal statistics
    
    Returns:
        Plotly Figure table
    """
    if not deals_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No statistics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare table data
    titles = [stat["title"][:40] + "..." if len(stat["title"]) > 40 else stat["title"] 
              for stat in deals_stats]
    current_prices = [f"${stat['current_price']:.2f}" for stat in deals_stats]
    min_prices = [f"${stat['min_price']:.2f}" for stat in deals_stats]
    max_prices = [f"${stat['max_price']:.2f}" for stat in deals_stats]
    avg_prices = [f"${stat['avg_price']:.2f}" for stat in deals_stats]
    trends = [stat['price_trend'] for stat in deals_stats]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Deal Title</b>', '<b>Current</b>', '<b>Min</b>', 
                   '<b>Max</b>', '<b>Avg</b>', '<b>Trend</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[titles, current_prices, min_prices, max_prices, avg_prices, trends],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title="Price Statistics Summary",
        height=400
    )
    
    return fig

