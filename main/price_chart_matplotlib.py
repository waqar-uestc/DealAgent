"""
Price Chart Generator using Matplotlib
Creates static price trend charts as images
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict
import io
import base64


def generate_price_trend_chart_img(price_data: List[Dict], title: str) -> str:
    """
    Generate a price trend chart as base64 encoded image.
    
    Args:
        price_data: List of price history entries
        title: Deal title
    
    Returns:
        Base64 encoded PNG image string
    """
    if not price_data:
        # Create empty chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No price history available', 
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Price Trend: {title[:60]}...")
    else:
        # Extract data
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in price_data]
        prices = [entry["price"] for entry in price_data]
        estimated_values = [entry.get("estimated_value", 0) for entry in price_data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual price
        ax.plot(timestamps, prices, 'o-', linewidth=2, markersize=8, 
                label='Actual Price', color='#1f77b4')
        
        # Plot estimated value if available
        if any(estimated_values):
            ax.plot(timestamps, estimated_values, '--', linewidth=2, 
                    label='Estimated Value', color='#ff7f0e', alpha=0.7)
        
        # Calculate statistics
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Format
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f"Price Trend: {title[:60]}...", fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add statistics text
        stats_text = f"Min: ${min_price:.2f} | Max: ${max_price:.2f} | Avg: ${avg_price:.2f}"
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, 
                ha='center', fontsize=10, color='gray')
        
        plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_base64}" style="width:100%; max-width:800px; margin:10px 0;"/>'

