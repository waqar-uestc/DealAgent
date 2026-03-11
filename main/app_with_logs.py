import os
# Set environment variables before importing ML libraries
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import html
import plotly.express as px  # Used in generate_visualization for 3D scatter plots

from fetch_deals import fetch_deals_rss, search_deals_by_keyword
from gpt_evaluator import gpt_evaluate_deal
from price_model import predict_value
from rag_faiss import query_deals_rag, query_search_rag
from model_manager import ModelManager
from shared_data import (set_top_deals, get_top_deals, append_log, set_llm_provider,
                        set_search_results, get_search_results, get_current_search_keyword)
from config import Config
from price_history import get_price_history_manager
from price_chart import generate_summary_statistics_table
from price_chart_matplotlib import generate_price_trend_chart_img

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize default LLM provider
DEFAULT_PROVIDER = "OpenAI"
set_llm_provider(Config.DEFAULT_LLM_PROVIDER)

# Force light theme CSS overrides (ensure white backgrounds)
LIGHT_THEME_CSS = """
:root { color-scheme: light; }
body, .gradio-container { background: #ffffff !important; color: #111 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container .prose,
.gradio-container .form, .gradio-container .panel, .gradio-container .container,
.gradio-container .wrap, .gradio-container .block, .gradio-container .tabs,
.gradio-container .tabitem { background-color: #ffffff !important; color: #111 !important; }
.gradio-container button { color: #111; }
/* Force normal (sans-serif) font for tabs and buttons */
.gradio-container button,
.gradio-container [role="tab"] {
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif !important;
    font-weight: 500;
}

/* Enlarge Deals HTML panel */
#deal_output { width: 100% !important; }
#deal_output .html-container {
    min-height: 300px !important;
    max-height: 50vh;
    padding: 8px;
    font-size: 16px;
    resize: vertical;
    overflow: auto;
}

/* Make RAG textareas larger by default and resizable */
#rag_question textarea,
#rag_answer textarea {
    min-height: 160px;
    max-height: 50vh;
    resize: vertical;
}

/* Ensure Plotly charts display properly */
.plotly-graph-div {
    width: 100% !important;
    min-height: 450px !important;
}
"""



def _evaluate_single_deal(deal: dict) -> dict:
    """
    Evaluate a single deal: get AI evaluation, predict value, calculate discount, and apply quality checks.
    
    Args:
        deal: Dictionary with 'title', 'price', 'link', 'summary'
    
    Returns:
        dict: Processed deal dictionary with evaluation, or None if rejected
    """
    title = deal['title']
    price = deal['price']
    link = deal['link']
    summary = deal.get('summary', '')
    
    # Get AI evaluation with error handling
    try:
        gpt_response = gpt_evaluate_deal(title, price)
    except Exception as e:
        append_log(f"[{datetime.now()}] ⚠️ LLM evaluation failed for {title[:50]}: {str(e)}")
        gpt_response = "Evaluation unavailable"
    
    est_value = predict_value(title)
    discount = round((1 - price / est_value) * 100, 2) if est_value else 0.0

    # Apply data quality checks
    deal_quality = _check_deal_quality(title, price, est_value, discount)
    
    # Skip deals with critical issues
    if deal_quality['status'] == 'reject':
        append_log(f"[{datetime.now()}] ⚠️ Filtered out deal: {title} (Reason: {deal_quality['reason']})")
        return None
    
    # Apply corrections if needed
    if deal_quality['status'] == 'corrected':
        est_value = deal_quality.get('corrected_est_value', est_value)
        discount = deal_quality.get('corrected_discount', discount)
        append_log(f"[{datetime.now()}] 🔧 Corrected deal: {title} (Reason: {deal_quality['reason']})")

    # Record price history
    price_mgr = get_price_history_manager()
    price_mgr.add_price_entry(
        title=title,
        price=price,
        link=link,
        estimated_value=est_value,
        discount=discount
    )

    processed_deal = {
        'title': title,
        'price': price,
        'link': link,
        'gpt_response': gpt_response,
        'est_value': est_value,
        'discount': discount,
        'description': f"{title} at ${price:.2f}. {summary}" if summary else f"{title} at ${price:.2f}",
        'quality_flag': deal_quality.get('flag', 'normal'),
    }
    
    append_log(f"[{datetime.now()}] Deal evaluated: {title} (${price}) → {gpt_response}")
    return processed_deal


def _check_deal_quality(title: str, price: float, est_value: float, discount: float) -> dict:
    """
    P1: Anomaly Filter - Data Quality Check and Correction.
    
    Rules:
    1. Negative discount filter: Reject if discount < 0 (product overpriced)
    2. Extreme discount check: Flag if discount > 95% and price > $10 (suspicious)
    3. Price floor: For items < $5, cap estimate at $20 to prevent false discounts
    
    Returns:
        dict: {
            'status': 'accept' | 'reject' | 'corrected',
            'reason': str,
            'corrected_est_value': float (optional),
            'corrected_discount': float (optional),
            'flag': 'normal' | 'suspicious' | 'low_price'
        }
    """
    result = {
        'status': 'accept',
        'reason': '',
        'flag': 'normal'
    }
    
    # Rule 1: Negative discount filter (reject overpriced items)
    if discount < 0:
        result['status'] = 'reject'
        result['reason'] = f'Negative discount ({discount:.1f}%) - product appears overpriced'
        return result
    
    # Rule 2: Extreme discount check (suspicious data)
    if discount > 95.0 and price > 10.0:
        result['status'] = 'corrected'
        result['reason'] = f'Extreme discount ({discount:.1f}%) - possible pricing error'
        result['flag'] = 'suspicious'
        # Cap discount at 90% for suspicious items
        result['corrected_discount'] = min(discount, 90.0)
        result['corrected_est_value'] = price / (1 - result['corrected_discount'] / 100)
        return result
    
    # Rule 3: Price floor for low-price items
    if price < 5.0:
        if est_value > 20.0:
            result['status'] = 'corrected'
            result['reason'] = f'Low price item (${price:.2f}) with inflated estimate (${est_value:.2f})'
            result['flag'] = 'low_price'
            # Cap estimate at $20 for items under $5
            result['corrected_est_value'] = min(est_value, 20.0)
            result['corrected_discount'] = round((1 - price / result['corrected_est_value']) * 100, 2)
            return result
    
    # Rule 4: Sanity check for extremely high estimates
    if est_value > 5000 and price < 100:
        result['status'] = 'corrected'
        result['reason'] = f'Estimate too high (${est_value:.2f}) for price (${price:.2f})'
        result['flag'] = 'suspicious'
        # Recalculate with more conservative estimate
        result['corrected_est_value'] = price * 1.5  # Assume 33% discount max
        result['corrected_discount'] = round((1 - price / result['corrected_est_value']) * 100, 2)
        return result
    
    return result


def format_deals_as_html(deals):
    """
    Format deals as HTML with proper escaping to prevent XSS.
    
    Args:
        deals: List of deal dictionaries
    
    Returns:
        str: Formatted HTML string
    """
    html_output = """
    <style>
        .deal-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #fdfdfd;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.06);
        }
        .deal-title {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 5px;
        }
        .deal-meta {
            font-size: 14px;
            color: #333;
        }
        .deal-meta span {
            display: inline-block;
            margin-right: 10px;
        }
        .deal-eval {
            margin-top: 10px;
            font-style: italic;
            color: #333;
        }
        .deal-link a {
            display: inline-block;
            margin-top: 10px;
            color: #1a73e8;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
    """
    
    for d in deals:
        # Escape all user-provided content
        safe_title = html.escape(d.get('title', ''))
        safe_response = html.escape(d.get('gpt_response', ''))
        safe_link = d.get('link', '')
        
        # Validate link starts with http/https
        if not safe_link.startswith(('http://', 'https://')):
            safe_link = "#"
        else:
            safe_link = html.escape(safe_link)
        
        # Numeric values are safe but still validate
        price = float(d.get('price', 0))
        est_value = float(d.get('est_value', 0))
        discount = float(d.get('discount', 0))
        
        html_output += f"""
        <div class="deal-box">
            <div class="deal-title">{safe_title}</div>
            <div class="deal-meta">
                <span><strong>Price:</strong> ${price:.2f}</span>
                <span><strong>Estimated Value:</strong> ${est_value:.2f}</span>
                <span><strong>Discount:</strong> {discount:.2f}%</span>
            </div>
            <div class="deal-eval">{safe_response}</div>
            <div class="deal-link"><a href="{safe_link}" target="_blank" rel="noopener noreferrer">🔗 View Deal</a></div>
        </div>
        """
    
    return html_output


def show_top_deals(search_keyword: str = ""):
    """
    Fetch and display deals, optionally filtered by search keyword.
    
    Args:
        search_keyword: Optional keyword to filter deals. If empty, fetches all deals.
    
    Returns:
        str: HTML formatted deals
    """
    # If search keyword provided, use search function
    if search_keyword and search_keyword.strip():
        keyword = search_keyword.strip()
        append_log(f"[{datetime.now()}] 🔍 Fetching deals for keyword: {keyword}")
        raw_deals = search_deals_by_keyword(keyword)
        
        if not raw_deals:
            return f"<p>⚠️ No deals found matching '{html.escape(keyword)}'. Try a different keyword or leave search empty to fetch all deals.</p>"
    else:
        # Fetch all deals from RSS
        append_log(f"[{datetime.now()}] 📥 Fetching all deals from RSS feeds...")
        raw_deals = fetch_deals_rss()
    
    if not raw_deals:
        return "<p>⚠️ No deals found. Please check RSS sources or try a different search keyword.</p>"
    
    price_mgr = get_price_history_manager()
    processed_deals = []
    
    # Process deals (limit to DEALS_COUNT, default 150)
    max_deals = min(len(raw_deals), Config.DEALS_COUNT)
    
    # Limit processing to prevent timeout (process max deals at a time for better performance)
    process_limit = min(max_deals, Config.DEAL_PROCESS_LIMIT)
    if len(raw_deals) > process_limit:
        append_log(f"[{datetime.now()}] ⚡ Processing {process_limit} deals (out of {len(raw_deals)} found) for faster response...")
    else:
        append_log(f"[{datetime.now()}] ⚡ Processing {len(raw_deals)} deals...")
    
    for deal in raw_deals[:process_limit]:
        processed_deal = _evaluate_single_deal(deal, price_mgr)
        if processed_deal:
            processed_deals.append(processed_deal)

    # Set the processed deals (with description)
    set_top_deals(processed_deals)
    
    # Add header indicating search status
    header = ""
    if search_keyword and search_keyword.strip():
        header = f"<h3>🔍 Filtered Deals for: <strong>{html.escape(search_keyword)}</strong></h3>"
        header += f"<p>Found {len(processed_deals)} deals matching your search</p><hr>"
    else:
        header = f"<h3>📥 Deals from RSS Feeds</h3>"
        header += f"<p>Showing {len(processed_deals)} deals (up to {Config.DEALS_COUNT} total)</p><hr>"

    return header + format_deals_as_html(processed_deals)


def rag_answer(query: str):
    """Answer questions about current deals using RAG."""
    try:
        response = query_deals_rag(query)
        append_log(f"[{datetime.now()}] RAG Q: {query} → {response[:100]}...")
        return response
    except Exception as e:
        error_msg = f"⚠️ Error answering question: {str(e)}"
        append_log(f"[{datetime.now()}] ❌ RAG Error: {error_msg}")
        return error_msg


def generate_visualization(use_search_results: bool = False):
    """Generate 3D visualization of deal clusters using TSNE."""
    if use_search_results:
        deals = get_search_results()
        keyword = get_current_search_keyword()
        if not deals or len(deals) < 3:
            return f"⚠️ Need at least 3 search results for '{keyword}' to generate a 3D plot."
        title_suffix = f" - Search: {keyword}"
    else:
        deals = fetch_deals_rss()[:Config.MAX_DEALS]
        if not deals or len(deals) < 3:
            return "⚠️ Need at least 3 deals to generate a 3D plot."
        title_suffix = ""

    titles = [deal['title'] for deal in deals]
    prices = [deal['price'] for deal in deals]
    model = ModelManager.get_sentence_model()
    embeddings = model.encode(titles)

    n_samples = len(embeddings)
    perplexity = max(Config.TSNE_PERPLEXITY_MIN, min(Config.TSNE_PERPLEXITY_MAX, n_samples - 1))

    try:
        tsne = TSNE(n_components=3, perplexity=perplexity, random_state=Config.TSNE_RANDOM_STATE)
        tsne_3d = tsne.fit_transform(embeddings)

        fig = px.scatter_3d(
            x=tsne_3d[:, 0],
            y=tsne_3d[:, 1],
            z=tsne_3d[:, 2],
            color=prices,
            hover_name=titles,
            labels={"color": "Price ($)"},
            title=f"3D Semantic Clustering of Deal Titles{title_suffix}",
        )
        fig.update_layout(height=Config.PLOT_HEIGHT, width=Config.PLOT_WIDTH)
        fig.update_traces(marker=dict(size=Config.PLOT_MARKER_SIZE))

        if use_search_results:
            safe_keyword = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in keyword)[:50]
            file_path = Config.get_full_path(f"search_results_plot_{safe_keyword}.html")
        else:
            file_path = Config.get_full_path(Config.PLOT_OUTPUT_FILE)
        fig.write_html(file_path)
        append_log(f"[{datetime.now()}] 📊 3D plot generated{title_suffix}.")
        return file_path

    except Exception as e:
        error_msg = f"Plot error: {str(e)}"
        append_log(f"[{datetime.now()}] ❌ {error_msg}")
        return error_msg


def read_logs():
    """Read and return the contents of the logs file."""
    log_path = Config.get_full_path(Config.LOGS_FILE)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No logs found."


def update_llm_provider(choice: str):
    """Update LLM provider with error handling."""
    try:
        provider_name = choice.lower().strip()
        set_llm_provider(provider_name)
        
        # Test if provider is available (without causing errors)
        try:
            from gpt_evaluator import _get_provider
            provider = _get_provider(provider_name)
            if provider is None:
                return f"⚠️ LLM Provider set to: {choice}\n(Error: Could not initialize provider)"
            
            # Check availability with timeout to prevent hanging
            try:
                if provider.is_available():
                    return f"✅ LLM Provider set to: {choice}"
                else:
                    api_key_name = {
                        "openai": "OPENAI_API_KEY",
                        "gemini": "GEMINI_API_KEY", 
                        "deepseek": "DEEPSEEK_API_KEY"
                    }.get(provider_name, "API_KEY")
                    return f"⚠️ LLM Provider set to: {choice}\n(Note: {api_key_name} not configured or unavailable)"
            except Exception as e:
                # If availability check fails, still set it but warn user
                return f"⚠️ LLM Provider set to: {choice}\n(Warning: Could not verify availability: {str(e)})"
        except Exception as e:
            # If provider creation fails, still set it but warn user
            return f"⚠️ LLM Provider set to: {choice}\n(Warning: Could not initialize provider: {str(e)})"
            
    except Exception as e:
        # Catch any unexpected errors and return error message instead of crashing
        append_log(f"Error updating LLM provider: {str(e)}")
        return f"❌ Error setting LLM provider: {str(e)}\n(Provider remains unchanged)"


def show_price_trends_for_top_deals():
    """Generate price trend charts for current deals using Matplotlib."""
    top_deals = get_top_deals()
    price_mgr = get_price_history_manager()
    
    if not top_deals:
        return """<div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
        <strong>⚠️ No deals loaded</strong><br>
        Please fetch deals first using the 'Fetch Deals' button in the Deal Evaluations tab.
        </div>"""
    
    # Prepare data for chart generation
    charts_html = "<h3>📊 Price Trends for Deals (Last 90 Days)</h3>"
    
    for deal in top_deals[:3]:  # Show top 3 to avoid clutter
        title = deal['title']
        history = price_mgr.get_price_history(title, days=90)
        
        if history:
            try:
                chart_html = generate_price_trend_chart_img(history, title)
                charts_html += chart_html + "<br>"
            except Exception as e:
                charts_html += f"""<div style="padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; margin-bottom: 10px;">
                <strong>❌ Error generating chart for:</strong> {html.escape(title[:100])}<br>
                <small>{html.escape(str(e))}</small>
                </div>"""
        else:
            # Show info for deals without history
            charts_html += f"""<div style="padding: 15px; background-color: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 5px; margin-bottom: 15px;">
            <strong>📦 {html.escape(title[:100])}</strong><br>
            <small>Current Price: ${deal.get('price', 0):.2f} | Estimated Value: ${deal.get('est_value', 0):.2f}</small><br>
            <em style="color: #666;">ℹ️ No historical data yet. Price tracking has just started.</em>
            </div>"""
    
    if not any(price_mgr.get_price_history(deal['title'], days=90) for deal in top_deals[:3]):
        charts_html += """<div style="padding: 20px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; margin-top: 20px;">
        <strong>ℹ️ Building Price History</strong><br>
        Price tracking has just started for these deals. Charts will become available after multiple fetches over time.<br><br>
        <strong>Tips:</strong>
        <ul>
        <li>Fetch deals multiple times to build up history</li>
        <li>Check back in a few hours/days to see trends</li>
        <li>Current deal information is shown above</li>
        </ul>
        </div>"""
    
    return charts_html


def show_all_price_statistics():
    """Show price statistics table for all tracked deals."""
    price_mgr = get_price_history_manager()
    all_history = price_mgr.get_all_deals_history(days=90)
    
    if not all_history:
        return """<div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
        <strong>⚠️ No historical data available</strong><br>
        Start by fetching deals in the 'Deal Evaluations' tab. Price tracking will begin automatically.
        </div>"""
    
    # Get statistics for all deals
    stats_list = []
    for deal_id, deal_data in all_history.items():
        history = deal_data["price_history"]
        if history:
            prices = [entry["price"] for entry in history]
            
            # Calculate trend
            if len(prices) > 1:
                first_price = prices[0]
                current_price = prices[-1]
                price_change = ((current_price - first_price) / first_price) * 100
                if price_change < -5:
                    trend = "📉 Decreasing"
                elif price_change > 5:
                    trend = "📈 Increasing"
                else:
                    trend = "➡️ Stable"
            else:
                trend = "🆕 New"
            
            stats_list.append({
                "title": deal_data["title"],
                "current_price": prices[-1],
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_trend": trend,
                "data_points": len(prices)
            })
    
    if not stats_list:
        return """<div style="padding: 20px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px;">
        <strong>ℹ️ No statistics to display</strong><br>
        Price data is being collected but no valid statistics are available yet.
        </div>"""
    
    # Sort by current price (highest first)
    stats_list.sort(key=lambda x: x['current_price'], reverse=True)
    
    # Generate table with additional info
    result_html = f"<h3>📊 Price Statistics for {len(stats_list)} Tracked Deals</h3>"
    result_html += "<p><em>Showing data from the last 90 days</em></p>"
    
    try:
        fig = generate_summary_statistics_table(stats_list)
        # Include Plotly.js with CDN for tables
        table_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        result_html += table_html
        
        # Add summary statistics
        total_deals = len(stats_list)
        new_deals = sum(1 for s in stats_list if s['data_points'] == 1)
        
        result_html += f"""<div style="padding: 15px; background-color: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 5px; margin-top: 20px;">
        <strong>📈 Summary:</strong><br>
        • Total tracked deals: {total_deals}<br>
        • New deals (single data point): {new_deals}<br>
        • Deals with trend data: {total_deals - new_deals}<br>
        <br>
        <em>Tip: Fetch deals regularly to accumulate more data points and see clearer trends.</em>
        </div>"""
        
    except Exception as e:
        result_html += f"""<div style="padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
        <strong>❌ Error generating table:</strong> {html.escape(str(e))}
        </div>"""
    
    return result_html


def download_price_history():
    """Export all price history to CSV."""
    price_mgr = get_price_history_manager()
    csv_path = price_mgr.export_to_csv(deal_id=None, days=90)
    
    if csv_path:
        append_log(f"[{datetime.now()}] 📥 Price history exported to {csv_path}")
        return csv_path, "✅ Historical data exported successfully!"
    else:
        return None, "❌ No data available to export."


def search_product(keyword: str):
    """
    Search for products matching the keyword.
    Uses the same format and processing as show_top_deals for consistency.
    """
    if not keyword or not keyword.strip():
        return "", "⚠️ Please enter a search keyword."
    
    keyword = keyword.strip()
    append_log(f"[{datetime.now()}] 🔍 Searching for: {keyword}")
    
    # Search for matching deals (same as show_top_deals)
    raw_deals = search_deals_by_keyword(keyword)
    
    if not raw_deals:
        return "", f"⚠️ No results found for '{keyword}'. Try a different keyword."
    
    # Process results with AI evaluation (same logic as show_top_deals)
    price_mgr = get_price_history_manager()
    processed_results = []
    
    # Process all results (no limit, unlike show_top_deals which limits to DEALS_COUNT)
    for deal in raw_deals:
        processed_deal = _evaluate_single_deal(deal, price_mgr)
        if processed_deal:
            processed_results.append(processed_deal)
    
    # Store search results
    set_search_results(keyword, processed_results)
    
    # Format as HTML using the same format_deals_as_html function (same as show_top_deals)
    header = f"<h3>🔍 Search Results for: <strong>{html.escape(keyword)}</strong></h3>"
    header += f"<p>Found <strong>{len(processed_results)}</strong> matching deals</p><hr>"
    
    html_output = header + format_deals_as_html(processed_results)
    
    append_log(f"[{datetime.now()}] ✅ Search completed: {len(processed_results)} results")
    
    # Return same format as show_top_deals: (html_output, status_message)
    return html_output, f"✅ Found {len(processed_results)} deals matching '{keyword}'"


def search_rag_answer(query: str):
    """Answer questions based on search results."""
    keyword = get_current_search_keyword()
    if not keyword:
        return "⚠️ Please search for a product first."
    
    response = query_search_rag(query)
    append_log(f"[{datetime.now()}] Search RAG Q: {query} → {response[:100]}...")
    return response


def export_search_results():
    """Export search results to CSV dataset."""
    import csv
    from pathlib import Path
    
    results = get_search_results()
    keyword = get_current_search_keyword()
    
    if not results:
        return None, "⚠️ No search results to export. Please search first."
    
    # Create export file
    export_dir = Path("price_data")
    export_dir.mkdir(exist_ok=True)
    safe_keyword = keyword.replace(' ', '_').replace('/', '_')[:50] if keyword else "search"
    export_file = export_dir / f"search_results_{safe_keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        with open(export_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Search Keyword', 'Title', 'Price', 'Estimated Value', 'Discount %',
                'Link', 'Summary', 'AI Evaluation'
            ])
            
            for deal in results:
                writer.writerow([
                    keyword,
                    deal.get('title', ''),
                    f"${deal.get('price', 0):.2f}",
                    f"${deal.get('est_value', 0):.2f}",
                    f"{deal.get('discount', 0):.1f}%",
                    deal.get('link', ''),
                    deal.get('summary', '')[:200],  # Truncate long summaries
                    deal.get('gpt_response', '')[:300]  # Truncate long responses
                ])
        
        append_log(f"[{datetime.now()}] 📥 Search results exported: {export_file}")
        return str(export_file), f"✅ Exported {len(results)} results to CSV"
    
    except Exception as e:
        append_log(f"[{datetime.now()}] ❌ Export error: {e}")
        return None, f"❌ Export failed: {str(e)}"


def download_current_deals_with_trends():
    """Export current top deals with their price trends."""
    import csv
    from pathlib import Path
    
    top_deals = get_top_deals()
    price_mgr = get_price_history_manager()
    
    if not top_deals:
        return None, "Please fetch deals first."
    
    # Create export file
    export_dir = Path("price_data")
    export_dir.mkdir(exist_ok=True)
    export_file = export_dir / f"current_deals_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        with open(export_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Title', 'Current Price', 'Estimated Value', 'Discount %',
                'Min Price (90d)', 'Max Price (90d)', 'Avg Price (90d)', 
                'Price Trend', 'Data Points', 'Link'
            ])
            
            for deal in top_deals:
                title = deal['title']
                stats = price_mgr.get_price_statistics(title, days=90)
                
                writer.writerow([
                    title,
                    f"${deal['price']:.2f}",
                    f"${deal['est_value']:.2f}",
                    f"{deal['discount']:.1f}%",
                    f"${stats['min_price']:.2f}",
                    f"${stats['max_price']:.2f}",
                    f"${stats['avg_price']:.2f}",
                    stats['price_trend'],
                    stats['data_points'],
                    deal['link']
                ])
        
        append_log(f"[{datetime.now()}] 📥 Current deals analysis exported")
        return str(export_file), "✅ Current deals with trends exported successfully!"
    
    except Exception as e:
        append_log(f"[{datetime.now()}] ❌ Export error: {e}")
        return None, f"❌ Export failed: {str(e)}"


with gr.Blocks(theme=gr.themes.Soft(), css=LIGHT_THEME_CSS) as demo:
    gr.Markdown("## 🤖 Deal Agent v1.4 - Intelligent Deal Analysis System")

    with gr.Row():
        provider_selector = gr.Radio(
            choices=["OpenAI", "DeepSeek", "Gemini"],
            value=DEFAULT_PROVIDER,
            label="🤖 LLM Provider"
        )
        provider_status = gr.Markdown(f"✅ Using {DEFAULT_PROVIDER} (change above)")
        provider_selector.change(fn=update_llm_provider, inputs=provider_selector, outputs=provider_status)

    with gr.Tab("📊 Deal Evaluations"):
        gr.Markdown("### 📥 Fetch Deals from RSS Feeds")
        gr.Markdown("**Option 1:** Leave search empty to fetch all recent deals (up to 150)")
        gr.Markdown("**Option 2:** Enter a keyword to fetch only matching deals (e.g., 'NVIDIA', 'iPhone', 'laptop')")
        
        with gr.Row():
            with gr.Column(scale=3):
                fetch_search_input = gr.Textbox(
                    label="🔍 Search Keyword (Optional)", 
                    placeholder="e.g., NVIDIA, iPhone 15, gaming laptop (leave empty for all deals)",
                    lines=1
                )
            with gr.Column(scale=1):
                fetch_btn = gr.Button("🔍 Fetch Deals", variant="primary", size="lg")
        
        deal_output = gr.HTML(elem_id="deal_output")
        fetch_btn.click(fn=show_top_deals, inputs=fetch_search_input, outputs=deal_output)
        
        gr.Markdown("### 📈 Price Trends (Last 3 Months)")
        with gr.Row():
            with gr.Column(scale=1):
                show_trends_btn = gr.Button("📊 Show Price Trends", variant="secondary")
            with gr.Column(scale=1):
                show_stats_btn = gr.Button("📋 Show Statistics Table", variant="secondary")
        
        trend_output = gr.HTML(label="Price Trends")
        show_trends_btn.click(fn=show_price_trends_for_top_deals, outputs=trend_output)
        show_stats_btn.click(fn=show_all_price_statistics, outputs=trend_output)

    with gr.Tab("💬 Ask the Deal Expert (RAG)"):
        gr.Markdown("### 💡 Ask Questions About Current Deals")
        question = gr.Textbox(label="❓ Ask your question", lines=6, elem_id="rag_question", placeholder="e.g., What's the best deal? Which product has the highest discount?")
        answer = gr.Textbox(label="💬 Expert Answer", lines=10, elem_id="rag_answer")
        gr.Button("🚀 Submit").click(fn=rag_answer, inputs=question, outputs=answer)

    with gr.Tab("🔎 Product Search"):
        gr.Markdown("### 🔍 Search for Specific Products")
        gr.Markdown("Enter a product name or keyword to search across all RSS feeds (e.g., 'GTX560', 'NVIDIA', 'iPhone 15')")
        
        with gr.Row():
            with gr.Column(scale=3):
                search_input = gr.Textbox(label="🔍 Search Keyword", placeholder="e.g., GTX560, NVIDIA, iPhone 15", lines=1)
            with gr.Column(scale=1):
                search_btn = gr.Button("🔎 Search", variant="primary", size="lg")
        
        search_status = gr.Textbox(label="📊 Status", lines=1, interactive=False)
        search_output = gr.HTML(label="📋 Search Results")
        search_btn.click(fn=search_product, inputs=search_input, outputs=[search_output, search_status])
        
        gr.Markdown("---")
        gr.Markdown("### 🤖 Ask AI About Search Results")
        search_question = gr.Textbox(label="❓ Ask about the search results", lines=4, placeholder="e.g., What's the average price? Which deal is the best value?")
        search_answer = gr.Textbox(label="💬 AI Answer", lines=8)
        gr.Button("🚀 Submit Question").click(fn=search_rag_answer, inputs=search_question, outputs=search_answer)
        
        gr.Markdown("---")
        gr.Markdown("### 📊 3D Visualization & Export")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**📈 Generate 3D Visualization**")
                gr.Markdown("_Create 3D cluster plot of search results_")
                search_plot_file = gr.File(label="📁 Search Results 3D Plot")
                gr.Button("📈 Generate 3D Plot").click(
                    fn=lambda: generate_visualization(use_search_results=True),
                    outputs=search_plot_file
                )
            with gr.Column():
                gr.Markdown("**💾 Export Dataset**")
                gr.Markdown("_Export search results as CSV_")
                export_search_status = gr.Textbox(label="📊 Status", lines=1)
                export_search_file = gr.File(label="📁 Search Results CSV")
                gr.Button("💾 Export Search Results").click(
                    fn=export_search_results,
                    outputs=[export_search_file, export_search_status]
                )

    with gr.Tab("📥 Download Analysis & Data"):
        gr.Markdown("### 📈 3D Visualization")
        file_output_3d = gr.File(label="📁 3D Cluster Plot")
        gr.Button("📈 Generate & Download 3D Plot").click(fn=lambda: generate_visualization(use_search_results=False), outputs=file_output_3d)
        
        gr.Markdown("---")
        gr.Markdown("### 📊 Historical Price Data")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**📥 Download Complete Price History (90 days)**")
                gr.Markdown("_All tracked deals with timestamp, price, and discount data_")
                download_all_status = gr.Textbox(label="📊 Status", lines=1)
                download_all_file = gr.File(label="📁 Complete History CSV")
                gr.Button("📥 Download All Historical Data").click(
                    fn=download_price_history, 
                    outputs=[download_all_file, download_all_status]
                )
            
            with gr.Column():
                gr.Markdown("**📊 Download Current Deals with Trends**")
                gr.Markdown("_Deals with min/max/avg prices and trend analysis_")
                download_current_status = gr.Textbox(label="📊 Status", lines=1)
                download_current_file = gr.File(label="📁 Current Deals Analysis CSV")
                gr.Button("📊 Download Current Deals Analysis").click(
                    fn=download_current_deals_with_trends,
                    outputs=[download_current_file, download_current_status]
                )

    with gr.Tab("📜 Logs"):
        logs_text = gr.Textbox(lines=20, label="📜 Application Logs")
        gr.Button("🔄 Refresh Logs").click(fn=read_logs, outputs=logs_text)


demo.launch(
    server_name=Config.SERVER_HOST,
    server_port=Config.SERVER_PORT,
    show_error=Config.SHOW_ERROR,
    share=False,
    inbrowser=False
)