"""
P2: Load Historical Data to RAG Knowledge Base
Loads high-quality historical deals from CSV/JSON into FAISS vector database
for enhanced RAG Q&A capabilities.

Usage:
    python load_history_to_rag.py [--csv path/to/history.csv] [--min-discount 20]
"""
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from shared_data import append_log
from rag_faiss import build_top5_index, query_deals_rag
from price_history import get_price_history_manager


def load_deals_from_csv(csv_path: str, min_discount: float = 20.0) -> List[Dict]:
    """
    Load high-quality deals from CSV file.
    
    Args:
        csv_path: Path to CSV file
        min_discount: Minimum discount percentage to include (default: 20%)
    
    Returns:
        List of deal dictionaries with title, price, discount, etc.
    """
    deals = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            delimiter = ',' if sample.count(',') > sample.count(';') else ';'
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                try:
                    # Extract relevant fields (handle different CSV formats)
                    title = row.get('Title', row.get('title', row.get('Product', ''))).strip()
                    if not title:
                        continue
                    
                    # Extract price (handle different formats)
                    price_str = str(row.get('Price', row.get('price', row.get('Current Price', '0'))))
                    price = float(price_str.replace('$', '').replace(',', '').strip() or 0)
                    
                    # Extract discount
                    discount_str = str(row.get('Discount', row.get('discount', row.get('Discount_Percent', '0'))))
                    discount = float(discount_str.replace('%', '').strip() or 0)
                    
                    # Extract link if available
                    link = row.get('Link', row.get('link', row.get('URL', ''))).strip()
                    
                    # Extract timestamp if available
                    timestamp = row.get('Timestamp', row.get('timestamp', row.get('Date', ''))).strip()
                    
                    # Only include deals with good discounts
                    if discount >= min_discount and price > 0:
                        deals.append({
                            'title': title,
                            'price': price,
                            'discount': discount,
                            'link': link,
                            'timestamp': timestamp,
                            'description': f"{title} at ${price:.2f} with {discount:.1f}% discount"
                        })
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue
        
        append_log(f"[{datetime.now()}] ✅ Loaded {len(deals)} high-quality deals from CSV: {csv_path}")
        
    except FileNotFoundError:
        append_log(f"[{datetime.now()}] ❌ CSV file not found: {csv_path}")
    except Exception as e:
        append_log(f"[{datetime.now()}] ❌ Error loading CSV: {str(e)}")
    
    return deals


def load_deals_from_price_history(min_discount: float = 20.0, days: int = 90) -> List[Dict]:
    """
    Load high-quality deals from price_history.json.
    
    Args:
        min_discount: Minimum discount percentage to include
        days: Number of days of history to include
    
    Returns:
        List of deal dictionaries
    """
    deals = []
    price_mgr = get_price_history_manager()
    
    try:
        # Get all deals from price history
        all_deals = price_mgr.data
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for deal_id, deal_data in all_deals.items():
            title = deal_data.get('title', '')
            if not title:
                continue
            
            # Get the best price entry (highest discount)
            price_history = deal_data.get('price_history', [])
            if not price_history:
                continue
            
            # Find entry with highest discount
            best_entry = max(price_history, key=lambda x: x.get('discount', 0))
            
            discount = best_entry.get('discount', 0)
            price = best_entry.get('price', 0)
            
            # Check if within time range
            entry_timestamp = best_entry.get('timestamp', '')
            if entry_timestamp:
                try:
                    entry_date = datetime.fromisoformat(entry_timestamp)
                    if entry_date < cutoff_date:
                        continue
                except:
                    pass
            
            # Only include deals with good discounts
            if discount >= min_discount and price > 0:
                deals.append({
                    'title': title,
                    'price': price,
                    'discount': discount,
                    'link': deal_data.get('link', ''),
                    'timestamp': entry_timestamp,
                    'description': f"{title} at ${price:.2f} with {discount:.1f}% discount"
                })
        
        append_log(f"[{datetime.now()}] ✅ Loaded {len(deals)} high-quality deals from price history")
        
    except Exception as e:
        append_log(f"[{datetime.now()}] ❌ Error loading price history: {str(e)}")
    
    return deals


def add_deals_to_rag_knowledge_base(deals: List[Dict]):
    """
    Add deals to RAG knowledge base by updating top_deals.
    This will trigger FAISS index rebuild on next RAG query.
    
    Args:
        deals: List of deal dictionaries
    """
    from shared_data import set_top_deals
    
    if not deals:
        append_log(f"[{datetime.now()}] ⚠️ No deals to add to RAG knowledge base")
        return
    
    # Format deals for RAG (same structure as show_top_deals)
    formatted_deals = []
    for deal in deals:
        formatted_deals.append({
            'title': deal['title'],
            'price': deal['price'],
            'link': deal.get('link', ''),
            'gpt_response': f"Historical deal with {deal['discount']:.1f}% discount",
            'est_value': deal['price'] / (1 - deal['discount'] / 100) if deal['discount'] < 100 else deal['price'] * 1.5,
            'discount': deal['discount'],
            'description': deal.get('description', f"{deal['title']} at ${deal['price']:.2f}")
        })
    
    # Update top_deals (this will be used by RAG)
    set_top_deals(formatted_deals)
    
    # Force rebuild of FAISS index
    from rag_faiss import build_top5_index
    build_top5_index()  # Rebuild with new data
    
    append_log(f"[{datetime.now()}] ✅ Added {len(formatted_deals)} deals to RAG knowledge base")


def main():
    """Main function to load historical data to RAG."""
    parser = argparse.ArgumentParser(description='Load historical deals to RAG knowledge base')
    parser.add_argument('--csv', type=str, help='Path to CSV file with historical deals')
    parser.add_argument('--min-discount', type=float, default=20.0, 
                       help='Minimum discount percentage to include (default: 20.0)')
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days of history to include from price_history.json (default: 90)')
    parser.add_argument('--use-price-history', action='store_true',
                       help='Also load from price_history.json')
    
    args = parser.parse_args()
    
    all_deals = []
    
    # Load from CSV if provided
    if args.csv:
        csv_deals = load_deals_from_csv(args.csv, args.min_discount)
        all_deals.extend(csv_deals)
    
    # Load from price history if requested
    if args.use_price_history or not args.csv:
        history_deals = load_deals_from_price_history(args.min_discount, args.days)
        all_deals.extend(history_deals)
    
    # Remove duplicates (same title)
    seen_titles = set()
    unique_deals = []
    for deal in all_deals:
        title_key = deal['title'].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_deals.append(deal)
    
    print(f"📊 Found {len(unique_deals)} unique high-quality deals (discount >= {args.min_discount}%)")
    
    if unique_deals:
        # Add to RAG knowledge base
        add_deals_to_rag_knowledge_base(unique_deals)
        print(f"✅ Successfully loaded {len(unique_deals)} deals to RAG knowledge base")
        print("💡 You can now ask questions like:")
        print("   - 'What was the lowest price for Dell monitors in the past month?'")
        print("   - 'Show me deals with more than 30% discount'")
        print("   - 'What are the best deals on gaming laptops?'")
    else:
        print("⚠️ No deals found matching criteria")


if __name__ == "__main__":
    main()

