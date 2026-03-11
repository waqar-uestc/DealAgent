"""
Price History Management Module
Tracks and manages historical price data for deals
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class PriceHistory:
    """Manage price history data with JSON storage."""
    
    def __init__(self, data_dir: str = "price_data"):
        """
        Initialize price history manager.
        
        Args:
            data_dir: Directory to store price history data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "price_history.json"
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load price history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading price history: {e}")
                return {}
        return {}
    
    def _save_data(self):
        """Save price history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving price history: {e}")
    
    def _generate_deal_id(self, title: str) -> str:
        """
        Generate a unique ID for a deal based on title.
        
        Args:
            title: Deal title
        
        Returns:
            str: Normalized deal ID
        """
        # Normalize title for consistent matching
        return title.lower().strip().replace(' ', '_')[:100]
    
    def add_price_entry(self, title: str, price: float, link: str = "", 
                       estimated_value: float = 0.0, discount: float = 0.0):
        """
        Add a price entry for a deal.
        
        Args:
            title: Deal title
            price: Current price
            link: Deal link
            estimated_value: Estimated value
            discount: Discount percentage
        """
        deal_id = self._generate_deal_id(title)
        timestamp = datetime.now().isoformat()
        
        if deal_id not in self.data:
            self.data[deal_id] = {
                "title": title,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "link": link,
                "price_history": []
            }
        
        # Update last seen and link
        self.data[deal_id]["last_seen"] = timestamp
        if link:
            self.data[deal_id]["link"] = link
        
        # Add price entry
        self.data[deal_id]["price_history"].append({
            "timestamp": timestamp,
            "price": price,
            "estimated_value": estimated_value,
            "discount": discount
        })
        
        self._save_data()
    
    def get_price_history(self, title: str, days: int = 90) -> List[Dict]:
        """
        Get price history for a deal within specified days.
        
        Args:
            title: Deal title
            days: Number of days to look back (default 90)
        
        Returns:
            List of price entries
        """
        deal_id = self._generate_deal_id(title)
        if deal_id not in self.data:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        history = self.data[deal_id]["price_history"]
        
        # Filter by date
        filtered = [
            entry for entry in history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        
        return filtered
    
    def get_all_deals_history(self, days: int = 90) -> Dict[str, Dict]:
        """
        Get price history for all deals.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dictionary of deal histories
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        result = {}
        
        for deal_id, deal_data in self.data.items():
            history = [
                entry for entry in deal_data["price_history"]
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
            ]
            
            if history:
                result[deal_id] = {
                    "title": deal_data["title"],
                    "link": deal_data.get("link", ""),
                    "first_seen": deal_data["first_seen"],
                    "last_seen": deal_data["last_seen"],
                    "price_history": history
                }
        
        return result
    
    def get_price_statistics(self, title: str, days: int = 90) -> Dict:
        """
        Get price statistics for a deal.
        
        Args:
            title: Deal title
            days: Number of days to analyze
        
        Returns:
            Dictionary with statistics (min, max, avg, current, trend)
        """
        history = self.get_price_history(title, days)
        
        if not history:
            return {
                "min_price": 0,
                "max_price": 0,
                "avg_price": 0,
                "current_price": 0,
                "price_trend": "No data",
                "data_points": 0
            }
        
        prices = [entry["price"] for entry in history]
        current_price = prices[-1]
        
        # Calculate trend
        if len(prices) > 1:
            first_price = prices[0]
            price_change = ((current_price - first_price) / first_price) * 100
            if price_change < -5:
                trend = "Decreasing"
            elif price_change > 5:
                trend = "Increasing"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "current_price": current_price,
            "price_trend": trend,
            "data_points": len(prices),
            "history": history
        }
    
    def export_to_csv(self, deal_id: str = None, days: int = 90) -> Optional[str]:
        """
        Export price history to CSV file.
        
        Args:
            deal_id: Specific deal ID to export (None for all deals)
            days: Number of days to include
        
        Returns:
            Path to exported CSV file
        """
        try:
            if deal_id:
                # Export single deal
                if deal_id not in self.data:
                    return None
                
                deal_data = self.data[deal_id]
                history = self.get_price_history(deal_data["title"], days)
                
                if not history:
                    return None
                
                # Create DataFrame
                df = pd.DataFrame(history)
                df['title'] = deal_data['title']
                df['link'] = deal_data.get('link', '')
                
                filename = self.data_dir / f"{deal_id}_history.csv"
                df.to_csv(filename, index=False)
                
                return str(filename)
            else:
                # Export all deals
                all_history = self.get_all_deals_history(days)
                
                if not all_history:
                    return None
                
                # Flatten all data
                rows = []
                for deal_id, deal_data in all_history.items():
                    for entry in deal_data["price_history"]:
                        rows.append({
                            "deal_id": deal_id,
                            "title": deal_data["title"],
                            "link": deal_data.get("link", ""),
                            "timestamp": entry["timestamp"],
                            "price": entry["price"],
                            "estimated_value": entry.get("estimated_value", 0),
                            "discount": entry.get("discount", 0)
                        })
                
                df = pd.DataFrame(rows)
                filename = self.data_dir / "all_deals_history.csv"
                df.to_csv(filename, index=False)
                
                return str(filename)
                
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None
    
    def cleanup_old_data(self, days: int = 180):
        """
        Remove price entries older than specified days.
        
        Args:
            days: Keep data newer than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for deal_id in list(self.data.keys()):
            # Filter price history
            self.data[deal_id]["price_history"] = [
                entry for entry in self.data[deal_id]["price_history"]
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
            ]
            
            # Remove deals with no recent history
            if not self.data[deal_id]["price_history"]:
                del self.data[deal_id]
        
        self._save_data()


# Global price history instance
_price_history = None


def get_price_history_manager() -> PriceHistory:
    """Get or create global price history manager."""
    global _price_history
    if _price_history is None:
        _price_history = PriceHistory()
    return _price_history

