import feedparser
import random
import re
import socket
import urllib.request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError


def load_rss_sources(file_path="rss_sources.txt"):
    """
    Load RSS feed URLs from file.
    Supports comments (lines starting with #) and empty lines.
    
    Args:
        file_path: Path to RSS sources file
    
    Returns:
        list: List of valid RSS feed URLs
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sources = []
            for line in f.readlines():
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    sources.append(line)
            # Validate URLs
            valid_sources = [url for url in sources if _is_valid_url(url)]
            print(f"✅ Loaded {len(valid_sources)} valid RSS sources from {len(sources)} total entries")
            return valid_sources
    except FileNotFoundError:
        print(f"⚠️ {file_path} not found.")
        return []
    except Exception as e:
        print(f"❌ Error loading RSS sources: {e}")
        return []


def _is_valid_url(url: str) -> bool:
    """Validate URL format for security."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False


def _parse_rss_feed(url: str, timeout: int) -> tuple:
    """
    Parse an RSS feed with timeout protection.
    
    Args:
        url: RSS feed URL
        timeout: Timeout in seconds
    
    Returns:
        tuple: (feed object, error_message) - error_message is None if successful
    """
    try:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        
        try:
            feed = feedparser.parse(
                url,
                request_headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        finally:
            socket.setdefaulttimeout(old_timeout)
        
        # Check for parsing errors
        if hasattr(feed, 'bozo') and feed.bozo:
            error_msg = str(feed.bozo_exception) if hasattr(feed, 'bozo_exception') else 'Parse error'
            return None, error_msg
        
        return feed, None
    except Exception as e:
        return None, str(e)


def _extract_entry_data(entry) -> dict:
    """
    Extract data from an RSS entry.
    
    Args:
        entry: RSS feed entry
    
    Returns:
        dict: {'title': str, 'link': str, 'summary': str} or None if invalid
    """
    try:
        title = getattr(entry, "title", "").strip()
        if not title:
            return None
        
        link = getattr(entry, "link", "").strip()
        summary = entry.get("summary", "") if isinstance(entry, dict) else getattr(entry, "summary", "")
        
        return {
            'title': title,
            'link': link if _is_valid_url(link) else "#",
            'summary': summary[:200] if summary else ""
        }
    except Exception:
        return None


def extract_price(text: str) -> float:
    """
    Extract price from text using regex.
    
    Args:
        text: Text containing price information
    
    Returns:
        float: Extracted price or random fallback value
    """
    if not text:
        return round(random.uniform(5, 50), 2)
    
    match = re.search(r"\$\s?(\d+(\.\d{1,2})?)", text)
    if match:
        try:
            price = float(match.group(1))
            # Sanity check: price should be reasonable
            if 0 < price < 100000:
                return price
        except (ValueError, AttributeError):
            pass
    
    return round(random.uniform(5, 50), 2)  # fallback dummy price


def fetch_deals_rss(timeout: int = None, max_sources: int = None) -> list:
    """
    Fetch deals from RSS sources with error handling and timeout.
    
    Args:
        timeout: Timeout in seconds for each feed request (default from Config)
        max_sources: Maximum number of sources to process (default from Config)
    
    Returns:
        list: List of deal dictionaries with title, price, and link
    """
    from config import Config
    
    if timeout is None:
        timeout = Config.RSS_TIMEOUT
    if max_sources is None:
        max_sources = Config.RSS_MAX_SOURCES
    
    sources = load_rss_sources()
    all_deals = []
    
    if not sources:
        print("⚠️ No valid RSS sources found.")
        return []

    # Limit number of sources to process to prevent excessive wait times
    original_count = len(sources)
    sources = sources[:max_sources]
    if original_count > max_sources:
        print(f"📡 Processing {len(sources)} RSS sources (limited from {original_count} for performance)...")
    else:
        print(f"📡 Processing {len(sources)} RSS sources...")

    successful_sources = 0
    failed_sources = 0
    
    for url in sources:
        feed, error_msg = _parse_rss_feed(url, timeout)
        
        if feed is None:
            if error_msg:
                print(f"⚠️ Feed parse warning for {url}: {error_msg}")
            failed_sources += 1
            continue
        
        entries = getattr(feed, "entries", [])
        if not entries:
            print(f"⚠️ No entries found in {url}")
            failed_sources += 1
            continue
        
        entry_count = 0
        for entry in entries:
            entry_data = _extract_entry_data(entry)
            if entry_data is None:
                continue
            
            try:
                price = extract_price(f"{entry_data['title']} {entry_data['summary']}")
                all_deals.append({
                    "title": entry_data['title'],
                    "price": price,
                    "link": entry_data['link'],
                    "summary": entry_data['summary']
                })
                entry_count += 1
            except Exception as e:
                print(f"⚠️ Error processing entry from {url}: {e}")
                continue
        
        if entry_count > 0:
            successful_sources += 1
            print(f"✅ Fetched {entry_count} deals from {url}")
        else:
            failed_sources += 1

    print(f"📊 RSS Fetch Summary: {successful_sources} successful, {failed_sources} failed, {len(all_deals)} total deals")
    return all_deals  # return all parsed entries (no limit)


def search_deals_by_keyword(keyword: str, timeout: int = None, max_sources: int = None) -> list:
    """
    Search for deals matching a specific keyword with improved matching and timeout.
    
    Args:
        keyword: Search keyword (e.g., "GTX560", "NVIDIA", "nvidia card")
        timeout: Timeout in seconds for each feed request (default from Config)
        max_sources: Maximum number of sources to process (default from Config)
    
    Returns:
        list: List of matching deal dictionaries
    """
    from config import Config
    
    if timeout is None:
        timeout = Config.RSS_TIMEOUT
    if max_sources is None:
        max_sources = Config.RSS_MAX_SOURCES
    
    if not keyword or not keyword.strip():
        return []
    
    keyword_original = keyword.strip()
    keyword_lower = keyword_original.lower()
    
    # Split keyword into individual words for better matching
    keyword_words = [w.strip() for w in keyword_lower.split() if w.strip()]
    
    sources = load_rss_sources()
    matching_deals = []
    
    if not sources:
        print("⚠️ No valid RSS sources found.")
        return []
    
    # Limit number of sources to process to prevent excessive wait times
    original_count = len(sources)
    sources = sources[:max_sources]
    if original_count > max_sources:
        print(f"🔍 Searching for: '{keyword_original}' across {len(sources)} RSS feeds (limited from {original_count} for performance)...")
    else:
        print(f"🔍 Searching for: '{keyword_original}' across {len(sources)} RSS feeds...")
    
    successful_sources = 0
    failed_sources = 0
    
    for url in sources:
        feed, error_msg = _parse_rss_feed(url, timeout)
        
        if feed is None:
            if error_msg:
                print(f"⚠️ Feed parse warning for {url}: {error_msg}")
            failed_sources += 1
            continue
        
        entries = getattr(feed, "entries", [])
        if not entries:
            failed_sources += 1
            continue
        
        matches_from_source = 0
        for entry in entries:
            entry_data = _extract_entry_data(entry)
            if entry_data is None:
                continue
            
            try:
                # Improved matching: check both exact phrase and individual words
                search_text = f"{entry_data['title']} {entry_data['summary']}".lower()
                
                # Match if:
                # 1. Exact phrase match (e.g., "nvidia card" in title)
                # 2. All words present (e.g., both "nvidia" and "card" in title)
                # 3. At least 50% of words match for multi-word searches
                matches = False
                
                if keyword_lower in search_text:
                    # Exact phrase match
                    matches = True
                elif len(keyword_words) > 1:
                    # Multi-word: check if at least 50% of words match
                    matched_words = sum(1 for word in keyword_words if word in search_text)
                    if matched_words >= max(1, len(keyword_words) * 0.5):
                        matches = True
                elif len(keyword_words) == 1:
                    # Single word: must be present
                    matches = keyword_words[0] in search_text
                
                if matches:
                    price = extract_price(f"{entry_data['title']} {entry_data['summary']}")
                    matching_deals.append({
                        "title": entry_data['title'],
                        "price": price,
                        "link": entry_data['link'],
                        "summary": entry_data['summary']
                    })
                    matches_from_source += 1
            except Exception as e:
                print(f"⚠️ Error processing entry from {url}: {e}")
                continue
        
        if matches_from_source > 0:
            successful_sources += 1
            print(f"  ✓ {url}: {matches_from_source} matches")
        else:
            failed_sources += 1
    
    print(f"✅ Search complete: Found {len(matching_deals)} deals matching '{keyword_original}' from {successful_sources} sources")
    return matching_deals