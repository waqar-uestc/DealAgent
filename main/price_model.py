"""
Enhanced Price Prediction Model with Category-Aware Logic and LLM Assistance.
Implements layered valuation: Rule-based category detection + ML model + LLM fallback.
"""
from sklearn.ensemble import RandomForestRegressor
from model_manager import ModelManager
from shared_data import get_llm_provider
from gpt_evaluator import _get_provider
import re

# Lazy-loaded regressor
_regressor = None
_llm_price_cache = {}  # Cache LLM price estimates to reduce API calls


def _get_embed_model():
    """Get shared sentence model from ModelManager."""
    return ModelManager.get_sentence_model()


def _get_regressor():
    """Get or create the ML regressor model."""
    global _regressor
    if _regressor is None:
        model = _get_embed_model()
        X_train = model.encode([
            "Wireless Mouse",
            "Gaming Laptop",
            "Bluetooth Speaker",
            "USB-C Cable",
            "Smartphone",
            "Noise Cancelling Headphones",
            "External SSD",
            "Office Chair",
            "Mechanical Keyboard",
            "4K Monitor",
        ])
        y_train = [15, 900, 45, 10, 500, 180, 100, 120, 70, 300]
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train, y_train)
        _regressor = reg
    return _regressor


def _get_category_base_price(title: str) -> float:
    """
    Category-Aware Rule-Based Baseline Pricing (Method A).
    Returns a base price estimate based on product category keywords.
    """
    title_lower = title.lower()
    
    # High-end electronics and computers
    if any(kw in title_lower for kw in ["rtx", "gtx", "gpu", "graphics card", "video card", 
                                         "gaming pc", "gaming computer", "workstation", 
                                         "high-end", "premium", "pro", "professional"]):
        return 1500.0  # High-end PC components
    
    if any(kw in title_lower for kw in ["laptop", "notebook", "pc", "computer", "desktop"]):
        return 800.0  # General computers
    
    if any(kw in title_lower for kw in ["monitor", "display", "screen"]):
        return 300.0  # Monitors
    
    if any(kw in title_lower for kw in ["iphone", "smartphone", "phone", "mobile"]):
        return 500.0  # Smartphones
    
    if any(kw in title_lower for kw in ["tablet", "ipad"]):
        return 400.0  # Tablets
    
    # Mid-range electronics
    if any(kw in title_lower for kw in ["headphones", "earphones", "earbuds", "speaker", "audio"]):
        return 100.0  # Audio devices
    
    if any(kw in title_lower for kw in ["keyboard", "mouse", "webcam", "microphone"]):
        return 50.0  # Peripherals
    
    if any(kw in title_lower for kw in ["ssd", "hard drive", "storage", "usb", "cable", "charger"]):
        return 30.0  # Storage and accessories
    
    # Furniture and office
    if any(kw in title_lower for kw in ["chair", "desk", "table", "furniture"]):
        return 150.0  # Furniture
    
    # Food and consumables
    if any(kw in title_lower for kw in ["cookie", "food", "snack", "beverage", "drink", 
                                         "grocery", "candy", "chocolate"]):
        return 5.0  # Food items
    
    # Clothing and apparel
    if any(kw in title_lower for kw in ["shirt", "pants", "shoes", "clothing", "apparel", "jacket"]):
        return 50.0  # Clothing
    
    # Default for unknown categories
    return 200.0  # Default baseline


def _llm_estimate_price(title: str) -> float:
    """
    LLM-Assisted Price Estimation (Method B - Recommended).
    Uses LLM to extract or estimate MSRP from product title.
    """
    # Check cache first
    cache_key = title.lower().strip()
    if cache_key in _llm_price_cache:
        return _llm_price_cache[cache_key]
    
    try:
        provider = _get_provider(get_llm_provider())
        if not provider or not provider.is_available():
            return None  # Fallback to category-based
        
        # Use LLM to estimate price
        prompt = f"""Analyze the product: '{title}'

What is the typical market price (MSRP) for this specific item? Consider:
- Product category and brand
- Specifications and features mentioned
- Market standards for similar products

Output ONLY a single number representing the estimated average market price in USD. 
Do not include currency symbols, commas, or any other text. Just the number.

Example outputs:
- For "RTX 5080 Gaming PC": 2500
- For "Wireless Mouse": 25
- For "Chocolate Cookies 12-pack": 8"""

        # Use answer_question method with minimal context
        response = provider.answer_question(
            question=prompt,
            context="You are a price estimation expert."
        )
        
        # Extract number from response
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            price = float(numbers[0])
            # Sanity check: reasonable price range
            if 0.1 <= price <= 100000:
                _llm_price_cache[cache_key] = price
                return price
        
    except Exception as e:
        # If LLM fails, return None to fallback
        pass
    
    return None


def predict_value(title: str, use_llm: bool = True) -> float:
    """
    Enhanced price prediction with layered approach:
    1. Try LLM estimation (if enabled and available)
    2. Use category-aware baseline
    3. Apply ML model adjustment
    4. Final sanity check
    
    Args:
        title: Product title
        use_llm: Whether to attempt LLM-based estimation (default: True)
    
    Returns:
        float: Estimated market value
    """
    if not title or not title.strip():
        return 200.0  # Default fallback
    
    title = title.strip()
    
    # Step 1: Try LLM estimation (if enabled)
    llm_price = None
    if use_llm:
        llm_price = _llm_estimate_price(title)
    
    # Step 2: Get category-based baseline
    category_base = _get_category_base_price(title)
    
    # Step 3: Get ML model prediction
    try:
        model = _get_embed_model()
        regressor = _get_regressor()
        embedding = model.encode([title])
        ml_prediction = float(regressor.predict(embedding)[0])
    except Exception:
        ml_prediction = category_base
    
    # Step 4: Combine estimates with weighted average
    # Priority: LLM > Category Baseline > ML Model
    if llm_price is not None:
        # LLM is most reliable, use it as primary with slight ML adjustment
        final_price = 0.8 * llm_price + 0.2 * ml_prediction
    else:
        # No LLM, combine category baseline and ML model
        # Category baseline is more reliable for extreme cases
        if category_base < 50 or category_base > 1000:
            # Extreme categories: trust category baseline more
            final_price = 0.7 * category_base + 0.3 * ml_prediction
        else:
            # Normal range: balance both
            final_price = 0.5 * category_base + 0.5 * ml_prediction
    
    # Step 5: Final sanity checks
    # Enforce minimum price
    if final_price < 0.5:
        final_price = 0.5
    
    # For very low prices (< $5), cap the estimate to prevent false discounts
    if final_price < 5.0:
        final_price = min(final_price, 20.0)  # Cap at $20 for low-price items
    
    return round(final_price, 2)