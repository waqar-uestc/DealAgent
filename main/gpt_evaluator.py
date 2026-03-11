from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from shared_data import get_llm_provider
from config import Config

# Load .env for local development
load_dotenv()

# Lazy provider instances
_providers = {}


class LLMProvider(ABC):    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def evaluate_deal(self, title: str, price: float) -> str:
        pass
    
    @abstractmethod
    def answer_question(self, question: str, context: str) -> str:
        pass


class OpenAIProvider(LLMProvider):
    
    def __init__(self):
        self._client = None
        self._init_error = None
        try:
            self._initialize()
        except Exception as e:
            # Catch any initialization errors to prevent app crash
            self._client = None
            self._init_error = str(e)
    
    def _initialize(self):
        if self._client is not None:
            return
        
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except Exception as e:
            # Log error but don't crash - just set client to None
            self._client = None
            # Store error for debugging (optional)
            self._init_error = str(e)
    
    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._client is not None
        except Exception:
            # If initialization fails, return False instead of crashing
            return False
    
    def evaluate_deal(self, title: str, price: float) -> str:
        if not self.is_available():
            return "OpenAI error: OPENAI_API_KEY not configured."
        
        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful shopping assistant."},
                    {"role": "user", "content": f"Evaluate this deal: '{title}' priced at ${price:.2f}. Is it a good value? Answer briefly."},
                ],
                timeout=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {str(e)}"

    def answer_question(self, question: str, context: str) -> str:
        if not self.is_available():
            return "OpenAI error: OPENAI_API_KEY not configured."

        prompt = f"""You are a deal expert. Based on the following deals:
{context}

Answer this user question concisely:
{question}
"""

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful deal assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {str(e)}"


class GeminiProvider(LLMProvider):
    
    def __init__(self):
        self._model = None
        self._init_error = None
        try:
            self._initialize()
        except Exception as e:
            # Catch any initialization errors to prevent app crash
            self._model = None
            self._init_error = str(e)
    
    def _initialize(self):
        if self._model is not None:
            return
        
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            # Use timeout to prevent hanging on network issues
            self._model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            # Log error but don't crash - just set model to None
            self._model = None
            # Store error for debugging (optional)
            self._init_error = str(e)
    
    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._model is not None
        except Exception:
            # If initialization fails, return False instead of crashing
            return False
    
    def evaluate_deal(self, title: str, price: float) -> str:
        """Evaluate deal using Gemini."""
        if not self.is_available():
            return "Gemini error: GEMINI_API_KEY not configured."
        
        try:
            prompt = f"Evaluate this deal: '{title}' priced at ${price:.2f}. Is it a good value? Answer briefly."
            response = self._model.generate_content(prompt)
            return (getattr(response, "text", "").strip() or "No response from Gemini.")
        except Exception as e:
            return f"Gemini error: {str(e)}"
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer question using Gemini."""
        if not self.is_available():
            return "Gemini error: GEMINI_API_KEY not configured."
        
        prompt = f"""You are a deal expert. Based on the following deals:
{context}

Answer this user question concisely:
{question}
"""
        
        try:
            response = self._model.generate_content(prompt)
            return (getattr(response, "text", "").strip() or "No response from Gemini.")
        except Exception as e:
            return f"Gemini error: {str(e)}"


class DeepSeekProvider(LLMProvider):
    """DeepSeek AI provider implementation."""
    
    def __init__(self):
        self._client = None
        self._init_error = None
        try:
            self._initialize()
        except Exception as e:
            # Catch any initialization errors to prevent app crash
            self._client = None
            self._init_error = str(e)
    
    def _initialize(self):
        """Lazy initialization of DeepSeek client."""
        if self._client is not None:
            return
        
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return
        
        try:
            from openai import OpenAI
            # DeepSeek uses OpenAI-compatible API
            self._client = OpenAI(
                api_key=key,
                base_url=Config.DEEPSEEK_API_BASE
            )
        except Exception as e:
            # Log error but don't crash - just set client to None
            self._client = None
            # Store error for debugging (optional)
            self._init_error = str(e)
    
    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._client is not None
        except Exception:
            # If initialization fails, return False instead of crashing
            return False
    
    def evaluate_deal(self, title: str, price: float) -> str:
        if not self.is_available():
            return "DeepSeek error: DEEPSEEK_API_KEY not configured."
        
        try:
            response = self._client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful shopping assistant."},
                    {"role": "user", "content": f"Evaluate this deal: '{title}' priced at ${price:.2f}. Is it a good value? Answer briefly."},
                ],
                timeout=Config.LLM_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"DeepSeek error: {str(e)}"
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer question using DeepSeek."""
        if not self.is_available():
            return "DeepSeek error: DEEPSEEK_API_KEY not configured."
        
        prompt = f"""You are a deal expert. Based on the following deals:
{context}

Answer this user question concisely:
{question}
"""
        
        try:
            response = self._client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful deal assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=Config.LLM_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"DeepSeek error: {str(e)}"


def _get_provider(name: str) -> LLMProvider:
    """Get or create a provider instance with error handling."""
    name = name.lower().strip()
    
    if name not in _providers:
        try:
            if name == "openai":
                _providers[name] = OpenAIProvider()
            elif name == "gemini":
                _providers[name] = GeminiProvider()
            elif name == "deepseek":
                _providers[name] = DeepSeekProvider()
            else:
                _providers[name] = OpenAIProvider()  # default
        except Exception as e:
            # If provider creation fails, return a safe fallback
            # Log error but don't crash
            from shared_data import append_log
            append_log(f"Error creating LLM provider {name}: {str(e)}")
            # Return OpenAI as fallback
            if "openai" not in _providers:
                try:
                    _providers["openai"] = OpenAIProvider()
                except:
                    pass
            return _providers.get("openai", None) if _providers else None
    
    return _providers.get(name)


def _get_available_provider(preferred: str = None) -> LLMProvider:
    # Try preferred provider
    if preferred:
        provider = _get_provider(preferred)
        if provider.is_available():
            return provider
    
    # Fallback order based on preference
    if preferred == "deepseek":
        fallback_order = ["deepseek", "openai", "gemini"]
    elif preferred == "gemini":
        fallback_order = ["gemini", "openai", "deepseek"]
    else:
        fallback_order = ["openai", "deepseek", "gemini"]
    
    for name in fallback_order:
        provider = _get_provider(name)
        if provider.is_available():
            return provider
    
    # Return OpenAI as last resort (will show error message)
    return _get_provider("openai")


def gpt_evaluate_deal(title: str, price: float, provider: str = None) -> str:
    
    prov_name = provider or get_llm_provider() or "openai"
    provider_instance = _get_available_provider(prov_name)
    return provider_instance.evaluate_deal(title, price)


def gpt_answer(question: str, context: str, provider: str = None) -> str:
    
    prov_name = provider or get_llm_provider() or "openai"
    provider_instance = _get_available_provider(prov_name)
    return provider_instance.answer_question(question, context)
