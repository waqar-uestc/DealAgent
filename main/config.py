import os
class Config:
    
    # Server Settings
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 7860
    SHOW_ERROR = True
    
    # Model Settings
    SENTENCE_MODEL = "all-MiniLM-L6-v2"
    OPENAI_MODEL = "gpt-3.5-turbo"
    GEMINI_MODEL = "gemini-1.5-flash"
    DEEPSEEK_MODEL = "deepseek-chat"
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER = "openai"  # "openai", "gemini", or "deepseek"
    LLM_TIMEOUT = 30  # seconds
    
    # Deal Fetching Settings
    MAX_DEALS = 20
    DEALS_COUNT = 150  # Number of deals to fetch and process (changed from TOP_DEALS_COUNT)
    RSS_TIMEOUT = 10  # seconds (reduced from 15 for faster response)
    RSS_MAX_SOURCES = 15  # Maximum RSS sources to process (prevents long waits)
    RSS_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    DEAL_PROCESS_LIMIT = 50  # Maximum deals to process at once (prevents timeout)
    
    # RAG Settings
    RAG_TOP_K = 2  # Number of relevant chunks to retrieve
    
    # Visualization Settings
    TSNE_PERPLEXITY_MIN = 2
    TSNE_PERPLEXITY_MAX = 30
    TSNE_RANDOM_STATE = 42
    PLOT_HEIGHT = 700
    PLOT_WIDTH = 1000
    PLOT_MARKER_SIZE = 6
    
    # Price Model Settings
    PRICE_MIN = 0.01
    PRICE_MAX = 100000
    PRICE_FALLBACK_MIN = 5
    PRICE_FALLBACK_MAX = 50
    
    # File Paths
    RSS_SOURCES_FILE = "rss_sources.txt"
    LOGS_FILE = "logs.txt"
    PLOT_OUTPUT_FILE = "deal_clusters_plot.html"
    
    # Environment Variables
    ENV_OPENAI_KEY = "OPENAI_API_KEY"
    ENV_GEMINI_KEY = "GEMINI_API_KEY"
    ENV_DEEPSEEK_KEY = "DEEPSEEK_API_KEY"
    
    # DeepSeek API Settings
    DEEPSEEK_API_BASE = "https://api.deepseek.com"
    
    @classmethod
    def from_env(cls):
        """Load configuration overrides from environment variables."""
        cls.SERVER_PORT = int(os.getenv("SERVER_PORT", cls.SERVER_PORT))
        cls.SERVER_HOST = os.getenv("SERVER_HOST", cls.SERVER_HOST)
        cls.DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", cls.DEFAULT_LLM_PROVIDER).lower()
        cls.MAX_DEALS = int(os.getenv("MAX_DEALS", cls.MAX_DEALS))
        cls.DEALS_COUNT = int(os.getenv("DEALS_COUNT", cls.DEALS_COUNT))
    
    @classmethod
    def get_project_path(cls):
        """Get the project root directory."""
        return os.path.dirname(os.path.abspath(__file__))
    
    @classmethod
    def get_full_path(cls, filename: str) -> str:
        """Get full path for a file in the project directory."""
        return os.path.join(cls.get_project_path(), filename)


# Load environment overrides on module import
Config.from_env()

