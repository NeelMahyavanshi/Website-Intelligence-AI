from supabase import create_client, Client
from utils.logger import get_logger
import os
from dotenv import load_dotenv
load_dotenv(override=True)

logger = get_logger("DATABASE")

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)

db = get_client()
logger.info("Supabase client initialized")
