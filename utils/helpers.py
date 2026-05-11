import tldextract
import hashlib


def extract_company_id(url: str) -> str:
    """Extract company identifier from URL using tldextract.
    
    Args:
        url: The URL to extract from
    Returns:
        A string representing the company identifier (e.g. "example" from "https://www.example.co.uk")
    """
    extracted = tldextract.extract(url)
    if not extracted.domain:
        raise ValueError(f"Could not extract company ID from URL: {url}")
    return f"{extracted.domain}.{extracted.suffix}"



def page_hash(content: str) -> str:
    """Computes MD5 hash of page content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()
