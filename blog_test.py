"""
Test script to verify WordPress connection
"""
import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

WORDPRESS_BASE_URL = os.getenv("WORDPRESS_BASE_URL", "https://mediumblue-shrew-791406.hostingersite.com")

async def test_wordpress_connection():
    print(f"Testing connection to: {WORDPRESS_BASE_URL}")
    
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0, follow_redirects=True) as client:
            response = await client.get(WORDPRESS_BASE_URL)
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print(f"Content Length: {len(response.content)}")
            print(f"First 200 chars: {response.text[:200]}")
            return True
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_wordpress_connection())
