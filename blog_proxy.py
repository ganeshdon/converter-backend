# Blog Proxy for WordPress Integration
# Add this to your FastAPI backend to proxy /blog requests to Hostinger

import httpx
from fastapi import Request, HTTPException
from fastapi.responses import Response

# Your Hostinger WordPress URL (replace with actual)
WORDPRESS_BASE_URL = "https://your-hostinger-site.com/blog"

async def proxy_blog_request(request: Request, path: str = ""):
    """Proxy blog requests to WordPress on Hostinger"""
    
    # Construct the target URL
    target_url = f"{WORDPRESS_BASE_URL}/{path}" if path else WORDPRESS_BASE_URL
    
    # Forward query parameters
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Forward the request to WordPress
            response = await client.request(
                method=request.method,
                url=target_url,
                headers={
                    key: value for key, value in request.headers.items() 
                    if key.lower() not in ['host', 'content-length']
                },
                content=await request.body() if request.method in ['POST', 'PUT', 'PATCH'] else None
            )
            
            # Return the WordPress response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    key: value for key, value in response.headers.items()
                    if key.lower() not in ['content-encoding', 'transfer-encoding']
                }
            )
            
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Blog proxy error: {str(e)}")

# Add these routes to your main server.py:
"""
# Blog proxy routes
@app.get("/blog/{path:path}")
async def blog_get(request: Request, path: str = ""):
    return await proxy_blog_request(request, path)

@app.post("/blog/{path:path}")  
async def blog_post(request: Request, path: str = ""):
    return await proxy_blog_request(request, path)

@app.get("/blog")
async def blog_root(request: Request):
    return await proxy_blog_request(request, "")
"""