from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
import os
from dotenv import load_dotenv
import tempfile
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import uuid
from bson import ObjectId
import logging
from pathlib import Path
from pydantic import BaseModel, Field
import aiohttp
import json
import httpx
import resend
# Removed Stripe integration - now using Dodo Payments

# Import our modules
from auth import get_password_hash, verify_password, create_access_token, verify_token, verify_jwt_token
from models import (
    UserSignup, UserLogin, UserResponse, TokenResponse, DocumentResponse,
    PagesCheckRequest, PagesCheckResponse, SubscriptionTier, SubscriptionPlan,
    UserUpdate, PasswordReset, PasswordResetRequest, PasswordChange, BillingInterval, GoogleUserData, UserSession,
    AnonymousConversionCheck, AnonymousConversionResponse, AnonymousConversionRecord,
    SubscriptionPackage, PaymentSessionRequest, PaymentSessionResponse, PaymentTransaction, WebhookEventResponse
)
import dodo_routes


ROOT_DIR = Path(__file__).parent
env_path = ROOT_DIR / '.env'
load_dotenv(env_path)

# Configure logging early for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug: Log .env file location and if it exists
logger.info(f"Loading .env from: {env_path}")
logger.info(f".env file exists: {env_path.exists()}")

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# CORS origins - default to production frontend URL
# For production, set CORS_ORIGINS environment variable to your frontend URL(s)
# Example: CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "https://yourbankstatementconverter.com,http://localhost:3000,http://127.0.0.1:3000")
# Handle wildcard - if "*" is provided, use specific origins for development
if CORS_ORIGINS_ENV.strip() == "*":
    CORS_ORIGINS = ["https://yourbankstatementconverter.com", "http://localhost:3000", "http://127.0.0.1:3000"]
else:
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",") if origin.strip()]
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable is required")

# Email configuration (Resend)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
RESEND_FROM_NAME = os.getenv("RESEND_FROM_NAME", "Bank Statement Converter")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://yourbankstatementconverter.com")

# Debug: Log loaded environment variables (without sensitive data)
logger.info(f"FRONTEND_URL loaded: {FRONTEND_URL}")
logger.info(f"CORS_ORIGINS loaded: {CORS_ORIGINS}")
logger.info(f"RESEND_API_KEY configured: {'Yes' if RESEND_API_KEY else 'No'}")

# Define subscription packages - SECURITY: Server-side only pricing
# WordPress Hostinger Configuration
WORDPRESS_BASE_URL = os.getenv("WORDPRESS_BASE_URL", "https://yourbankstatementconverter.com")

SUBSCRIPTION_PACKAGES = {
    "starter": {
        "name": "Starter",
        "monthly_price": 15.0,
        "annual_price": 12.0,  # 20% discount
        "pages_limit": 400,
        "features": ["400 pages/month", "Email support", "PDF conversion"]
    },
    "professional": {
        "name": "Professional", 
        "monthly_price": 30.0,
        "annual_price": 24.0,  # 20% discount
        "pages_limit": 1000,
        "features": ["1000 pages/month", "Priority support", "Advanced features"]
    },
    "business": {
        "name": "Business",
        "monthly_price": 50.0,
        "annual_price": 40.0,  # 20% discount
        "pages_limit": 4000,
        "features": ["4000 pages/month", "Priority support", "Team features"]
    },
    "enterprise": {
        "name": "Enterprise",
        "monthly_price": 100.0,  # Custom pricing starts here
        "annual_price": 80.0,
        "pages_limit": -1,  # Unlimited
        "features": ["Unlimited pages", "Dedicated support", "Custom integration"]
    }
}

client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Collections
users_collection = db.users
documents_collection = db.documents
subscriptions_collection = db.subscriptions
user_sessions_collection = db.user_sessions
anonymous_conversions_collection = db.anonymous_conversions
payment_transactions_collection = db.payment_transactions
password_reset_tokens_collection = db.password_reset_tokens

# Create the main app without a prefix
app = FastAPI()

# CORS configuration - MUST be added BEFORE routers
# Use specific origins to allow credentials
# For production, set CORS_ORIGINS environment variable to your frontend URL(s)
# Example: CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
print(f"CORS Origins configured: {CORS_ORIGINS}")  # Debug log
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class BankStatementData(BaseModel):
    accountInfo: dict
    deposits: list
    atmWithdrawals: list
    checksPaid: list
    visaPurchases: Optional[list] = []

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Authentication endpoints
@api_router.post("/auth/signup", response_model=TokenResponse)
async def signup(user_data: UserSignup):
    """Register a new user"""
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    now = datetime.now(timezone.utc)
    user_doc = {
        "_id": user_id,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "password_hash": hashed_password,
        "subscription_tier": SubscriptionTier.DAILY_FREE,
        "pages_remaining": 7,  # Daily free tier starts with 7 pages
        "pages_limit": 7,
        "billing_cycle_start": now,
        "daily_reset_time": now,
        "language_preference": "en",
        "created_at": now,
        "updated_at": now
    }
    
    await users_collection.insert_one(user_doc)
    
    # Create access token
    access_token = create_access_token(data={"sub": user_id, "email": user_data.email})
    
    # Return user data
    user_response = UserResponse(
        id=user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        subscription_tier=SubscriptionTier.DAILY_FREE,
        pages_remaining=7,
        pages_limit=7,
        billing_cycle_start=now,
        daily_reset_time=now,
        language_preference="en",
        billing_interval=None
    )
    
    return TokenResponse(access_token=access_token, token_type="bearer", user=user_response)

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Login user"""
    user = await users_collection.find_one({"email": credentials.email})
    
    # Check if user exists
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Check if user is an OAuth user (no password_hash)
    if not user.get("password_hash"):
        raise HTTPException(
            status_code=401, 
            detail="This account was created with Google. Please use Google login instead."
        )
    
    # Verify password for regular users
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Check if daily free user needs reset
    if user["subscription_tier"] == SubscriptionTier.DAILY_FREE:
        await check_and_reset_daily_pages(user["_id"])
        user = await users_collection.find_one({"_id": user["_id"]})  # Refresh user data
    
    # Create access token
    access_token = create_access_token(data={"sub": user["_id"], "email": user["email"]})
    
    user_response = UserResponse(
        id=user["_id"],
        email=user["email"],
        full_name=user["full_name"],
        subscription_tier=user["subscription_tier"],
        pages_remaining=user["pages_remaining"],
        pages_limit=user["pages_limit"],
        billing_cycle_start=user.get("billing_cycle_start"),
        daily_reset_time=user.get("daily_reset_time"),
        language_preference=user.get("language_preference", "en"),
        billing_interval=user.get("billing_interval")
    )
    
    return TokenResponse(access_token=access_token, token_type="bearer", user=user_response)

@api_router.post("/auth/logout")
async def logout(current_user: dict = Depends(verify_token)):
    """Logout user (client should delete token)"""
    return {"message": "Logged out successfully"}

async def send_password_reset_email(to_email: str, reset_link: str, user_name: str = None):
    """Send password reset email to user using Resend"""
    try:
        # If Resend API key is not configured, log the email instead
        if not RESEND_API_KEY:
            logger.warning(f"Resend API key not configured. Password reset link for {to_email}: {reset_link}")
            logger.info(f"To enable email sending, configure RESEND_API_KEY environment variable")
            return False
        
        # Validate API key format
        if not RESEND_API_KEY.startswith('re_'):
            logger.error(f"Invalid Resend API key format. API key should start with 're_'. Current key starts with: {RESEND_API_KEY[:3] if len(RESEND_API_KEY) >= 3 else 'N/A'}")
            return False
        
        # Log configuration (without exposing full API key)
        logger.info(f"Attempting to send password reset email to {to_email}")
        logger.info(f"Using FROM email: {RESEND_FROM_EMAIL}")
        logger.info(f"Resend API key configured: Yes (starts with {RESEND_API_KEY[:5]}...)")
        
        # Set Resend API key (for version 2.4.0)
        resend.api_key = RESEND_API_KEY
        
        # Email body (HTML)
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">Password Reset Request</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <p>Hello{f' {user_name}' if user_name else ''},</p>
                <p>We received a request to reset your password for your Bank Statement Converter account.</p>
                <p>Click the button below to reset your password:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}" style="background-color: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Reset Password</a>
                </div>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #667eea;">{reset_link}</p>
                <p><strong>This link will expire in 1 hour.</strong></p>
                <p>If you didn't request a password reset, please ignore this email. Your password will remain unchanged.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 12px; color: #666;">This is an automated message, please do not reply to this email.</p>
            </div>
        </body>
        </html>
        """
        
        # Plain text version
        text_body = f"""
        Password Reset Request
        
        Hello{f' {user_name}' if user_name else ''},
        
        We received a request to reset your password for your Bank Statement Converter account.
        
        Click the following link to reset your password:
        {reset_link}
        
        This link will expire in 1 hour.
        
        If you didn't request a password reset, please ignore this email. Your password will remain unchanged.
        
        This is an automated message, please do not reply to this email.
        """
        
        # Send email using Resend
        # Run in thread pool to avoid blocking
        def send_sync():
            try:
                params = {
                    "from": f"{RESEND_FROM_NAME} <{RESEND_FROM_EMAIL}>",
                    "to": [to_email],
                    "subject": "Reset Your Password - Bank Statement Converter",
                    "html": html_body,
                    "text": text_body,
                }
                
                logger.info(f"Sending email via Resend with params: from={RESEND_FROM_EMAIL}, to={to_email}")
                result = resend.Emails.send(params)
                
                # Log the full result for debugging
                logger.info(f"Resend API response: {result}")
                
                # Resend returns a dict with 'id' if successful
                if result and isinstance(result, dict):
                    if result.get('id'):
                        logger.info(f"Resend email sent successfully. Email ID: {result.get('id')}")
                        return True
                    elif result.get('error'):
                        # Resend API error response
                        error_msg = result.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Resend API error: {error_msg}")
                        logger.error(f"Full error response: {result}")
                        return False
                    else:
                        logger.error(f"Resend returned unexpected response (no 'id' or 'error' field): {result}")
                        return False
                else:
                    logger.error(f"Resend returned unexpected response type: {type(result)}, value: {result}")
                    return False
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.error(f"Exception sending email via Resend: {error_type}: {error_msg}")
                logger.exception(e)  # Log full traceback for debugging
                
                # Check for common error patterns
                if 'unauthorized' in error_msg.lower() or '401' in error_msg:
                    logger.error("Possible issue: Invalid Resend API key or API key not authorized")
                elif 'domain' in error_msg.lower() or 'from' in error_msg.lower():
                    logger.error(f"Possible issue: FROM email ({RESEND_FROM_EMAIL}) might not be verified in Resend")
                elif 'rate limit' in error_msg.lower():
                    logger.error("Possible issue: Rate limit exceeded for Resend API")
                
                return False
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, send_sync)
        
        if success:
            logger.info(f"Password reset email sent successfully to {to_email} via Resend")
        else:
            logger.error(f"Failed to send password reset email to {to_email} via Resend")
            # Still log the link for manual testing
            logger.warning(f"Password reset link for manual testing: {reset_link}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in send_password_reset_email: {str(e)}")
        logger.exception(e)  # Log full traceback
        return False

@api_router.post("/auth/forgot-password")
async def forgot_password(reset_request: PasswordReset):
    """Request password reset - sends reset token to user's email"""
    user = await users_collection.find_one({"email": reset_request.email})
    
    # Don't reveal if user exists or not (security best practice)
    if not user:
        # Still return success to prevent email enumeration
        return {"message": "If an account with that email exists, a password reset link has been sent."}
    
    # Check if user is an OAuth user (no password_hash)
    if not user.get("password_hash"):
        # Still return success to prevent revealing account type
        return {"message": "If an account with that email exists, a password reset link has been sent."}
    
    # Generate reset token
    reset_token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # Token expires in 1 hour
    
    # Store reset token
    reset_token_doc = {
        "user_id": user["_id"],
        "email": user["email"],
        "token": reset_token,
        "expires_at": expires_at,
        "used": False,
        "created_at": datetime.now(timezone.utc)
    }
    
    # Delete any existing reset tokens for this user
    await password_reset_tokens_collection.delete_many({"user_id": user["_id"], "used": False})
    
    # Insert new reset token
    await password_reset_tokens_collection.insert_one(reset_token_doc)
    
    # Generate reset link
    reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"
    
    # Send email with reset link
    user_name = user.get("full_name", "")
    email_sent = await send_password_reset_email(user["email"], reset_link, user_name)
    
    if not email_sent:
        # Log the link if email sending failed (for development/testing)
        logger.warning(f"Email sending failed. Password reset link for {user['email']}: {reset_link}")
    
    return {"message": "If an account with that email exists, a password reset link has been sent."}

@api_router.post("/auth/reset-password")
async def reset_password(reset_request: PasswordResetRequest):
    """Reset password using reset token"""
    # Find the reset token
    reset_token_doc = await password_reset_tokens_collection.find_one({
        "token": reset_request.token,
        "used": False
    })
    
    if not reset_token_doc:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Check if token has expired
    expires_at = reset_token_doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        # Mark as used even though expired
        await password_reset_tokens_collection.update_one(
            {"token": reset_request.token},
            {"$set": {"used": True}}
        )
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")
    
    # Get user
    user = await users_collection.find_one({"_id": reset_token_doc["user_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash new password
    hashed_password = get_password_hash(reset_request.new_password)
    
    # Update user password
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"password_hash": hashed_password, "updated_at": datetime.now(timezone.utc)}}
    )
    
    # Mark reset token as used
    await password_reset_tokens_collection.update_one(
        {"token": reset_request.token},
        {"$set": {"used": True}}
    )
    
    # Delete all other unused reset tokens for this user
    await password_reset_tokens_collection.delete_many({
        "user_id": user["_id"],
        "used": False,
        "token": {"$ne": reset_request.token}
    })
    
    logger.info(f"Password reset successful for user {user['email']}")
    
    return {"message": "Password has been reset successfully. You can now login with your new password."}

# Google OAuth Authentication using Emergent Auth
@api_router.get("/auth/oauth/session-data")
async def get_session_data(request: Request):
    """Process session_id from Emergent Auth and return user data"""
    session_id = request.headers.get("X-Session-ID")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    
    try:
        # Call Emergent Auth API to get user data
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_id}
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Invalid session ID")
                
                oauth_data = await response.json()
        
        # Check if user exists by email
        existing_user = await users_collection.find_one({"email": oauth_data["email"]})
        
        if existing_user:
            user_id = existing_user["_id"]
            # Update user data if needed (but don't overwrite existing data)
            if not existing_user.get("picture"):
                await users_collection.update_one(
                    {"_id": user_id},
                    {"$set": {"picture": oauth_data.get("picture")}}
                )
        else:
            # Create new user from OAuth data
            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            user_doc = {
                "_id": user_id,
                "email": oauth_data["email"],
                "full_name": oauth_data["name"],
                "picture": oauth_data.get("picture"),
                "subscription_tier": SubscriptionTier.DAILY_FREE,
                "pages_remaining": 7,  # Daily free tier starts with 7 pages
                "pages_limit": 7,
                "billing_cycle_start": now,
                "daily_reset_time": now,
                "language_preference": "en",
                "created_at": now,
                "updated_at": now,
                "oauth_provider": "google"
            }
            await users_collection.insert_one(user_doc)
        
        # Create or update session token
        session_token = oauth_data["session_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        session_doc = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc)
        }
        
        # Upsert session (replace existing or create new)
        await user_sessions_collection.replace_one(
            {"user_id": user_id},
            session_doc,
            upsert=True
        )
        
        # Get updated user data
        user = await users_collection.find_one({"_id": user_id})
        
        # Return user data with session_token for OAuth users
        user_response = UserResponse(
            id=user["_id"],
            email=user["email"],
            full_name=user["full_name"],
            subscription_tier=user["subscription_tier"],
            pages_remaining=user["pages_remaining"],
            pages_limit=user["pages_limit"],
            billing_cycle_start=user.get("billing_cycle_start"),
            daily_reset_time=user.get("daily_reset_time"),
            language_preference=user.get("language_preference", "en")
        )
        
        # Return as dict to include session_token
        response_dict = user_response.dict()
        response_dict["session_token"] = session_token
        return response_dict
        
    except Exception as e:
        logger.error(f"OAuth session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process OAuth session")

@api_router.post("/auth/oauth/logout")
async def oauth_logout(request: Request, response: Response):
    """Logout user - delete session and clear cookie"""
    # Try to get session token from cookie first, then from Authorization header
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    
    if session_token:
        # Delete session from database
        await user_sessions_collection.delete_one({"session_token": session_token})
        
        # Clear cookie
        response.delete_cookie("session_token", path="/", secure=True, samesite="none")
    
    return {"message": "Logged out successfully"}

# Helper function to get current user from session token (for OAuth users)
async def get_current_user_from_session(session_token: str) -> Optional[dict]:
    """Get user from session token"""
    session = await user_sessions_collection.find_one({"session_token": session_token})
    if not session:
        return None
    
    # Handle timezone-aware comparison
    expires_at = session["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        return None
    
    user = await users_collection.find_one({"_id": session["user_id"]})
    if user:
        return {"user_id": user["_id"], "email": user["email"]}
    return None

# Updated auth dependency to support both JWT and OAuth session tokens
async def get_current_user(request: Request):
    """Get current user from JWT token or OAuth session token"""
    # First try to get session token from cookie
    session_token = request.cookies.get("session_token")
    if session_token:
        user = await get_current_user_from_session(session_token)
        if user:
            return user
    
    # Fallback to Authorization header (for both JWT and session tokens)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    
    # Try as session token first
    user = await get_current_user_from_session(token)
    if user:
        return user
    
    # Try as JWT token
    try:
        return verify_jwt_token(token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid token")

@api_router.get("/user/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    user = await users_collection.find_one({"_id": current_user["user_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if daily free user needs reset
    if user["subscription_tier"] == SubscriptionTier.DAILY_FREE:
        await check_and_reset_daily_pages(user["_id"])
        user = await users_collection.find_one({"_id": user["_id"]})
    
    return UserResponse(
        id=user["_id"],
        email=user["email"],
        full_name=user["full_name"],
        subscription_tier=user["subscription_tier"],
        pages_remaining=user["pages_remaining"],
        pages_limit=user["pages_limit"],
        billing_cycle_start=user.get("billing_cycle_start"),
        daily_reset_time=user.get("daily_reset_time"),
        language_preference=user.get("language_preference", "en"),
        billing_interval=user.get("billing_interval")
    )

@api_router.put("/user/profile", response_model=UserResponse)
async def update_profile(updates: UserUpdate, current_user: dict = Depends(get_current_user)):
    """Update user profile"""
    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc)
    
    await users_collection.update_one(
        {"_id": current_user["user_id"]},
        {"$set": update_data}
    )
    
    user = await users_collection.find_one({"_id": current_user["user_id"]})
    return UserResponse(
        id=user["_id"],
        email=user["email"],
        full_name=user["full_name"],
        subscription_tier=user["subscription_tier"],
        pages_remaining=user["pages_remaining"],
        pages_limit=user["pages_limit"],
        billing_cycle_start=user.get("billing_cycle_start"),
        daily_reset_time=user.get("daily_reset_time"),
        language_preference=user.get("language_preference", "en"),
        billing_interval=user.get("billing_interval")
    )

@api_router.post("/user/pages/check", response_model=PagesCheckResponse)
async def check_pages(pages_request: PagesCheckRequest, current_user: dict = Depends(get_current_user)):
    """Check if user has enough pages for conversion"""
    user = await users_collection.find_one({"_id": current_user["user_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    tier = user["subscription_tier"]
    # Check if daily_free (handle both enum and string values)
    # Convert enum to string for comparison
    tier_str = str(tier).lower() if tier else ""
    is_daily_free = "daily_free" in tier_str
    
    logger.info(f"check_pages: user_id={current_user['user_id']}, tier={tier}, is_daily_free={is_daily_free}, pages_remaining={user['pages_remaining']}, requested={pages_request.page_count}")
    
    # Check if daily free user needs reset
    if is_daily_free:
        await check_and_reset_daily_pages(user["_id"])
        user = await users_collection.find_one({"_id": user["_id"]})
        logger.info(f"check_pages after reset: pages_remaining={user['pages_remaining']}")
    
    can_convert = user["pages_remaining"] >= pages_request.page_count
    
    if is_daily_free:
        daily_reset_time = user["daily_reset_time"]
        if daily_reset_time and daily_reset_time.tzinfo is None:
            daily_reset_time = daily_reset_time.replace(tzinfo=timezone.utc)
        next_reset = daily_reset_time + timedelta(days=1)
        message = f"You have {user['pages_remaining']} pages remaining today. Resets in {(next_reset - datetime.now(timezone.utc)).seconds // 3600} hours."
    else:
        billing_cycle_start = user.get("billing_cycle_start", datetime.now(timezone.utc))
        if billing_cycle_start and billing_cycle_start.tzinfo is None:
            billing_cycle_start = billing_cycle_start.replace(tzinfo=timezone.utc)
        next_reset = billing_cycle_start + timedelta(days=30)
        message = f"You have {user['pages_remaining']} pages remaining this month."
    
    if not can_convert:
        if is_daily_free:
            message = "You've used all your daily pages. Upgrade to continue or wait for reset."
        else:
            message = "You've used all your monthly pages. Upgrade your plan to continue."
    
    return PagesCheckResponse(
        can_convert=can_convert,
        pages_remaining=user["pages_remaining"],
        pages_limit=user["pages_limit"],
        reset_date=next_reset,
        message=message
    )


# NOTE: Transactions listing endpoint removed per request — transaction records may still
# be stored by webhook handlers but are no longer exposed via this API.

@api_router.post("/process-pdf")
async def process_pdf_with_ai(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Process PDF bank statement using AI for enhanced accuracy"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # First count pages in the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Count pages (simple implementation - you can enhance this)
        page_count = await count_pdf_pages(tmp_file_path)
        
        # Check if user has enough pages
        user = await users_collection.find_one({"_id": current_user["user_id"]})
        tier = user["subscription_tier"]
        # Check if daily_free (handle both enum and string values)
        tier_str = str(tier).lower() if tier else ""
        is_daily_free = "daily_free" in tier_str
        
        logger.info(f"process_pdf: user_id={current_user['user_id']}, tier={tier}, is_daily_free={is_daily_free}, page_count={page_count}, pages_remaining={user['pages_remaining']}")
        
        # Reset daily pages if needed
        if is_daily_free:
            await check_and_reset_daily_pages(user["_id"])
            user = await users_collection.find_one({"_id": user["_id"]})
            logger.info(f"process_pdf after reset: pages_remaining={user['pages_remaining']}")
        
        if user["pages_remaining"] < page_count:
            logger.error(f"Insufficient pages: need {page_count}, have {user['pages_remaining']}")
            os.unlink(tmp_file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient pages. You need {page_count} pages but only have {user['pages_remaining']} remaining."
            )
        
        # Process with AI
        extracted_data = await extract_with_ai(tmp_file_path)
        
        # Deduct pages after successful conversion
        await users_collection.update_one(
            {"_id": current_user["user_id"]},
            {"$inc": {"pages_remaining": -page_count}}
        )
        
        # Save document record with 24-hour expiration
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=24)  # Documents expire after 24 hours
        document_doc = {
            "_id": doc_id,
            "user_id": current_user["user_id"],
            "original_filename": file.filename,
            "file_size": len(content),
            "page_count": page_count,
            "pages_deducted": page_count,
            "conversion_date": now,
            "expires_at": expires_at,  # 24-hour expiration
            "download_count": 0,
            "status": "completed"
        }
        await documents_collection.insert_one(document_doc)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {"success": True, "data": extracted_data, "pages_used": page_count}
        
    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"PDF processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@api_router.get("/documents", response_model=List[DocumentResponse])
async def get_documents(current_user: dict = Depends(get_current_user)):
    """Get user's document history (only non-expired documents)"""
    now = datetime.now(timezone.utc)
    # Only return documents that haven't expired
    documents = await documents_collection.find(
        {
            "user_id": current_user["user_id"],
            "$or": [
                {"expires_at": {"$exists": False}},  # Legacy documents without expiration
                {"expires_at": {"$gt": now}}  # Documents that haven't expired
            ]
        }
    ).sort("conversion_date", -1).to_list(length=100)
    
    return [DocumentResponse(
        id=doc["_id"],
        original_filename=doc["original_filename"],
        file_size=doc["file_size"],
        page_count=doc["page_count"],
        pages_deducted=doc["pages_deducted"],
        conversion_date=doc["conversion_date"],
        download_count=doc.get("download_count", 0),
        status=doc["status"]
    ) for doc in documents]

@api_router.get("/documents/{doc_id}/download")
async def download_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Download converted document (mock implementation)"""
    document = await documents_collection.find_one({
        "_id": doc_id,
        "user_id": current_user["user_id"]
    })
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # In a real implementation, you'd retrieve the actual converted file
    # For now, we'll return a sample CSV
    sample_csv = """Account Summary,Value,
Account Number,000009752,
Statement Date,June 5 2003,
Beginning Balance,$7126.11,
Ending Balance,$10521.19,"""
    
    # Update download count
    await documents_collection.update_one(
        {"_id": doc_id},
        {"$inc": {"download_count": 1}}
    )
    
    from fastapi.responses import Response
    return Response(
        content=sample_csv,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={document['original_filename'].replace('.pdf', '-converted.csv')}"}
    )

@api_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document"""
    result = await documents_collection.delete_one({
        "_id": doc_id,
        "user_id": current_user["user_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}

# Anonymous conversion tracking endpoints
@api_router.post("/anonymous/check", response_model=AnonymousConversionResponse)
async def check_anonymous_conversion(request: Request, conversion_check: AnonymousConversionCheck):
    """Check if anonymous user can perform a free conversion"""
    try:
        # Get IP address from request
        ip_address = request.client.host
        
        # Count existing conversions for this fingerprint + IP combo
        existing_conversions = await anonymous_conversions_collection.count_documents({
            "$or": [
                {"browser_fingerprint": conversion_check.browser_fingerprint},
                {"ip_address": ip_address}
            ]
        })
        
        can_convert = existing_conversions == 0
        
        if can_convert:
            message = "You have 1 free conversion available!"
        else:
            message = "Free conversion limit reached. Please sign up for unlimited conversions."
        
        return AnonymousConversionResponse(
            can_convert=can_convert,
            conversions_used=existing_conversions,
            conversions_limit=1,
            message=message,
            requires_signup=not can_convert
        )
        
    except Exception as e:
        logger.error(f"Anonymous conversion check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check conversion limit")

@api_router.post("/anonymous/convert")
async def anonymous_convert_pdf(request: Request, file: UploadFile = File(...)):
    """Process PDF for anonymous users (1 free conversion)"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="AI processing service not available")
    
    try:
        # Get client info
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent", "")
        browser_fingerprint = request.headers.get("X-Browser-Fingerprint")
        
        if not browser_fingerprint:
            raise HTTPException(status_code=400, detail="Browser fingerprint required")
        
        # Check if user has already used free conversion
        existing_conversion = await anonymous_conversions_collection.find_one({
            "$or": [
                {"browser_fingerprint": browser_fingerprint},
                {"ip_address": ip_address}
            ]
        })
        
        if existing_conversion:
            raise HTTPException(
                status_code=403, 
                detail="Free conversion limit reached. Please sign up for unlimited conversions."
            )
        
        # Process PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Count pages
        page_count = await count_pdf_pages(tmp_file_path)
        
        # Extract data with AI
        extracted_data = await extract_with_ai(tmp_file_path)
        
        # Record the anonymous conversion
        conversion_record = {
            "browser_fingerprint": browser_fingerprint,
            "ip_address": ip_address,
            "filename": file.filename,
            "file_size": len(content),
            "page_count": page_count,
            "conversion_date": datetime.now(timezone.utc),
            "user_agent": user_agent
        }
        
        await anonymous_conversions_collection.insert_one(conversion_record)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            "success": True, 
            "data": extracted_data, 
            "message": "Free conversion completed! Sign up for unlimited conversions.",
            "pages_processed": page_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"Anonymous PDF processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# Dodo Payments - Integrated via dodo_routes.py (removed Stripe)

@api_router.get("/pricing/plans")
async def get_pricing_plans():
    """Get available pricing plans"""
    plans = []
    
    for package_id, package_data in SUBSCRIPTION_PACKAGES.items():
        plan = {
            "tier": package_id,
            "name": package_data["name"],
            "price_monthly": package_data["monthly_price"],
            "price_annual": package_data["annual_price"],
            "pages_limit": package_data["pages_limit"],
            "features": package_data["features"],
            "is_popular": package_id == "professional"  # Mark professional as popular
        }
        plans.append(plan)
    
    # Add daily free plan
    plans.insert(0, {
        "tier": "daily_free",
        "name": "Daily Free",
        "price_monthly": 0,
        "price_annual": 0,
        "pages_limit": 7,
        "features": ["7 pages per day", "Resets every 24 hours", "Basic support"],
        "is_popular": False
    })
    
    return plans

# Blog Proxy Functionality
async def proxy_blog_request(request: Request, path: str = ""):
    """Proxy blog requests to WordPress on Hostinger"""
    
    # Construct the target URL using WORDPRESS_BASE_URL environment variable
    if path:
        target_url = f"{WORDPRESS_BASE_URL}/{path}"
    else:
        target_url = f"{WORDPRESS_BASE_URL}/"
    
    # Forward query parameters
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False  # For development, set to True in production
        ) as client:
            
            # Prepare headers (exclude problematic ones including Accept-Encoding)
            # Important: Strip Accept-Encoding to prevent intermediate proxies from re-compressing
            headers = {
                key: value for key, value in request.headers.items() 
                if key.lower() not in [
                    'host', 'content-length', 'content-encoding', 
                    'transfer-encoding', 'connection', 'accept-encoding'
                ]
            }
            
            # Add proper headers for WordPress
            # Explicitly request no compression with Accept-Encoding: identity
            headers.update({
                'User-Agent': 'BankStatementConverter-Proxy/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'identity',  # Explicitly request no compression
                'Cache-Control': 'no-cache',
            })
            
            # Handle request body for POST/PUT/PATCH
            content = None
            if request.method in ['POST', 'PUT', 'PATCH']:
                content = await request.body()
            
            # Make request to WordPress
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=content
            )
            
            # Prepare response headers
            # Strip headers that could cause compression issues with intermediate proxies
            response_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() not in [
                    'content-encoding', 'transfer-encoding', 'connection',
                    'server', 'date', 'content-length', 'vary'
                ]
            }
            
            # Add CORS headers if needed
            response_headers.update({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            })
            
            # Return the WordPress response
            # httpx automatically handles decompression, so response.text should be decoded
            content_type = response.headers.get('content-type', 'text/html')
            
            # Log for debugging
            logger.info(f"WordPress response - Status: {response.status_code}, Content-Type: {content_type}, Original encoding: {response.headers.get('content-encoding', 'none')}")
            
            # Rewrite URLs in HTML/CSS/JS content to use production domain
            response_headers['Content-Type'] = response.headers.get('content-type', 'text/html; charset=UTF-8')
            response_headers['Cache-Control'] = 'no-cache, no-store, no-transform, must-revalidate'
            response_headers['Content-Encoding'] = 'identity'
            
            # Get the content
            content = response.content
            
            # For HTML, CSS, and JS files, rewrite URLs
            if 'text/html' in content_type or 'text/css' in content_type or 'javascript' in content_type:
                try:
                    text_content = response.text
                    
                    # Replace Hostinger URLs with production URLs
                    # Replace absolute URLs
                    text_content = text_content.replace(
                        'https://mediumblue-shrew-791406.hostingersite.com',
                        'https://yourbankstatementconverter.com/blog'
                    )
                    text_content = text_content.replace(
                        'http://mediumblue-shrew-791406.hostingersite.com',
                        'https://yourbankstatementconverter.com/blog'
                    )
                    
                    # Replace protocol-relative URLs
                    text_content = text_content.replace(
                        '//mediumblue-shrew-791406.hostingersite.com',
                        '//yourbankstatementconverter.com/blog'
                    )
                    
                    # For HTML, also update wp-content and wp-includes paths to go through proxy
                    if 'text/html' in content_type:
                        # Fix relative paths for assets
                        text_content = text_content.replace('src="/wp-', 'src="/api/blog/wp-')
                        text_content = text_content.replace('href="/wp-', 'href="/api/blog/wp-')
                        text_content = text_content.replace('src=\"/wp-', 'src=\"/api/blog/wp-')
                        text_content = text_content.replace('href=\"/wp-', 'href=\"/api/blog/wp-')
                    
                    content = text_content.encode('utf-8')
                except Exception as e:
                    logger.warning(f"Failed to rewrite URLs in content: {str(e)}")
                    # Use original content if rewriting fails
                    content = response.content
            
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get('content-type', 'text/html')
            )
            
    except httpx.TimeoutException:
        logger.error(f"Timeout while proxying to WordPress: {target_url}")
        return HTMLResponse(
            content="<h1>Blog Temporarily Unavailable</h1><p>Please try again later.</p>",
            status_code=504
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else f"{error_type} with no message"
        logger.error(f"Blog proxy error for {target_url}: {error_msg}")
        logger.error(f"Error type: {error_type}")
        logger.error(f"Full traceback: {error_details}")
        return HTMLResponse(
            content=f"<h1>Blog Error</h1><p>Unable to load blog content.</p><p>Error: {error_msg}</p><p>Type: {error_type}</p><pre style='background: #f5f5f5; padding: 10px; overflow: auto;'>{error_details}</pre>",
            status_code=502
        )

# WordPress Blog Proxy Routes - Serves content at /api/blog
@api_router.get("/blog/{path:path}")
async def blog_proxy_get(request: Request, path: str):
    """Proxy GET requests to WordPress blog"""
    return await proxy_blog_request(request, path)

@api_router.post("/blog/{path:path}")
async def blog_proxy_post(request: Request, path: str):
    """Proxy POST requests to WordPress blog (for admin, forms, etc.)"""
    return await proxy_blog_request(request, path)

@api_router.get("/blog/health")
async def blog_health_check():
    """Health check for WordPress proxy"""
    try:
        wordpress_url = os.getenv("WORDPRESS_BASE_URL", "https://mediumblue-shrew-791406.hostingersite.com")
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            response = await client.get(wordpress_url)
            return {
                "status": "ok",
                "wordpress_url": wordpress_url,
                "wordpress_status": response.status_code,
                "can_connect": True
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "wordpress_url": os.getenv("WORDPRESS_BASE_URL", "not set"),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }

@api_router.get("/blog")
async def blog_root_get(request: Request):
    """Proxy GET requests to WordPress blog root"""
    return await proxy_blog_request(request, "")

@api_router.post("/blog")
async def blog_root_post(request: Request):
    """Proxy POST requests to WordPress blog root"""
    return await proxy_blog_request(request, "")

# Handle WordPress admin redirect
@api_router.get("/blog/admin")
async def blog_admin_redirect():
    """Redirect /blog/admin to /blog/wp-admin"""
    return Response(
        status_code=301,
        headers={"Location": "/api/blog/wp-admin"}
    )

# Handle WordPress admin routes  
@api_router.get("/blog/wp-admin/{path:path}")
async def blog_wp_admin_get(request: Request, path: str):
    """Proxy WordPress admin GET requests"""
    return await proxy_blog_request(request, f"wp-admin/{path}")

@api_router.post("/blog/wp-admin/{path:path}")
async def blog_wp_admin_post(request: Request, path: str):
    """Proxy WordPress admin POST requests"""
    return await proxy_blog_request(request, f"wp-admin/{path}")

@api_router.get("/blog/wp-admin")
async def blog_wp_admin_root(request: Request):
    """Proxy WordPress admin root"""
    return await proxy_blog_request(request, "wp-admin/")

# Handle WordPress content (images, CSS, JS)
@api_router.get("/blog/wp-content/{path:path}")
async def blog_wp_content(request: Request, path: str):
    """Proxy WordPress content files"""
    return await proxy_blog_request(request, f"wp-content/{path}")

@api_router.get("/blog/wp-includes/{path:path}")
async def blog_wp_includes(request: Request, path: str):
    """Proxy WordPress includes files"""
    return await proxy_blog_request(request, f"wp-includes/{path}")

async def count_pdf_pages(pdf_path: str) -> int:
    """Count pages in PDF file"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
    except Exception as e:
        print(f"Error counting PDF pages: {e}")
        return 1  # Default to 1 page if counting fails

async def cleanup_expired_documents():
    """Delete documents that have expired (older than 24 hours)"""
    try:
        now = datetime.now(timezone.utc)
        # Delete documents that have an expires_at field and it's in the past
        result = await documents_collection.delete_many({
            "expires_at": {"$exists": True, "$lte": now}
        })
        if result.deleted_count > 0:
            logger.info(f"Cleaned up {result.deleted_count} expired documents")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up expired documents: {str(e)}")
        return 0

async def periodic_cleanup():
    """Background task that runs every hour to clean up expired documents"""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await cleanup_expired_documents()
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {str(e)}")
            await asyncio.sleep(3600)  # Wait before retrying

async def check_and_reset_daily_pages(user_id: str):
    """Check if daily free tier user needs page reset"""
    user = await users_collection.find_one({"_id": user_id})
    if not user or user["subscription_tier"] != SubscriptionTier.DAILY_FREE:
        return
    
    now = datetime.now(timezone.utc)
    last_reset = user.get("daily_reset_time", now)
    
    # Ensure last_reset has timezone info
    if last_reset and last_reset.tzinfo is None:
        last_reset = last_reset.replace(tzinfo=timezone.utc)
    
    # Check if 24 hours have passed
    if (now - last_reset).total_seconds() >= 24 * 3600:  # 24 hours
        await users_collection.update_one(
            {"_id": user_id},
            {
                "$set": {
                    "pages_remaining": 7,  # Reset to 7 pages
                    "daily_reset_time": now
                }
            }
        )

async def extract_with_ai(pdf_path: str):
    """Use Google Generative AI to extract bank statement data from PDF"""
    
    try:
        import google.generativeai as genai
        
        logger.info("Using google-generativeai for PDF extraction")
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Upload the PDF file
        logger.info(f"Uploading PDF file: {pdf_path}")
        uploaded_file = genai.upload_file(pdf_path)
        logger.info(f"File uploaded successfully: {uploaded_file.name}")
        
        # Create the model - try multiple models with fallback
        model = None
        models_to_try = [
            'gemini-2.5-flash',           # Newest, fastest
            'gemini-2.5-flash-latest',    # Latest 2.5
            'gemini-1.5-flash-latest',    # Fallback to 1.5
            'gemini-1.5-flash',           # Stable 1.5
            'gemini-1.5-pro'              # Last resort
        ]
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Successfully initialized model: {model_name}")
                break
            except Exception as model_error:
                logger.warning(f"Model {model_name} not available: {model_error}")
                continue
        
        if model is None:
            raise Exception("No available Gemini models found. Please check your API key and quota.")
        
        # Create the prompt
        prompt = """You are a specialized bank statement data extraction expert. 
Your task is to extract ALL transaction data from PDF bank statements with 100% accuracy.

Extract and return data in this exact JSON structure:
{
  "accountInfo": {
    "accountNumber": "string",
    "statementDate": "string", 
    "beginningBalance": number,
    "endingBalance": number
  },
  "deposits": [
    {
      "dateCredited": "MM-DD format",
      "description": "full description",
      "amount": number
    }
  ],
  "atmWithdrawals": [
    {
      "tranDate": "MM-DD format",
      "datePosted": "MM-DD format", 
      "description": "full description",
      "amount": negative_number
    }
  ],
  "checksPaid": [
    {
      "datePaid": "MM-DD format",
      "checkNumber": "string",
      "amount": number,
      "referenceNumber": "string"
    }
  ],
  "visaPurchases": [
    {
      "tranDate": "MM-DD format",
      "datePosted": "MM-DD format",
      "description": "full description", 
      "amount": negative_number
    }
  ]
}

CRITICAL REQUIREMENTS:
- Extract ALL transactions with exact amounts, dates, and descriptions
- Use exact date formats (MM-DD like "05-15")
- Negative amounts for withdrawals/debits
- Include complete descriptions and reference numbers
- Return ONLY valid JSON, no additional text

Extract ALL bank statement transaction data from this PDF with complete accuracy."""
        
        # Generate content
        logger.info("Generating AI response...")
        result = model.generate_content([prompt, uploaded_file])
        response = result.text
        logger.info(f"AI Response received (length: {len(response)} chars)")
        
        # Parse JSON response
        import json
        try:
            # Clean response and extract JSON
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            response_text = response_text.strip()
            
            extracted_data = json.loads(response_text)
            logger.info("Successfully parsed JSON response")
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response (first 500 chars): {response[:500]}")
            raise Exception("AI returned invalid JSON format")
            
    except Exception as e:
        logger.error(f"AI extraction error: {str(e)}")
        raise Exception(f"AI extraction failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

# Include Dodo Payments routes
app.include_router(dodo_routes.router)

# Logger already configured above

# Background task for cleaning up expired documents
async def periodic_cleanup():
    """Background task that runs every hour to clean up expired documents"""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await cleanup_expired_documents()
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {str(e)}")
            await asyncio.sleep(3600)  # Wait before retrying

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    global client, db
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[os.environ['DB_NAME']]
        logger.info("Connected to MongoDB successfully")
        
        # Run initial cleanup of expired documents
        deleted_count = await cleanup_expired_documents()
        if deleted_count > 0:
            logger.info(f"Initial cleanup: Deleted {deleted_count} expired documents")
        
        # Start background task for periodic cleanup (runs every hour)
        asyncio.create_task(periodic_cleanup())
        logger.info("Started background task for periodic document cleanup")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
