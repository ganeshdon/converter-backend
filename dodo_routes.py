"""
Dodo Payments Integration Routes
Handles subscription creation, customer portal, and webhooks
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import uuid
from standardwebhooks.webhooks import Webhook
from motor.motor_asyncio import AsyncIOMotorClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import asyncio
import resend

from dodo_payments import get_dodo_client, get_product_id, get_dodo_api_base_url
from models import PaymentSessionRequest, PaymentSessionResponse
from auth import verify_jwt_token

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Email configuration (Resend)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
RESEND_FROM_NAME = os.getenv("RESEND_FROM_NAME", "Bank Statement Converter")
ENTERPRISE_CONTACT_EMAIL = os.getenv("ENTERPRISE_CONTACT_EMAIL", "info@yourbankstatementconverter.com")

# Setup logging
logger = logging.getLogger(__name__)

# Plan normalization: map Dodo plan names to internal subscription tier names
# Use descriptive tier names that work with the app's business logic
PLAN_TO_TIER_MAPPING = {
    "starter": "starter",        # Keep as-is (now in SubscriptionTier enum)
    "professional": "professional",
    "business": "business",
    "enterprise": "enterprise"
}

def normalize_plan_name(plan: str) -> str:
    """Normalize Dodo plan name to consistent internal tier value."""
    return PLAN_TO_TIER_MAPPING.get(plan.lower(), plan.lower())

# Helper function to get current user (duplicated from server.py to avoid circular import)
async def get_current_user(request: Request):
    """Get current user from JWT token or OAuth session token"""
    # MongoDB client for session lookup
    client = AsyncIOMotorClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
    db_name = os.getenv("DB_NAME", "test_database")
    db = client[db_name]
    
    # First try to get session token from cookie
    session_token = request.cookies.get("session_token")
    if session_token:
        session = await db.user_sessions.find_one({"session_token": session_token})
        if session and session.get("expires_at") > datetime.utcnow():
            user = await db.users.find_one({"_id": session["user_id"]})
            if user:
                return {"user_id": user["_id"], "email": user["email"], "name": user.get("name", "")}
    
    # Fallback to Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    
    # Try as session token first
    session = await db.user_sessions.find_one({"session_token": token})
    if session and session.get("expires_at") > datetime.utcnow():
        user = await db.users.find_one({"_id": session["user_id"]})
        if user:
            return {"user_id": user["_id"], "email": user["email"], "name": user.get("name", "")}
    
    # Try as JWT token
    try:
        return verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Create router
router = APIRouter(prefix="/api", tags=["dodo-payments"])

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test_database")
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Get frontend URL from environment variable
# For production, this should be set to your actual domain (e.g., https://yourbankstatementconverter.com)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://yourbankstatementconverter.com")

# Log FRONTEND_URL for debugging
logger.info(f"dodo_routes.py - FRONTEND_URL loaded: {FRONTEND_URL}")

def parse_datetime(date_value):
    """Parse datetime from various formats"""
    if not date_value:
        return datetime.utcnow()
    if isinstance(date_value, datetime):
        return date_value
    if isinstance(date_value, str):
        try:
            # Try ISO format with Z
            if date_value.endswith('Z'):
                return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            # Try ISO format
            return datetime.fromisoformat(date_value)
        except:
            try:
                # Try parsing with strptime
                return datetime.strptime(date_value, "%Y-%m-%dT%H:%M:%S.%fZ")
            except:
                return datetime.utcnow()
    return datetime.utcnow()

@router.post("/dodo/create-subscription")
async def create_dodo_subscription(
    request: PaymentSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Dodo Payments subscription checkout session
    """
    try:
        # Initialize Dodo client
        dodo_client = get_dodo_client()
        
        # Get product ID based on plan and billing interval
        product_id = get_product_id(request.package_id, request.billing_interval)
        
        # Get user details
        user_email = current_user.get("email")
        user_name = current_user.get("name", "")
        user_id = current_user.get("user_id")
        
        logger.info(f"Creating Dodo subscription for user {user_email}, plan: {request.package_id}, interval: {request.billing_interval}")
        logger.info(f"Using return URL: {FRONTEND_URL}/?payment=success")
        
        # Create subscription with payment link
        subscription_response = await dodo_client.subscriptions.create(
            product_id=product_id,
            quantity=1,
            payment_link=True,
            return_url=f"{FRONTEND_URL}/?payment=success",
            customer={
                "email": user_email,
                "name": user_name
            },
            billing={
                "name": user_name,
                "email": user_email,
                "country": "US",        # Default to US, can be made configurable later
                "state": "CA",          # Default to California, can be made configurable later
                "city": "San Francisco", # Default city, can be made configurable later
                "street": "123 Main St", # Default street, can be made configurable later
                "zipcode": "94102"      # Default zipcode, can be made configurable later
            },
            metadata={
                "user_id": user_id,
                "plan": request.package_id,
                "billing_interval": request.billing_interval
            }
        )
        
        # Extract payment link and subscription ID
        payment_link = subscription_response.payment_link
        subscription_id = subscription_response.subscription_id
        
        logger.info(f"Dodo subscription created: {subscription_id}")
        
        # Store initial subscription record
        await db.subscriptions.insert_one({
            "user_id": user_id,
            "subscription_id": subscription_id,
            "plan": request.package_id,
            "billing_interval": request.billing_interval,
            "status": "pending",
            "payment_provider": "dodo",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        return PaymentSessionResponse(
            checkout_url=payment_link,
            session_id=subscription_id
        )
        
    except Exception as e:
        logger.error(f"Error creating Dodo subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")


@router.post("/dodo/create-portal-session")
async def create_dodo_portal_session(current_user: dict = Depends(get_current_user)):
    """
    Create a Dodo Payments customer portal session
    """
    try:
        dodo_client = get_dodo_client()
        user_email = current_user.get("email")
        
        # Get user's subscription to find customer_id
        subscription = await db.subscriptions.find_one({
            "user_id": current_user.get("user_id"),
            "payment_provider": "dodo"
        })
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        customer_id = subscription.get("customer_id")
        if not customer_id:
            raise HTTPException(status_code=404, detail="Customer ID not found in subscription")
        
        # Create customer portal session
        portal_response = await dodo_client.customers.customer_portal.create(
            customer_id=customer_id
        )
        
        return {"portal_url": portal_response.url}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating portal session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create portal session: {str(e)}")


@router.post("/dodo/check-subscription/{subscription_id}")
async def check_subscription_status(subscription_id: str, current_user: dict = Depends(get_current_user)):
    """
    Check subscription status with Dodo Payments and update database
    This is used after payment redirect when webhook might not have fired
    """
    logger.info(f"🔍 CHECK SUBSCRIPTION called for: {subscription_id}")
    logger.info(f"👤 User: {current_user.get('email')}")
    
    try:
        # Get Dodo client
        dodo_client = get_dodo_client()
        logger.info(f"✅ Dodo client initialized")
        
        # Fetch subscription from Dodo
        logger.info(f"📞 Fetching subscription from Dodo API...")

        async def fetch_subscription(client, subscription_id):
            """Try multiple possible Dodo client methods to fetch a subscription."""
            subs = getattr(client, 'subscriptions', None)

            # Try known method names on the subscriptions resource
            candidate_methods = [
                'get', 'retrieve', 'fetch', 'get_subscription', 'get_by_id', 'retrieve_subscription'
            ]

            if subs is not None:
                for name in candidate_methods:
                    fn = getattr(subs, name, None)
                    if callable(fn):
                        logger.info(f"Trying subscriptions.{name}()")
                        try:
                            result = fn(subscription_id)
                            if hasattr(result, '__await__'):
                                return await result
                            return result
                        except TypeError:
                            # try as keyword arg
                            try:
                                result = fn(id=subscription_id)
                                if hasattr(result, '__await__'):
                                    return await result
                                return result
                            except Exception:
                                logger.debug(f"subscriptions.{name} failed with TypeError for id")
                        except Exception as e:
                            logger.debug(f"subscriptions.{name} raised: {e}")

            # Try methods on the client itself
            for name in candidate_methods:
                fn = getattr(client, name, None)
                if callable(fn):
                    logger.info(f"Trying client.{name}()")
                    try:
                        result = fn(subscription_id)
                        if hasattr(result, '__await__'):
                            return await result
                        return result
                    except Exception as e:
                        logger.debug(f"client.{name} raised: {e}")

            raise Exception("Unable to fetch subscription from Dodo client: no supported method found")

        subscription = await fetch_subscription(dodo_client, subscription_id)
        
        logger.info(f"📡 Subscription status from Dodo: {subscription.status}")
        
        # MongoDB setup
        mongo_client = AsyncIOMotorClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        db_name = os.getenv("DB_NAME", "test_database")
        db = mongo_client[db_name]
        
        # Get subscription from database
        db_subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
        
        if not db_subscription:
            return {"status": "not_found", "message": "Subscription not found in database"}
        
        user_id = db_subscription["user_id"]
        plan = normalize_plan_name(db_subscription["plan"])
        billing_interval = db_subscription.get("billing_interval", "monthly")
        
        # Convert user_id to string
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        # If subscription is active, update database
        if subscription.status == "active":
            # Determine pages based on plan (must match SUBSCRIPTION_PACKAGES in server.py)
            pages_limit_map = {
                "starter": 2,
                "professional": 1000,
                "business": 4000,
                "enterprise": -1  # -1 means unlimited
            }
            pages_limit = pages_limit_map.get(plan, 2)
            new_plan_pages = pages_limit if pages_limit != -1 else -1
            
            # Get current user to check for existing subscription and pages
            user = await db.users.find_one({"_id": user_id})
            current_plan = user.get("subscription_tier") if user else None
            current_pages_remaining = user.get("pages_remaining", 0) if user else 0
            current_pages_limit = user.get("pages_limit", 0) if user else 0
            
            # Check if this is an upgrade (different plan) or first subscription
            is_upgrade = current_plan and current_plan != plan and current_plan != "daily_free"
            is_first_subscription = not current_plan or current_plan == "daily_free"
            
            # Calculate new pages_remaining and pages_limit
            if new_plan_pages == -1:
                # Enterprise plan - unlimited (always set to unlimited)
                pages_remaining = -1
                pages_limit = -1
                logger.info(f"Enterprise plan: Setting pages to unlimited")
            elif is_upgrade and current_pages_remaining > 0 and current_pages_remaining != -1:
                # User is upgrading: add new plan's pages to existing pages
                # Only add if current pages are not unlimited (-1)
                pages_remaining = current_pages_remaining + new_plan_pages
                # Also update pages_limit to reflect the total available pages
                pages_limit = current_pages_limit + new_plan_pages
                logger.info(f"Upgrade detected: Adding {new_plan_pages} pages to existing {current_pages_remaining} pages = {pages_remaining} total (limit: {pages_limit})")
            else:
                # First subscription, same plan, or upgrading from unlimited: use new plan's pages
                pages_remaining = new_plan_pages
                pages_limit = new_plan_pages
                logger.info(f"First subscription or same plan: Setting pages to {pages_remaining} (limit: {pages_limit})")
            
            # Update user with subscription details
            await db.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "subscription_status": "active",
                        "subscription_tier": plan,
                        "billing_interval": billing_interval,
                        "pages_limit": pages_limit,
                        "pages_remaining": pages_remaining,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Update subscription status
            await db.subscriptions.update_one(
                {"subscription_id": subscription_id},
                {
                    "$set": {
                        "status": "active",
                        "activated_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Successfully updated subscription for user {user_id}")
            
            return {
                "status": "success",
                "subscription_status": "active",
                "plan": plan,
                "pages_limit": pages_limit,
                "pages_remaining": pages_remaining
            }
        
        # If subscription is failed, record failed payment transaction
        elif subscription.status == "failed":
            logger.error(f"⚠️ Subscription failed: {subscription_id}")
            
            # Update subscription status in database
            await db.subscriptions.update_one(
                {"subscription_id": subscription_id},
                {
                    "$set": {
                        "status": "failed",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Try to fetch payments for this subscription to record failed payment
            try:
                api_key = os.getenv("DODO_PAYMENTS_API_KEY")
                dodo_base_url = get_dodo_api_base_url()
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Fetch payments for this subscription
                    payments_url = f"{dodo_base_url}/payments"
                    params = {"subscription_id": subscription_id}
                    
                    payments_response = await client.get(payments_url, headers=headers, params=params)
                    
                    if payments_response.status_code == 200:
                        payments_data = payments_response.json()
                        # Check "items" first, then "data" as fallback
                        if isinstance(payments_data, dict):
                            payments_list = payments_data.get("items", []) or payments_data.get("data", [])
                        elif isinstance(payments_data, list):
                            payments_list = payments_data
                        else:
                            payments_list = []
                        
                        logger.info(f"Found {len(payments_list)} payments for failed subscription {subscription_id}")
                        
                        # Find failed payments and record them
                        for payment in payments_list:
                            payment_id = payment.get("id") or payment.get("payment_id")
                            payment_status = payment.get("status", "").lower()
                            
                            # Skip payment_method_id (pm_ prefix)
                            if not payment_id or payment_id.startswith("pm_"):
                                continue
                            
                            # Only record if payment status is failed
                            if payment_status == "failed":
                                # Check if transaction already exists
                                existing = await db.payment_transactions.find_one({
                                    "payment_id": payment_id,
                                    "user_id": user_id_str
                                })
                                
                                if not existing:
                                    # Get amount from payment or calculate from plan
                                    amount = payment.get("amount")
                                    if not amount or amount == 0:
                                        # Calculate from plan
                                        plan_prices = {
                                            "starter": {"monthly": 15.0, "annual": 12.0},
                                            "professional": {"monthly": 49.0, "annual": 39.0},
                                            "business": {"monthly": 149.0, "annual": 119.0},
                                            "enterprise": {"monthly": 499.0, "annual": 399.0}
                                        }
                                        amount = plan_prices.get(plan, {}).get(billing_interval, 0.0)
                                    
                                    # Create failed payment transaction record
                                    tx_doc = {
                                        "transaction_id": payment_id or f"failed_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                                        "payment_id": payment_id,
                                        "subscription_id": subscription_id,
                                        "user_id": user_id_str,
                                        "package_id": plan,
                                        "amount": float(amount) if amount else 0.0,
                                        "currency": payment.get("currency", "usd"),
                                        "payment_status": "failed",
                                        "subscription_status": "failed",
                                        "billing_interval": billing_interval,
                                        "payment_provider": "dodo",
                                        "metadata": {
                                            "recorded_from_check_subscription": True,
                                            "subscription_status": "failed"
                                        },
                                        "created_at": parse_datetime(payment.get("created_at")) if payment.get("created_at") else datetime.utcnow(),
                                        "updated_at": datetime.utcnow()
                                    }
                                    
                                    logger.info(f"Recording failed payment transaction: {tx_doc}")
                                    result = await db.payment_transactions.insert_one(tx_doc)
                                    logger.info(f"✅ Recorded failed payment transaction: {payment_id} for user {user_id_str}")
                                    
                                    # Verify it was saved
                                    verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
                                    if verify:
                                        logger.info(f"Verified: Failed payment transaction saved successfully")
                                    else:
                                        logger.error(f"ERROR: Failed payment transaction was not saved!")
                                else:
                                    # Update existing transaction to failed if needed
                                    if existing.get("payment_status") != "failed":
                                        await db.payment_transactions.update_one(
                                            {"payment_id": payment_id, "user_id": user_id_str},
                                            {
                                                "$set": {
                                                    "payment_status": "failed",
                                                    "subscription_status": "failed",
                                                    "updated_at": datetime.utcnow()
                                                }
                                            }
                                        )
                                        logger.info(f"Updated existing transaction to failed: {payment_id}")
                    else:
                        logger.warning(f"Could not fetch payments for failed subscription: {payments_response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching payments for failed subscription: {str(e)}")
                logger.exception(e)
            
            return {
                "status": "failed",
                "message": f"Subscription status: {subscription.status}. Failed payment has been recorded."
            }
        
        # For other statuses, just return the status
        else:
            return {
                "status": subscription.status,
                "message": f"Subscription status: {subscription.status}"
            }
            
    except Exception as e:
        logger.error(f"Error checking subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/dodo")
async def dodo_webhook(request: Request):
    """
    Handle Dodo Payments webhook events
    Webhook URL should be: https://yourdomain.com/api/webhook/dodo
    For local testing, use ngrok: https://your-ngrok-url.ngrok-free.app/api/webhook/dodo
    """
    try:
        # Log incoming webhook request for debugging
        logger.info(f"=== Webhook Received ===")
        logger.info(f"Request from: {request.client.host if request.client else 'unknown'}")
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Request method: {request.method}")
        
        # Get webhook secret from environment
        webhook_secret = os.getenv("DODO_PAYMENTS_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.error("DODO_PAYMENTS_WEBHOOK_SECRET not configured")
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
        
        # Get raw body and headers
        body = (await request.body()).decode('utf-8')
        logger.info(f"Webhook body length: {len(body)} characters")
        
        headers = {
            "webhook-id": request.headers.get("webhook-id"),
            "webhook-signature": request.headers.get("webhook-signature"),
            "webhook-timestamp": request.headers.get("webhook-timestamp")
        }
        
        logger.info(f"Webhook headers received: webhook-id={headers.get('webhook-id')}, timestamp={headers.get('webhook-timestamp')}")
        
        # Check if required headers are present
        if not all([headers.get("webhook-id"), headers.get("webhook-signature"), headers.get("webhook-timestamp")]):
            logger.error(f"Missing webhook headers. Received: {headers}")
            raise HTTPException(status_code=400, detail="Missing required webhook headers")
        
        # Verify webhook signature
        wh = Webhook(webhook_secret)
        try:
            payload = wh.verify(body, headers)
            logger.info(f"Webhook signature verified successfully")
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {str(e)}")
            logger.error(f"Expected secret configured: {'Yes' if webhook_secret else 'No'}")
            raise HTTPException(status_code=400, detail=f"Invalid webhook signature: {str(e)}")
        
        # Process event based on type
        event_type = payload.get("type")
        event_data = payload.get("data", {})
        
        logger.info(f"Processing Dodo webhook event: {event_type}")
        logger.info(f"Event data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}")
        
        # Handle subscription.active event
        if event_type == "subscription.active":
            try:
                await handle_subscription_active(event_data)
                logger.info(f"Successfully handled subscription.active event")
            except Exception as e:
                logger.error(f"Error handling subscription.active: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process subscription.active: {str(e)}")
        
        # Handle subscription.renewed event
        elif event_type == "subscription.renewed":
            try:
                await handle_subscription_renewed(event_data)
                logger.info(f"Successfully handled subscription.renewed event")
            except Exception as e:
                logger.error(f"Error handling subscription.renewed: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process subscription.renewed: {str(e)}")
        
        # Handle subscription.on_hold event
        elif event_type == "subscription.on_hold":
            try:
                await handle_subscription_on_hold(event_data)
                logger.info(f"Successfully handled subscription.on_hold event")
            except Exception as e:
                logger.error(f"Error handling subscription.on_hold: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process subscription.on_hold: {str(e)}")
        
        # Handle subscription.cancelled event
        elif event_type == "subscription.cancelled":
            try:
                await handle_subscription_cancelled(event_data)
                logger.info(f"Successfully handled subscription.cancelled event")
            except Exception as e:
                logger.error(f"Error handling subscription.cancelled: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process subscription.cancelled: {str(e)}")
        
        # Handle subscription.failed event
        elif event_type == "subscription.failed":
            try:
                await handle_subscription_failed(event_data)
                logger.info(f"Successfully handled subscription.failed event")
            except Exception as e:
                logger.error(f"Error handling subscription.failed: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process subscription.failed: {str(e)}")
        
        # Handle payment.succeeded event
        elif event_type == "payment.succeeded":
            try:
                await handle_payment_succeeded(event_data)
                logger.info(f"Successfully handled payment.succeeded event")
            except Exception as e:
                logger.error(f"Error handling payment.succeeded: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process payment.succeeded: {str(e)}")
        
        # Handle payment.failed event
        elif event_type == "payment.failed":
            try:
                await handle_payment_failed(event_data)
                logger.info(f"Successfully handled payment.failed event")
            except Exception as e:
                logger.error(f"Error handling payment.failed: {str(e)}")
                logger.exception(e)
                raise HTTPException(status_code=500, detail=f"Failed to process payment.failed: {str(e)}")
        
        else:
            logger.warning(f"Unhandled webhook event type: {event_type}")
            # Return success for unhandled events so Dodo doesn't retry
            return {"status": "success", "event_type": event_type, "message": "Event type not handled"}
        
        logger.info(f"Successfully processed webhook event: {event_type}")
        return {"status": "success", "event_type": event_type}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        logger.exception(e)  # Log full traceback
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


async def handle_subscription_active(data: dict):
    """Handle subscription.active event"""
    subscription_id = data.get("subscription_id")
    customer_id = data.get("customer_id")
    product_id = data.get("product_id")
    amount = data.get("amount")  # Get amount from webhook if available
    
    logger.info(f"Subscription activated: {subscription_id}")
    
    # Update subscription in database
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "status": "active",
                "customer_id": customer_id,
                "activated_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Update user's subscription status and pages
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if subscription:
        plan = normalize_plan_name(subscription["plan"])
        
        # Determine pages based on plan (must match SUBSCRIPTION_PACKAGES in server.py)
        pages_limit_map = {
            "starter": 2,
            "professional": 1000,
            "business": 4000,
            "enterprise": -1  # -1 means unlimited
        }
        pages_limit = pages_limit_map.get(plan, 2)
        new_plan_pages = pages_limit if pages_limit != -1 else -1
        
        # Get billing_interval from subscription
        billing_interval = subscription.get("billing_interval", "monthly")
        
        # Get current user to check for existing subscription and pages
        user = await db.users.find_one({"_id": subscription["user_id"]})
        current_plan = user.get("subscription_tier") if user else None
        current_pages_remaining = user.get("pages_remaining", 0) if user else 0
        current_pages_limit = user.get("pages_limit", 0) if user else 0
        
        # Check if this is an upgrade (different plan) or first subscription
        is_upgrade = current_plan and current_plan != plan and current_plan != "daily_free"
        is_first_subscription = not current_plan or current_plan == "daily_free"
        
        # Calculate new pages_remaining and pages_limit
        if new_plan_pages == -1:
            # Enterprise plan - unlimited (always set to unlimited)
            pages_remaining = -1
            pages_limit = -1
            logger.info(f"Enterprise plan: Setting pages to unlimited")
        elif is_upgrade and current_pages_remaining > 0 and current_pages_remaining != -1:
            # User is upgrading: add new plan's pages to existing pages
            # Only add if current pages are not unlimited (-1)
            pages_remaining = current_pages_remaining + new_plan_pages
            # Also update pages_limit to reflect the total available pages
            pages_limit = current_pages_limit + new_plan_pages
            logger.info(f"Upgrade detected: Adding {new_plan_pages} pages to existing {current_pages_remaining} pages = {pages_remaining} total (limit: {pages_limit})")
        else:
            # First subscription, same plan, or upgrading from unlimited: use new plan's pages
            pages_remaining = new_plan_pages
            pages_limit = new_plan_pages
            logger.info(f"First subscription or same plan: Setting pages to {pages_remaining} (limit: {pages_limit})")
        
        await db.users.update_one(
            {"_id": subscription["user_id"]},
            {
                "$set": {
                    "subscription_status": "active",
                    "subscription_tier": plan,
                    "billing_interval": billing_interval,
                    "pages_limit": pages_limit,
                    "pages_remaining": pages_remaining,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Updated user {subscription['user_id']} with {pages_remaining} pages for {plan} plan ({billing_interval})")
        
        # Record invoice for initial subscription activation if amount is provided
        # This handles cases where payment.succeeded might not be sent separately
        if amount and subscription.get("user_id"):
            try:
                # Check if invoice already exists for this subscription
                existing_invoice = await db.payment_transactions.find_one({
                    "subscription_id": subscription_id,
                    "payment_status": {"$in": ["succeeded", "completed"]}
                })
                
                if not existing_invoice:
                    # Get plan price from SUBSCRIPTION_PACKAGES (import from server.py or define here)
                    plan_prices = {
                        "starter": {"monthly": 15.0, "annual": 12.0},
                        "professional": {"monthly": 30.0, "annual": 24.0},
                        "business": {"monthly": 50.0, "annual": 40.0},
                        "enterprise": {"monthly": 100.0, "annual": 80.0}
                    }
                    
                    # Use amount from webhook, or calculate from plan
                    invoice_amount = amount
                    if not invoice_amount:
                        price_map = plan_prices.get(plan, {})
                        invoice_amount = price_map.get(billing_interval, price_map.get("monthly", 0.0))
                    
                    # Ensure user_id is stored as string
                    from bson import ObjectId
                    user_id_val = subscription["user_id"]
                    if isinstance(user_id_val, ObjectId):
                        user_id_str = str(user_id_val)
                    else:
                        user_id_str = str(user_id_val)
                    
                    tx_doc = {
                        "transaction_id": f"sub_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                        "payment_id": data.get("payment_id"),
                        "subscription_id": subscription_id,
                        "user_id": user_id_str,  # Store as string
                        "package_id": plan,
                        "amount": invoice_amount,
                        "currency": data.get("currency", "usd"),
                        "payment_status": "succeeded",
                        "subscription_status": "active",
                        "billing_interval": billing_interval,
                        "payment_provider": "dodo",
                        "metadata": data.get("metadata", {}),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    logger.info(f"Inserting invoice for subscription activation: {tx_doc}")
                    result = await db.payment_transactions.insert_one(tx_doc)
                    logger.info(f"Insert result: {result.inserted_id}")
                    logger.info(f"Recorded invoice for subscription activation: {subscription_id}, user: {subscription['user_id']}")
                    
                    # Verify it was saved
                    verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
                    if verify:
                        logger.info(f"Verified: Invoice saved successfully in database")
                    else:
                        logger.error(f"ERROR: Invoice was not saved!")
                        raise Exception("Invoice was not saved to database")
            except Exception as e:
                logger.error(f"Failed to record invoice for subscription activation: {e}")
                logger.exception(e)
                raise  # Re-raise to let webhook handler know it failed


async def handle_subscription_renewed(data: dict):
    """Handle subscription.renewed event"""
    subscription_id = data.get("subscription_id")
    amount = data.get("amount")
    logger.info(f"Subscription renewed: {subscription_id}, amount: {amount}")
    
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "status": "active",
                "last_renewed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Reset user's pages when subscription renews
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if subscription:
        plan = normalize_plan_name(subscription["plan"])
        
        # Determine pages based on plan (must match SUBSCRIPTION_PACKAGES in server.py)
        pages_limit_map = {
            "starter": 2,
            "professional": 1000,
            "business": 4000,
            "enterprise": -1  # -1 means unlimited
        }
        pages_limit = pages_limit_map.get(plan, 2)
        pages_remaining = pages_limit if pages_limit != -1 else -1
        
        # Get billing_interval from subscription (preserve existing if not in subscription)
        billing_interval = subscription.get("billing_interval")
        
        update_data = {
            "pages_limit": pages_limit,
            "pages_remaining": pages_remaining,
            "updated_at": datetime.utcnow()
        }
        
        # Only update billing_interval if it exists in subscription
        if billing_interval:
            update_data["billing_interval"] = billing_interval
        
        await db.users.update_one(
            {"_id": subscription["user_id"]},
            {"$set": update_data}
        )
        
        logger.info(f"Renewed subscription for user {subscription['user_id']} - reset to {pages_remaining} pages for {plan} plan")
        
        # Record invoice for subscription renewal
        try:
            # Check if invoice already exists for this renewal
            existing_invoice = await db.payment_transactions.find_one({
                "subscription_id": subscription_id,
                "created_at": {
                    "$gte": datetime.utcnow() - timedelta(hours=1)  # Within last hour
                },
                "payment_status": {"$in": ["succeeded", "completed"]}
            })
            
            if not existing_invoice:
                # Get plan price
                plan_prices = {
                    "starter": {"monthly": 15.0, "annual": 12.0},
                    "professional": {"monthly": 30.0, "annual": 24.0},
                    "business": {"monthly": 50.0, "annual": 40.0},
                    "enterprise": {"monthly": 100.0, "annual": 80.0}
                }
                
                # Use amount from webhook, or calculate from plan
                invoice_amount = amount
                if not invoice_amount:
                    price_map = plan_prices.get(plan, {})
                    invoice_amount = price_map.get(billing_interval or "monthly", price_map.get("monthly", 0.0))
                
                    # Ensure user_id is stored as string
                    from bson import ObjectId
                    user_id_val = subscription["user_id"]
                    if isinstance(user_id_val, ObjectId):
                        user_id_str = str(user_id_val)
                    else:
                        user_id_str = str(user_id_val)
                    
                    tx_doc = {
                        "transaction_id": f"renew_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                        "payment_id": data.get("payment_id"),
                        "subscription_id": subscription_id,
                        "user_id": user_id_str,  # Store as string
                        "package_id": plan,
                        "amount": invoice_amount,
                        "currency": data.get("currency", "usd"),
                        "payment_status": "succeeded",
                        "subscription_status": "active",
                        "billing_interval": billing_interval,
                        "payment_provider": "dodo",
                        "metadata": data.get("metadata", {}),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                
                    logger.info(f"Inserting invoice for subscription renewal: {tx_doc}")
                    result = await db.payment_transactions.insert_one(tx_doc)
                    logger.info(f"Insert result: {result.inserted_id}")
                    logger.info(f"Recorded invoice for subscription renewal: {subscription_id}, user: {subscription['user_id']}, amount: ${invoice_amount}")
                    
                    # Verify it was saved
                    verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
                    if verify:
                        logger.info(f"Verified: Invoice saved successfully in database")
                    else:
                        logger.error(f"ERROR: Invoice was not saved!")
                        raise Exception("Invoice was not saved to database")
        except Exception as e:
            logger.error(f"Failed to record invoice for subscription renewal: {e}")
            logger.exception(e)
            raise  # Re-raise to let webhook handler know it failed


async def handle_subscription_on_hold(data: dict):
    """Handle subscription.on_hold event"""
    subscription_id = data.get("subscription_id")
    logger.warning(f"Subscription on hold: {subscription_id}")
    
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "status": "on_hold",
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Update user status
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if subscription:
        await db.users.update_one(
            {"_id": subscription["user_id"]},
            {"$set": {"subscription_status": "on_hold"}}
        )


async def handle_subscription_cancelled(data: dict):
    """Handle subscription.cancelled event"""
    subscription_id = data.get("subscription_id")
    logger.info(f"Subscription cancelled: {subscription_id}")
    
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "status": "cancelled",
                "cancelled_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    # Update user status
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if subscription:
        await db.users.update_one(
            {"_id": subscription["user_id"]},
            {"$set": {"subscription_status": "cancelled"}}
        )


async def handle_subscription_failed(data: dict):
    """Handle subscription.failed event"""
    subscription_id = data.get("subscription_id")
    logger.error(f"Subscription failed: {subscription_id}")
    
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "status": "failed",
                "updated_at": datetime.utcnow()
            }
        }
    )


async def handle_payment_succeeded(data: dict):
    """Handle payment.succeeded event"""
    payment_id = data.get("payment_id")
    subscription_id = data.get("subscription_id")
    amount = data.get("amount")
    
    logger.info(f"=== Processing payment.succeeded ===")
    logger.info(f"Payment ID: {payment_id}")
    logger.info(f"Subscription ID: {subscription_id}")
    logger.info(f"Amount: {amount}")
    logger.info(f"Full webhook data: {data}")
    
    # Record payment transaction
    try:
        # Try to find associated user_id from subscription metadata or subscriptions collection
        user_id = None
        package_id = None
        billing_interval = None
        subscription_status = None

        # If the webhook provided metadata with user_id, prefer that
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("user_id"):
            user_id = metadata.get("user_id")
            package_id = metadata.get("package_id") or metadata.get("plan")
            billing_interval = metadata.get("billing_interval")
            logger.info(f"Found user_id from metadata: {user_id}")

        # If we have a subscription_id but no user_id, look it up in our DB
        if subscription_id and not user_id:
            sub = await db.subscriptions.find_one({"subscription_id": subscription_id})
            if sub:
                user_id = sub.get("user_id")
                package_id = package_id or sub.get("plan")
                billing_interval = billing_interval or sub.get("billing_interval")
                subscription_status = sub.get("status")
                logger.info(f"Found user_id from subscription lookup: {user_id}")
            else:
                logger.warning(f"Subscription {subscription_id} not found in database")

        # If still no user_id, try to get it from the subscription object in webhook
        if not user_id:
            subscription_obj = data.get("subscription")
            if isinstance(subscription_obj, dict):
                sub_id = subscription_obj.get("subscription_id") or subscription_id
                if sub_id:
                    sub = await db.subscriptions.find_one({"subscription_id": sub_id})
                    if sub:
                        user_id = sub.get("user_id")
                        package_id = package_id or sub.get("plan")
                        billing_interval = billing_interval or sub.get("billing_interval")
                        logger.info(f"Found user_id from subscription object: {user_id}")

        if not user_id:
            logger.error(f"Could not find user_id for payment {payment_id}. Subscription: {subscription_id}, Metadata: {metadata}")
            logger.error(f"Full webhook data structure: {data}")
            # Try one more time - check if subscription_id exists at all
            if subscription_id:
                all_subs = await db.subscriptions.find({"subscription_id": subscription_id}).to_list(length=10)
                logger.error(f"Found {len(all_subs)} subscriptions with subscription_id {subscription_id}")
                if all_subs:
                    logger.error(f"Subscription data: {all_subs[0]}")
            raise ValueError(f"Could not find user_id for payment {payment_id}. Cannot create invoice without user_id.")

        # Get amount - use from webhook or calculate from plan
        if not amount:
            if package_id:
                plan_prices = {
                    "starter": {"monthly": 15.0, "annual": 12.0},
                    "professional": {"monthly": 30.0, "annual": 24.0},
                    "business": {"monthly": 50.0, "annual": 40.0},
                    "enterprise": {"monthly": 100.0, "annual": 80.0}
                }
                price_map = plan_prices.get(package_id, {})
                amount = price_map.get(billing_interval or "monthly", price_map.get("monthly", 0.0))
                logger.info(f"Calculated amount from plan: {amount} for {package_id} ({billing_interval})")
            elif subscription_id:
                # Try to get amount from subscription
                sub = await db.subscriptions.find_one({"subscription_id": subscription_id})
                if sub:
                    plan = sub.get("plan")
                    billing = sub.get("billing_interval", "monthly")
                    plan_prices = {
                        "starter": {"monthly": 15.0, "annual": 12.0},
                        "professional": {"monthly": 30.0, "annual": 24.0},
                        "business": {"monthly": 50.0, "annual": 40.0},
                        "enterprise": {"monthly": 100.0, "annual": 80.0}
                    }
                    price_map = plan_prices.get(plan, {})
                    amount = price_map.get(billing, price_map.get("monthly", 0.0))
                    package_id = plan  # Set package_id from subscription
                    billing_interval = billing  # Set billing_interval from subscription
                    logger.info(f"Calculated amount from subscription: {amount} for {plan} ({billing})")
        
        if not amount or amount == 0:
            logger.warning(f"Amount is 0 or None for payment {payment_id}. This might indicate an issue.")
            # Don't fail, but log warning

        # Ensure user_id is stored as string for consistent querying
        # Convert ObjectId to string if needed
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        tx_doc = {
            "transaction_id": payment_id or f"pay_{subscription_id}_{int(datetime.utcnow().timestamp())}",
            "payment_id": payment_id,
            "subscription_id": subscription_id,
            "user_id": user_id_str,  # Store as string for consistent querying
            "package_id": package_id,
            "amount": amount or 0.0,
            "currency": data.get("currency", "usd"),
            "payment_status": data.get("status") or "succeeded",
            "subscription_status": subscription_status or data.get("subscription_status"),
            "billing_interval": billing_interval or data.get("billing_interval"),
            "payment_provider": "dodo",
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Check if transaction already exists to avoid duplicates
        existing = await db.payment_transactions.find_one({
            "payment_id": payment_id,
            "subscription_id": subscription_id
        })
        
        if existing:
            logger.info(f"Payment transaction already exists: {payment_id}")
            logger.info(f"Existing transaction: {existing}")
        else:
            logger.info(f"Inserting payment transaction: {tx_doc}")
            result = await db.payment_transactions.insert_one(tx_doc)
            logger.info(f"Insert result: {result.inserted_id}")
            logger.info(f"Successfully recorded payment transaction for user: {user_id_str}, tx: {tx_doc['transaction_id']}, amount: ${amount}")
            
            # Verify it was saved
            verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
            if verify:
                logger.info(f"Verified: Invoice saved successfully in database")
            else:
                logger.error(f"ERROR: Invoice was not saved! Insert returned {result.inserted_id} but cannot find it in DB")
                raise Exception("Invoice was not saved to database")
    except Exception as e:
        logger.error(f"Failed to record payment transaction: {e}")
        logger.exception(e)  # Log full traceback
        raise  # Re-raise to let webhook handler know it failed


async def handle_payment_failed(data: dict):
    """Handle payment.failed event"""
    payment_id = data.get("payment_id")
    subscription_id = data.get("subscription_id")
    amount = data.get("amount")
    
    logger.error(f"=== Processing payment.failed ===")
    logger.error(f"Payment ID: {payment_id}")
    logger.error(f"Subscription ID: {subscription_id}")
    logger.error(f"Amount: {amount}")
    logger.error(f"Full webhook data: {data}")
    
    # Record failed payment transaction
    try:
        # Try to find associated user_id from subscription metadata or subscriptions collection
        user_id = None
        package_id = None
        billing_interval = None
        subscription_status = None

        # If the webhook provided metadata with user_id, prefer that
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("user_id"):
            user_id = metadata.get("user_id")
            package_id = metadata.get("package_id") or metadata.get("plan")
            billing_interval = metadata.get("billing_interval")
            logger.info(f"Found user_id from metadata: {user_id}")

        # If we have a subscription_id but no user_id, look it up in our DB
        if subscription_id and not user_id:
            sub = await db.subscriptions.find_one({"subscription_id": subscription_id})
            if sub:
                user_id = sub.get("user_id")
                package_id = package_id or sub.get("plan")
                billing_interval = billing_interval or sub.get("billing_interval")
                subscription_status = sub.get("status")
                logger.info(f"Found user_id from subscription lookup: {user_id}")
            else:
                logger.warning(f"Subscription {subscription_id} not found in database")

        if not user_id:
            logger.error(f"Cannot record failed payment: No user_id found for payment {payment_id}")
            return

        # Convert user_id to string
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)

        # Check if transaction already exists
        existing = await db.payment_transactions.find_one({
            "payment_id": payment_id,
            "user_id": user_id_str
        })
        
        if existing:
            # Update existing transaction to failed status
            await db.payment_transactions.update_one(
                {"payment_id": payment_id, "user_id": user_id_str},
                {
                    "$set": {
                        "payment_status": "failed",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            logger.info(f"Updated existing payment transaction to failed: {payment_id}")
        else:
            # Create new failed payment transaction record
            tx_doc = {
                "transaction_id": payment_id or f"failed_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                "payment_id": payment_id,
                "subscription_id": subscription_id,
                "user_id": user_id_str,
                "package_id": package_id,
                "amount": float(amount) if amount else 0.0,
                "currency": data.get("currency", "usd"),
                "payment_status": "failed",
                "subscription_status": subscription_status or "failed",
                "billing_interval": billing_interval,
                "payment_provider": "dodo",
                "metadata": data.get("metadata", {}),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            logger.info(f"Inserting failed payment transaction: {tx_doc}")
            result = await db.payment_transactions.insert_one(tx_doc)
            logger.info(f"Insert result: {result.inserted_id}")
            logger.info(f"Recorded failed payment transaction for user: {user_id_str}, payment: {payment_id}")
            
            # Verify it was saved
            verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
            if verify:
                logger.info(f"Verified: Failed payment transaction saved successfully in database")
            else:
                logger.error(f"ERROR: Failed payment transaction was not saved!")
    except Exception as e:
        logger.error(f"Failed to record failed payment transaction: {e}")
        logger.exception(e)
        raise


@router.get("/dodo/webhook-test")
async def test_webhook_endpoint():
    """
    Test endpoint to verify webhook URL is accessible
    Returns success if endpoint is reachable
    """
    return {
        "status": "success",
        "message": "Webhook endpoint is accessible",
        "endpoint": "/api/webhook/dodo",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/dodo/fetch-and-save-invoice")
async def fetch_and_save_invoice_after_payment(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch invoice from Dodo API after payment completion and save to database
    This is called directly after payment success, not relying on webhooks
    Accepts subscription_id as query parameter: ?subscription_id=sub_xxx
    """
    try:
        # Get subscription_id from query parameters
        subscription_id = request.query_params.get("subscription_id")
        
        if not subscription_id:
            raise HTTPException(status_code=400, detail="subscription_id query parameter is required")
        
        user_id = current_user.get("user_id")
        
        # Convert user_id to string
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        # Get subscription from database
        subscription = await db.subscriptions.find_one({
            "subscription_id": subscription_id
        })
        
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        # Verify subscription belongs to user
        sub_user_id = subscription.get("user_id")
        if isinstance(sub_user_id, ObjectId):
            sub_user_id_str = str(sub_user_id)
        else:
            sub_user_id_str = str(sub_user_id)
        
        if sub_user_id_str != user_id_str:
            raise HTTPException(status_code=403, detail="Subscription does not belong to user")
        
        api_key = os.getenv("DODO_PAYMENTS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DODO_PAYMENTS_API_KEY not configured")
        
        dodo_base_url = get_dodo_api_base_url()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"📄 FETCH INVOICE ENDPOINT CALLED - subscription: {subscription_id}, user: {user_id_str}")
        logger.info(f"📋 Subscription data: plan={subscription.get('plan')}, billing={subscription.get('billing_interval')}, status={subscription.get('status')}")
        
        saved_count = 0
        payments_list = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Method 1: Try to get subscription details from Dodo API (might have payment info)
            try:
                subscription_url = f"{dodo_base_url}/subscriptions/{subscription_id}"
                subscription_response = await client.get(subscription_url, headers=headers)
                
                if subscription_response.status_code == 200:
                    subscription_data = subscription_response.json()
                    logger.info(f"Got subscription data from Dodo API")
                    
                    # Check if subscription has payment information embedded
                    if isinstance(subscription_data, dict):
                        data_obj = subscription_data.get("data", subscription_data)
                        # Only use actual payment IDs (pay_ prefix), NOT payment_method_id (pm_ prefix)
                        payment_id = data_obj.get("payment_id") or data_obj.get("last_payment_id")
                        
                        # Validate that it's a payment ID, not a payment method ID
                        if payment_id and not payment_id.startswith("pm_"):
                            # Create a payment-like object from subscription
                            # Get amount from subscription, convert from cents if needed
                            sub_amount = data_obj.get("recurring_pre_tax_amount", 0)
                            if sub_amount and sub_amount > 100:
                                sub_amount = sub_amount / 100
                            
                            payments_list.append({
                                "id": payment_id,
                                "payment_id": payment_id,
                                "subscription_id": subscription_id,
                                "amount": sub_amount,
                                "currency": data_obj.get("currency", "usd"),
                                "status": "succeeded",
                                "created_at": data_obj.get("created_at") or data_obj.get("previous_billing_date")
                            })
                            logger.info(f"Found payment_id from subscription: {payment_id}, amount: {sub_amount}")
                        elif payment_id and payment_id.startswith("pm_"):
                            logger.warning(f"Ignoring payment_method_id (pm_): {payment_id}. Need actual payment_id (pay_)")
            except Exception as e:
                logger.warning(f"Could not fetch subscription details: {str(e)}")
            
            # Method 2: Get payments by subscription_id (FIX: Check "items" not "data")
            if not payments_list:
                try:
                    payments_url = f"{dodo_base_url}/payments"
                    params = {"subscription_id": subscription_id}
                    
                    payments_response = await client.get(payments_url, headers=headers, params=params)
                    
                    if payments_response.status_code == 200:
                        payments_data = payments_response.json()
                        # FIX: Check "items" first, then "data" as fallback
                        if isinstance(payments_data, dict):
                            payments_list = payments_data.get("items", []) or payments_data.get("data", [])
                        elif isinstance(payments_data, list):
                            payments_list = payments_data
                        else:
                            payments_list = []
                        
                        logger.info(f"Found {len(payments_list)} payments from payments API")
                    elif payments_response.status_code != 200:
                        logger.warning(f"Payments API returned {payments_response.status_code}: {payments_response.text}")
                except Exception as e:
                    logger.warning(f"Could not fetch payments by subscription_id: {str(e)}")
            
            # Method 3: Try by customer_id if available
            if not payments_list and subscription.get("customer_id"):
                try:
                    customer_id = subscription.get("customer_id")
                    payments_url = f"{dodo_base_url}/payments"
                    params = {"customer_id": customer_id}
                    
                    payments_response = await client.get(payments_url, headers=headers, params=params)
                    
                    if payments_response.status_code == 200:
                        payments_data = payments_response.json()
                        if isinstance(payments_data, dict):
                            customer_payments = payments_data.get("items", []) or payments_data.get("data", [])
                        elif isinstance(payments_data, list):
                            customer_payments = payments_data
                        else:
                            customer_payments = []
                        
                        # Filter to only payments for this subscription
                        payments_list = [p for p in customer_payments if p.get("subscription_id") == subscription_id]
                        logger.info(f"Found {len(payments_list)} payments from customer_id")
                except Exception as e:
                    logger.warning(f"Could not fetch payments by customer_id: {str(e)}")
            
            # Method 4: If still no payments, create invoice from subscription data
            if not payments_list:
                logger.info(f"No payments found via API, creating invoice from subscription data")
                
                # Check if invoice already exists
                existing = await db.payment_transactions.find_one({
                    "subscription_id": subscription_id,
                    "user_id": user_id_str,
                    "payment_status": {"$in": ["succeeded", "completed"]}
                })
                
                if existing:
                    logger.info(f"Invoice already exists for subscription {subscription_id}")
                    return {
                        "status": "success",
                        "message": "Invoice already exists",
                        "invoices_saved": 0,
                        "subscription_id": subscription_id
                    }
                
                # Calculate amount from subscription
                subscription_amount = subscription.get("recurring_pre_tax_amount")
                plan = subscription.get("plan")
                billing = subscription.get("billing_interval", "monthly")
                
                logger.info(f"💰 Calculating amount - plan: {plan}, billing: {billing}, recurring_pre_tax_amount: {subscription_amount}")
                
                if subscription_amount and subscription_amount > 100:
                    # Convert from cents to dollars
                    amount = subscription_amount / 100
                    logger.info(f"💰 Using amount from subscription: ${amount}")
                else:
                    # Calculate from plan
                    plan_prices = {
                        "starter": {"monthly": 15.0, "annual": 12.0},
                        "professional": {"monthly": 30.0, "annual": 24.0},
                        "business": {"monthly": 50.0, "annual": 40.0},
                        "enterprise": {"monthly": 100.0, "annual": 80.0}
                    }
                    amount = plan_prices.get(plan, {}).get(billing, 0.0)
                    logger.info(f"💰 Calculated amount from plan: ${amount}")
                
                # Create invoice record from subscription
                tx_doc = {
                    "transaction_id": f"sub_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                    "payment_id": None,  # Will be updated when payment appears in API
                    "subscription_id": subscription_id,
                    "user_id": user_id_str,
                    "package_id": subscription.get("plan"),
                    "amount": float(amount) if amount else 0.0,
                    "currency": subscription.get("currency", "usd").lower(),
                    "payment_status": "succeeded",
                    "subscription_status": subscription.get("status", "active"),
                    "billing_interval": subscription.get("billing_interval", "monthly"),
                    "payment_provider": "dodo",
                    "invoice_url": None,
                    "invoice_pdf_url": None,
                    "invoice_number": None,
                    "invoice_date": None,
                    "metadata": {
                        "created_from_subscription": True,
                        "created_date": datetime.utcnow().isoformat(),
                        "note": "Created from subscription data - payment_id will be updated when available"
                    },
                    "created_at": subscription.get("created_at") or subscription.get("activated_at") or datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                # Save to database
                logger.info(f"💾 Saving invoice to database for subscription {subscription_id}, amount: {amount}")
                result = await db.payment_transactions.insert_one(tx_doc)
                logger.info(f"✅ Invoice saved to database with _id: {result.inserted_id}")
                logger.info(f"Created invoice from subscription data, invoice ID: {result.inserted_id}")
                
                # Verify it was saved
                verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
                if verify:
                    saved_count = 1
                    logger.info(f"Verified: Invoice saved successfully from subscription data")
                    return {
                        "status": "success",
                        "message": "Invoice created from subscription data (payment will be linked when available)",
                        "invoices_saved": 1,
                        "subscription_id": subscription_id
                    }
                else:
                    logger.error(f"ERROR: Invoice was not saved")
                    raise HTTPException(status_code=500, detail="Failed to save invoice to database")
            
            # Step 2: For each payment, fetch invoice and save to DB
            for payment in payments_list:
                payment_id = payment.get("id") or payment.get("payment_id")
                if not payment_id:
                    continue
                
                # Skip payment_method_id (pm_ prefix) - we need actual payment IDs
                if payment_id.startswith("pm_"):
                    logger.warning(f"Skipping payment_method_id: {payment_id}. Need actual payment_id (pay_ prefix)")
                    continue
                
                # Check if invoice already exists
                existing = await db.payment_transactions.find_one({
                    "payment_id": payment_id,
                    "subscription_id": subscription_id
                })
                
                if existing:
                    logger.info(f"Invoice already exists for payment {payment_id}")
                    continue
                
                # Fetch invoice from Dodo API - THIS IS THE KEY API CALL
                invoice_data = None
                invoice_pdf_url = None
                invoice_url = None
                invoice_number = None
                invoice_date = None
                
                # First, check if payment object itself has invoice information
                if payment.get("invoice_url") or payment.get("invoice_pdf_url"):
                    invoice_pdf_url = payment.get("invoice_pdf_url") or payment.get("invoice_url")
                    invoice_url = invoice_pdf_url
                    logger.info(f"Found invoice URL in payment object: {invoice_pdf_url}")
                
                # Try to fetch from Dodo API
                if not invoice_pdf_url:
                    try:
                        invoice_api_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                        logger.info(f"Fetching invoice from Dodo API: {invoice_api_url}")
                        invoice_response = await client.get(invoice_api_url, headers=headers, follow_redirects=True)
                        
                        if invoice_response.status_code == 200:
                            content_type = invoice_response.headers.get("content-type", "")
                            logger.info(f"Invoice API response content-type: {content_type}")
                            
                            if "application/json" in content_type:
                                invoice_data = invoice_response.json()
                                logger.info(f"Invoice API JSON response: {invoice_data}")
                                
                                # Handle nested data structure
                                if isinstance(invoice_data, dict) and "data" in invoice_data:
                                    invoice_data = invoice_data["data"]
                                
                                # Try multiple possible fields for invoice URL
                                invoice_pdf_url = (
                                    invoice_data.get("pdf_url") or 
                                    invoice_data.get("invoice_pdf_url") or
                                    invoice_data.get("invoice_url") or 
                                    invoice_data.get("url") or
                                    invoice_data.get("download_url") or
                                    invoice_data.get("file_url")
                                )
                                
                                invoice_url = invoice_data.get("invoice_url") or invoice_pdf_url
                                invoice_number = invoice_data.get("invoice_number") or invoice_data.get("number")
                                invoice_date = invoice_data.get("invoice_date") or invoice_data.get("date") or invoice_data.get("created_at")
                                
                                if invoice_pdf_url:
                                    logger.info(f"✅ Found invoice URL from API response: {invoice_pdf_url}")
                                else:
                                    logger.warning(f"⚠️ No invoice URL found in API response, using constructed URL")
                                    invoice_pdf_url = invoice_api_url
                                    invoice_url = invoice_api_url
                            else:
                                # It's a PDF or redirect, use the final URL after redirects
                                invoice_pdf_url = str(invoice_response.url)
                                invoice_url = invoice_pdf_url
                                invoice_data = {"pdf_url": invoice_pdf_url}
                                logger.info(f"✅ Invoice is PDF/redirect, URL: {invoice_pdf_url}")
                        elif invoice_response.status_code == 404:
                            logger.warning(f"⚠️ Invoice not found for payment {payment_id} (404). Payment may not have invoice yet.")
                            # Don't save a broken URL - leave it as None
                            invoice_pdf_url = None
                            invoice_url = None
                        else:
                            logger.warning(f"⚠️ Could not fetch invoice for payment {payment_id}: {invoice_response.status_code} - {invoice_response.text[:200]}")
                            # Don't save a broken URL - leave it as None
                            invoice_pdf_url = None
                            invoice_url = None
                    except Exception as api_error:
                        logger.error(f"❌ Error fetching invoice from API: {str(api_error)}")
                        # Don't save a broken URL
                        invoice_pdf_url = None
                        invoice_url = None
                
                # Prepare invoice document (only if we have invoice_pdf_url or want to save anyway)
                try:
                    amount = payment.get("amount", 0.0)
                    if invoice_data and invoice_data.get("amount"):
                        amount = invoice_data.get("amount")
                    
                    # Convert amount from cents to dollars if needed
                    if amount and amount > 100:  # Likely in cents (changed from 1000 to 100)
                        amount = amount / 100
                    
                    # Fallback: If amount is still 0 or missing, calculate from subscription plan
                    if not amount or amount == 0:
                        plan = subscription.get("plan")
                        billing = subscription.get("billing_interval", "monthly")
                        plan_prices = {
                            "starter": {"monthly": 15.0, "annual": 12.0},
                            "professional": {"monthly": 30.0, "annual": 24.0},
                            "business": {"monthly": 50.0, "annual": 40.0},
                            "enterprise": {"monthly": 100.0, "annual": 80.0}
                        }
                        amount = plan_prices.get(plan, {}).get(billing, 0.0)
                        logger.info(f"💰 Amount was 0, calculated from plan: ${amount} for {plan} ({billing})")
                    
                    tx_doc = {
                        "transaction_id": payment_id,
                        "payment_id": payment_id,
                        "subscription_id": subscription_id,
                        "user_id": user_id_str,
                        "package_id": subscription.get("plan"),
                        "amount": float(amount) if amount else 0.0,
                        "currency": payment.get("currency", subscription.get("currency", "usd")).lower(),
                        "payment_status": payment.get("status", "succeeded"),
                        "subscription_status": subscription.get("status", "active"),
                        "billing_interval": subscription.get("billing_interval", "monthly"),
                        "payment_provider": "dodo",
                        "invoice_url": invoice_url,
                        "invoice_pdf_url": invoice_pdf_url,
                        "invoice_number": invoice_number,
                        "invoice_date": invoice_date,
                        "metadata": {
                            "fetched_from_api": True,
                            "fetch_date": datetime.utcnow().isoformat(),
                            "payment_data": payment,
                            "invoice_fetched": True
                        },
                        "created_at": parse_datetime(payment.get("created_at")),
                        "updated_at": datetime.utcnow()
                    }
                    
                    # Save to database
                    result = await db.payment_transactions.insert_one(tx_doc)
                    logger.info(f"Saved invoice for payment {payment_id}, invoice ID: {result.inserted_id}")
                    
                    # Verify it was saved
                    verify = await db.payment_transactions.find_one({"_id": result.inserted_id})
                    if verify:
                        saved_count += 1
                        logger.info(f"Verified: Invoice saved successfully for payment {payment_id}")
                    else:
                        logger.error(f"ERROR: Invoice was not saved for payment {payment_id}")
                
                except Exception as invoice_error:
                    logger.error(f"Error processing invoice for payment {payment_id}: {str(invoice_error)}")
                    logger.exception(invoice_error)
                    # Continue with next payment
                    continue
            
            return {
                "status": "success",
                "message": f"Fetched and saved {saved_count} invoice(s)",
                "invoices_saved": saved_count,
                "subscription_id": subscription_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching and saving invoice: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch invoice: {str(e)}")


@router.post("/dodo/sync-invoices")
async def sync_invoices_from_dodo(current_user: dict = Depends(get_current_user)):
    """
    Sync invoices from Dodo Payments API for the current user's subscription
    This fetches payments/invoices from Dodo and stores them in the database
    """
    try:
        user_id = current_user.get("user_id")
        
        # Convert user_id to string if it's ObjectId
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        api_key = os.getenv("DODO_PAYMENTS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DODO_PAYMENTS_API_KEY not configured")
        
        dodo_base_url = get_dodo_api_base_url()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get user's subscription
        subscription = await db.subscriptions.find_one({
            "user_id": user_id_str,
            "payment_provider": "dodo"
        })
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found for user")
        
        subscription_id = subscription.get("subscription_id")
        customer_id = subscription.get("customer_id")
        
        if not subscription_id:
            raise HTTPException(status_code=400, detail="Subscription ID not found")
        
        synced_count = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try to fetch payments by subscription_id
            # Method 1: Try to get payments for subscription
            try:
                payments_url = f"{dodo_base_url}/payments"
                params = {"subscription_id": subscription_id}
                if customer_id:
                    params["customer_id"] = customer_id
                
                payments_response = await client.get(payments_url, headers=headers, params=params)
                
                if payments_response.status_code == 200:
                    payments_data = payments_response.json()
                    # FIX: Check "items" first, then "data" as fallback
                    if isinstance(payments_data, dict):
                        payments_list = payments_data.get("items", []) or payments_data.get("data", [])
                    elif isinstance(payments_data, list):
                        payments_list = payments_data
                    else:
                        payments_list = []
                    
                    logger.info(f"Found {len(payments_list)} payments for subscription {subscription_id}")
                    
                    for payment in payments_list:
                        payment_id = payment.get("id") or payment.get("payment_id")
                        if not payment_id:
                            continue
                        
                        # Skip payment_method_id (pm_ prefix) - we need actual payment IDs
                        if payment_id.startswith("pm_"):
                            logger.warning(f"Skipping payment_method_id: {payment_id}. Need actual payment_id (pay_ prefix)")
                            continue
                        
                        # Check if we already have this payment in DB
                        existing = await db.payment_transactions.find_one({
                            "payment_id": payment_id,
                            "subscription_id": subscription_id
                        })
                        
                        if existing:
                            logger.info(f"Payment {payment_id} already exists in database")
                            continue
                        
                        # Fetch invoice for this payment
                        invoice_data = None
                        invoice_pdf_url = None
                        invoice_url = None
                        invoice_number = None
                        invoice_date = None
                        
                        # First, check if payment object itself has invoice information
                        if payment.get("invoice_url") or payment.get("invoice_pdf_url"):
                            invoice_pdf_url = payment.get("invoice_pdf_url") or payment.get("invoice_url")
                            invoice_url = invoice_pdf_url
                            logger.info(f"Found invoice URL in payment object: {invoice_pdf_url}")
                        
                        # Try to fetch from Dodo API
                        if not invoice_pdf_url:
                            try:
                                invoice_api_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                                logger.info(f"Fetching invoice from Dodo API: {invoice_api_url}")
                                invoice_response = await client.get(invoice_api_url, headers=headers, follow_redirects=True)
                                
                                if invoice_response.status_code == 200:
                                    content_type = invoice_response.headers.get("content-type", "")
                                    
                                    if "application/json" in content_type:
                                        invoice_data = invoice_response.json()
                                        # Handle nested data structure
                                        if isinstance(invoice_data, dict) and "data" in invoice_data:
                                            invoice_data = invoice_data["data"]
                                        
                                        # Try multiple possible fields for invoice URL
                                        invoice_pdf_url = (
                                            invoice_data.get("pdf_url") or 
                                            invoice_data.get("invoice_pdf_url") or
                                            invoice_data.get("invoice_url") or 
                                            invoice_data.get("url") or
                                            invoice_data.get("download_url") or
                                            invoice_data.get("file_url")
                                        )
                                        
                                        invoice_url = invoice_data.get("invoice_url") or invoice_pdf_url
                                        invoice_number = invoice_data.get("invoice_number") or invoice_data.get("number")
                                        invoice_date = invoice_data.get("invoice_date") or invoice_data.get("date") or invoice_data.get("created_at")
                                    else:
                                        # It's a PDF or redirect, use the final URL after redirects
                                        invoice_pdf_url = str(invoice_response.url)
                                        invoice_url = invoice_pdf_url
                                        invoice_data = {"pdf_url": invoice_pdf_url}
                                elif invoice_response.status_code == 404:
                                    logger.warning(f"Invoice not found for payment {payment_id} (404)")
                                    invoice_pdf_url = None
                                    invoice_url = None
                                else:
                                    logger.warning(f"Could not fetch invoice for payment {payment_id}: {invoice_response.status_code}")
                                    invoice_pdf_url = None
                                    invoice_url = None
                            except Exception as api_error:
                                logger.error(f"Error fetching invoice from API: {str(api_error)}")
                                invoice_pdf_url = None
                                invoice_url = None
                            
                            # Store payment transaction/invoice in database
                            try:
                                amount = payment.get("amount", 0.0)
                                if invoice_data and invoice_data.get("amount"):
                                    amount = invoice_data.get("amount")
                                
                                # Convert amount from cents to dollars if needed
                                if amount and amount > 100:  # Likely in cents (changed from 1000 to 100)
                                    amount = amount / 100
                                
                                # Fallback: If amount is still 0 or missing, calculate from subscription plan
                                if not amount or amount == 0:
                                    plan = subscription.get("plan")
                                    billing = subscription.get("billing_interval", "monthly")
                                    plan_prices = {
                                        "starter": {"monthly": 15.0, "annual": 12.0},
                                        "professional": {"monthly": 30.0, "annual": 24.0},
                                        "business": {"monthly": 50.0, "annual": 40.0},
                                        "enterprise": {"monthly": 100.0, "annual": 80.0}
                                    }
                                    amount = plan_prices.get(plan, {}).get(billing, 0.0)
                                    logger.info(f"💰 Amount was 0, calculated from plan: ${amount} for {plan} ({billing})")
                                
                                tx_doc = {
                                    "transaction_id": payment_id,
                                    "payment_id": payment_id,
                                    "subscription_id": subscription_id,
                                    "user_id": user_id_str,
                                    "package_id": subscription.get("plan"),
                                    "amount": float(amount) if amount else 0.0,
                                    "currency": payment.get("currency", subscription.get("currency", "usd")).lower(),
                                    "payment_status": payment.get("status", "succeeded"),
                                    "subscription_status": subscription.get("status", "active"),
                                    "billing_interval": subscription.get("billing_interval", "monthly"),
                                    "payment_provider": "dodo",
                                    "metadata": {
                                        "synced_from_dodo": True,
                                        "sync_date": datetime.utcnow().isoformat()
                                    },
                                    "invoice_url": invoice_url,
                                    "invoice_pdf_url": invoice_pdf_url,
                                    "invoice_number": invoice_number,
                                    "invoice_date": invoice_date,
                                    "created_at": parse_datetime(payment.get("created_at")),
                                    "updated_at": datetime.utcnow()
                                }
                                
                                await db.payment_transactions.insert_one(tx_doc)
                                synced_count += 1
                                logger.info(f"Synced payment {payment_id} for subscription {subscription_id}")
                            
                            except Exception as invoice_error:
                                logger.warning(f"Could not fetch invoice for payment {payment_id}: {str(invoice_error)}")
                            # Still store the payment even if invoice fetch fails
                            try:
                                tx_doc = {
                                    "transaction_id": payment_id,
                                    "payment_id": payment_id,
                                    "subscription_id": subscription_id,
                                    "user_id": user_id_str,
                                    "package_id": subscription.get("plan"),
                                    "amount": payment.get("amount", 0.0),
                                    "currency": payment.get("currency", "usd"),
                                    "payment_status": payment.get("status", "succeeded"),
                                    "subscription_status": subscription.get("status", "active"),
                                    "billing_interval": subscription.get("billing_interval", "monthly"),
                                    "payment_provider": "dodo",
                                    "metadata": {
                                        "synced_from_dodo": True,
                                        "sync_date": datetime.utcnow().isoformat(),
                                        "invoice_fetch_error": str(invoice_error)
                                    },
                                    "created_at": parse_datetime(payment.get("created_at")),
                                    "updated_at": datetime.utcnow()
                                }
                                await db.payment_transactions.insert_one(tx_doc)
                                synced_count += 1
                            except Exception as e:
                                logger.error(f"Failed to store payment {payment_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error fetching payments from Dodo API: {str(e)}")
                # Don't raise, try fallback instead
            
            # If no payments found, create invoice from subscription data
            if synced_count == 0:
                logger.info(f"No payments found via API, creating invoice from subscription data in sync")
                
                # Check if invoice already exists
                existing = await db.payment_transactions.find_one({
                    "subscription_id": subscription_id,
                    "user_id": user_id_str,
                    "payment_status": {"$in": ["succeeded", "completed"]}
                })
                
                if not existing:
                    # Calculate amount from subscription
                    subscription_amount = subscription.get("recurring_pre_tax_amount")
                    if subscription_amount and subscription_amount > 100:
                        amount = subscription_amount / 100
                    else:
                        plan = subscription.get("plan")
                        billing = subscription.get("billing_interval", "monthly")
                        plan_prices = {
                            "starter": {"monthly": 15.0, "annual": 12.0},
                            "professional": {"monthly": 30.0, "annual": 24.0},
                            "business": {"monthly": 50.0, "annual": 40.0},
                            "enterprise": {"monthly": 100.0, "annual": 80.0}
                        }
                        amount = plan_prices.get(plan, {}).get(billing, 0.0)
                    
                    tx_doc = {
                        "transaction_id": f"sub_{subscription_id}_{int(datetime.utcnow().timestamp())}",
                        "payment_id": None,
                        "subscription_id": subscription_id,
                        "user_id": user_id_str,
                        "package_id": subscription.get("plan"),
                        "amount": float(amount) if amount else 0.0,
                        "currency": subscription.get("currency", "usd").lower(),
                        "payment_status": "succeeded",
                        "subscription_status": subscription.get("status", "active"),
                        "billing_interval": subscription.get("billing_interval", "monthly"),
                        "payment_provider": "dodo",
                        "invoice_url": None,
                        "invoice_pdf_url": None,
                        "invoice_number": None,
                        "invoice_date": None,
                        "metadata": {
                            "created_from_subscription": True,
                            "created_date": datetime.utcnow().isoformat(),
                            "synced": True
                        },
                        "created_at": subscription.get("created_at") or subscription.get("activated_at") or datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    
                    await db.payment_transactions.insert_one(tx_doc)
                    synced_count = 1
                    logger.info(f"Created invoice from subscription data during sync")
        
        return {
            "status": "success",
            "message": f"Synced {synced_count} invoice(s) from Dodo Payments",
            "synced_count": synced_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to sync invoices: {str(e)}")


@router.get("/dodo/invoices")
async def get_customer_invoices(current_user: dict = Depends(get_current_user)):
    """
    Get invoices for the current user
    Uses official Dodo Payments API: https://test.dodopayments.com/invoices/payments/{payment_id}
    Returns invoices from both payment_transactions collection and Dodo Payments API
    """
    try:
        user_id = current_user.get("user_id")
        
        # Convert user_id to string if it's ObjectId
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        # Fetch invoices from payment_transactions collection
        # Include all payment statuses (succeeded, completed, failed, pending, cancelled)
        invoices = []
        cursor = db.payment_transactions.find(
            {
                "user_id": user_id_str,
                "payment_status": {"$in": ["succeeded", "completed", "failed", "pending", "cancelled"]}
            }
        ).sort("created_at", -1)  # Sort by most recent first
        
        async for invoice in cursor:
            invoices.append({
                "invoice_id": invoice.get("transaction_id") or invoice.get("payment_id"),
                "payment_id": invoice.get("payment_id"),
                "subscription_id": invoice.get("subscription_id"),
                "amount": invoice.get("amount", 0.0),
                "currency": invoice.get("currency", "usd"),
                "status": invoice.get("payment_status", "succeeded"),
                "package_id": invoice.get("package_id"),
                "billing_interval": invoice.get("billing_interval"),
                "created_at": invoice.get("created_at").isoformat() if invoice.get("created_at") else None,
                "updated_at": invoice.get("updated_at").isoformat() if invoice.get("updated_at") else None,
                "invoice_url": invoice.get("invoice_url"),
                "invoice_pdf_url": invoice.get("invoice_pdf_url"),
                "invoice_number": invoice.get("invoice_number"),
                "invoice_date": invoice.get("invoice_date")
            })
        
        # If no invoices found in DB, try to fetch from Dodo API
        if len(invoices) == 0:
            try:
                api_key = os.getenv("DODO_PAYMENTS_API_KEY")
                dodo_base_url = get_dodo_api_base_url()
                
                subscription = await db.subscriptions.find_one({
                    "user_id": user_id_str,
                    "payment_provider": "dodo"
                })
                
                if subscription and api_key:
                    subscription_id = subscription.get("subscription_id")
                    customer_id = subscription.get("customer_id")
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        # Try to get payments for subscription
                        try:
                            payments_url = f"{dodo_base_url}/payments"
                            params = {"subscription_id": subscription_id}
                            if customer_id:
                                params["customer_id"] = customer_id
                            
                            payments_response = await client.get(payments_url, headers=headers, params=params)
                            
                            if payments_response.status_code == 200:
                                payments_data = payments_response.json()
                                # FIX: Check "items" first, then "data" as fallback
                                if isinstance(payments_data, dict):
                                    payments_list = payments_data.get("items", []) or payments_data.get("data", [])
                                elif isinstance(payments_data, list):
                                    payments_list = payments_data
                                else:
                                    payments_list = []
                                
                                for payment in payments_list:
                                    payment_id = payment.get("id") or payment.get("payment_id")
                                    # Skip payment_method_id (pm_ prefix) - we need actual payment IDs
                                    if not payment_id or payment_id.startswith("pm_"):
                                        if payment_id and payment_id.startswith("pm_"):
                                            logger.warning(f"Skipping payment_method_id in get_customer_invoices: {payment_id}")
                                        continue
                                    if payment_id:
                                        try:
                                            invoice_api_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                                            invoice_response = await client.get(invoice_api_url, headers=headers, follow_redirects=True)
                                            
                                            invoice_data = None
                                            invoice_pdf_url = None
                                            invoice_url = None
                                            invoice_number = None
                                            invoice_date = None
                                            
                                            if invoice_response.status_code == 200:
                                                content_type = invoice_response.headers.get("content-type", "")
                                                if "application/json" in content_type:
                                                    invoice_data = invoice_response.json()
                                                    # Handle nested data structure
                                                    if isinstance(invoice_data, dict) and "data" in invoice_data:
                                                        invoice_data = invoice_data["data"]
                                                    
                                                    # Try multiple possible fields for invoice URL
                                                    invoice_pdf_url = (
                                                        invoice_data.get("pdf_url") or 
                                                        invoice_data.get("invoice_pdf_url") or
                                                        invoice_data.get("invoice_url") or 
                                                        invoice_data.get("url") or
                                                        invoice_data.get("download_url") or
                                                        invoice_data.get("file_url")
                                                    )
                                                    
                                                    invoice_url = invoice_data.get("invoice_url") or invoice_pdf_url
                                                    invoice_number = invoice_data.get("invoice_number") or invoice_data.get("number")
                                                    invoice_date = invoice_data.get("invoice_date") or invoice_data.get("date") or invoice_data.get("created_at")
                                                else:
                                                    # It's a PDF or redirect, use the final URL after redirects
                                                    invoice_pdf_url = str(invoice_response.url)
                                                    invoice_url = invoice_pdf_url
                                                    invoice_data = {"pdf_url": invoice_pdf_url}
                                            
                                            # Calculate amount with fallback
                                            invoice_amount = payment.get("amount", 0.0)
                                            if invoice_data and invoice_data.get("amount"):
                                                invoice_amount = invoice_data.get("amount")
                                            
                                            # Convert from cents if needed
                                            if invoice_amount and invoice_amount > 100:
                                                invoice_amount = invoice_amount / 100
                                            
                                            # Fallback: Calculate from subscription plan if amount is 0
                                            if not invoice_amount or invoice_amount == 0:
                                                plan = subscription.get("plan")
                                                billing = subscription.get("billing_interval", "monthly")
                                                plan_prices = {
                                                    "starter": {"monthly": 15.0, "annual": 12.0},
                                                    "professional": {"monthly": 30.0, "annual": 24.0},
                                                    "business": {"monthly": 50.0, "annual": 40.0},
                                                    "enterprise": {"monthly": 100.0, "annual": 80.0}
                                                }
                                                invoice_amount = plan_prices.get(plan, {}).get(billing, 0.0)
                                            
                                            invoices.append({
                                                "invoice_id": payment_id,
                                                "payment_id": payment_id,
                                                "subscription_id": subscription_id,
                                                "amount": float(invoice_amount) if invoice_amount else 0.0,
                                                "currency": payment.get("currency", "usd"),
                                                "status": payment.get("status", "succeeded"),
                                                "package_id": subscription.get("plan"),
                                                "billing_interval": subscription.get("billing_interval", "monthly"),
                                                "created_at": payment.get("created_at"),
                                                "updated_at": payment.get("updated_at"),
                                                "invoice_url": invoice_url,
                                                "invoice_pdf_url": invoice_pdf_url,
                                                "invoice_number": invoice_number,
                                                "invoice_date": invoice_date
                                            })
                                        except Exception as e:
                                            logger.warning(f"Could not fetch invoice for payment {payment_id}: {str(e)}")
                        except Exception as e:
                            logger.warning(f"Could not fetch payments from Dodo API: {str(e)}")
            except Exception as e:
                logger.warning(f"Error fetching invoices from Dodo API: {str(e)}")
        
        # Sort all invoices by created_at (most recent first)
        invoices.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        
        return {
            "invoices": invoices,
            "total": len(invoices)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch invoices: {str(e)}")


@router.post("/dodo/fix-invoice-urls")
async def fix_invoice_urls(current_user: dict = Depends(get_current_user)):
    """
    Fix invoice URLs for existing invoices that may have broken or incorrect URLs
    Re-fetches invoice URLs from Dodo API for all user's invoices
    """
    try:
        user_id = current_user.get("user_id")
        
        # Convert user_id to string
        from bson import ObjectId
        if isinstance(user_id, ObjectId):
            user_id_str = str(user_id)
        else:
            user_id_str = str(user_id)
        
        api_key = os.getenv("DODO_PAYMENTS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DODO_PAYMENTS_API_KEY not configured")
        
        dodo_base_url = get_dodo_api_base_url()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get all invoices for this user (include all statuses to fix URLs for failed payments too)
        cursor = db.payment_transactions.find({
            "user_id": user_id_str,
            "payment_status": {"$in": ["succeeded", "completed", "failed", "pending", "cancelled"]},
            "payment_id": {"$ne": None}  # Only invoices with payment_id
        })
        
        fixed_count = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            async for invoice in cursor:
                payment_id = invoice.get("payment_id")
                if not payment_id:
                    continue
                
                # Skip if invoice URL already exists and looks valid
                existing_url = invoice.get("invoice_pdf_url") or invoice.get("invoice_url")
                if existing_url and "dodopayments.com" in existing_url:
                    # Check if URL is accessible
                    try:
                        check_response = await client.head(existing_url, headers=headers, timeout=5.0)
                        if check_response.status_code == 200:
                            logger.info(f"Invoice URL for {payment_id} is valid, skipping")
                            continue
                    except:
                        pass  # URL is broken, continue to fix it
                
                # Try to fetch correct invoice URL
                try:
                    invoice_api_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                    logger.info(f"Fixing invoice URL for payment {payment_id}")
                    invoice_response = await client.get(invoice_api_url, headers=headers, follow_redirects=True)
                    
                    if invoice_response.status_code == 200:
                        content_type = invoice_response.headers.get("content-type", "")
                        
                        if "application/json" in content_type:
                            invoice_data = invoice_response.json()
                            if isinstance(invoice_data, dict) and "data" in invoice_data:
                                invoice_data = invoice_data["data"]
                            
                            invoice_pdf_url = (
                                invoice_data.get("pdf_url") or 
                                invoice_data.get("invoice_pdf_url") or
                                invoice_data.get("invoice_url") or 
                                invoice_data.get("url") or
                                invoice_data.get("download_url") or
                                invoice_data.get("file_url")
                            )
                            
                            invoice_url = invoice_data.get("invoice_url") or invoice_pdf_url
                            invoice_number = invoice_data.get("invoice_number") or invoice_data.get("number")
                            invoice_date = invoice_data.get("invoice_date") or invoice_data.get("date")
                        else:
                            # It's a PDF or redirect
                            invoice_pdf_url = str(invoice_response.url)
                            invoice_url = invoice_pdf_url
                            invoice_number = None
                            invoice_date = None
                        
                        if invoice_pdf_url:
                            # Update the invoice in database
                            await db.payment_transactions.update_one(
                                {"_id": invoice.get("_id")},
                                {
                                    "$set": {
                                        "invoice_url": invoice_url,
                                        "invoice_pdf_url": invoice_pdf_url,
                                        "invoice_number": invoice_number,
                                        "invoice_date": invoice_date,
                                        "updated_at": datetime.utcnow()
                                    }
                                }
                            )
                            fixed_count += 1
                            logger.info(f"✅ Fixed invoice URL for payment {payment_id}: {invoice_pdf_url}")
                        else:
                            logger.warning(f"⚠️ No invoice URL found in API response for {payment_id}")
                    else:
                        logger.warning(f"⚠️ Could not fetch invoice for payment {payment_id}: {invoice_response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Error fixing invoice URL for {payment_id}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Fixed {fixed_count} invoice URL(s)",
            "fixed_count": fixed_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fixing invoice URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fix invoice URLs: {str(e)}")


@router.post("/enterprise-contact")
async def enterprise_contact(request: Request):
    """
    Handle Enterprise tier contact form submissions
    """
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = ["name", "company_name", "email", "phone", "message"]
        for field in required_fields:
            if not data.get(field):
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Store in database
        submitted_at = datetime.utcnow()
        contact_data = {
            "name": data.get("name"),
            "company_name": data.get("company_name"),
            "website": data.get("website", ""),
            "phone": data.get("phone"),
            "email": data.get("email"),
            "message": data.get("message"),
            "submitted_at": submitted_at,
            "submitted_at_str": submitted_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "pending"
        }
        
        await db.enterprise_contacts.insert_one(contact_data)
        
        # Send email notification
        email_sent = False
        email_error_msg = None
        try:
            email_sent = await send_enterprise_contact_email(contact_data)
            if not email_sent:
                email_error_msg = "Email sending failed. Please check server logs for details."
                logger.error(f"Enterprise contact email failed to send for {contact_data['email']}")
        except Exception as email_error:
            email_error_msg = str(email_error)
            logger.error(f"Failed to send email notification: {str(email_error)}")
            logger.exception(email_error)
        
        # Return success even if email fails (data is stored), but log the issue
        response_message = "Your request has been submitted. We'll contact you soon!"
        if not email_sent:
            logger.warning(f"Enterprise contact form submitted but email notification failed. Contact: {contact_data['email']}, Company: {contact_data['company_name']}")
        
        return {
            "status": "success", 
            "message": response_message,
            "email_sent": email_sent,
            "email_error": email_error_msg if not email_sent else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enterprise contact: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process contact form")


async def send_enterprise_contact_email(contact_data: dict):
    """
    Send email notification for enterprise contact form using Resend
    """
    try:
        # Validate recipient email address
        if not ENTERPRISE_CONTACT_EMAIL or '@' not in ENTERPRISE_CONTACT_EMAIL:
            logger.error(f"Invalid ENTERPRISE_CONTACT_EMAIL: {ENTERPRISE_CONTACT_EMAIL}")
            return False
        
        # If Resend API key is not configured, log the email instead
        if not RESEND_API_KEY:
            logger.warning(f"Resend API key not configured. Enterprise contact form submission from {contact_data['email']}")
            logger.info(f"To enable email sending, configure RESEND_API_KEY environment variable")
            logger.info(f"Email would be sent to: {ENTERPRISE_CONTACT_EMAIL}")
            logger.info(f"Email content:\n{format_enterprise_contact_email_body(contact_data)}")
            return False
        
        # Validate API key format
        if not RESEND_API_KEY.startswith('re_'):
            logger.error(f"Invalid Resend API key format. API key should start with 're_'. Current key starts with: {RESEND_API_KEY[:3] if len(RESEND_API_KEY) >= 3 else 'N/A'}")
            return False
        
        # Log configuration
        logger.info(f"Attempting to send enterprise contact email to {ENTERPRISE_CONTACT_EMAIL}")
        logger.info(f"Using FROM email: {RESEND_FROM_EMAIL}")
        logger.info(f"Resend API key configured: Yes (starts with {RESEND_API_KEY[:5]}...)")
        
        # Set Resend API key
        resend.api_key = RESEND_API_KEY
        
        # Email content
        subject = f"Enterprise Inquiry from {contact_data['company_name']}"
        html_body = format_enterprise_contact_email_html(contact_data)
        text_body = format_enterprise_contact_email_body(contact_data)
        
        # Send email using Resend
        def send_sync():
            try:
                params = {
                    "from": f"{RESEND_FROM_NAME} <{RESEND_FROM_EMAIL}>",
                    "to": [ENTERPRISE_CONTACT_EMAIL],
                    "reply_to": contact_data.get('email'),  # Allow replies to go to the submitter
                    "subject": subject,
                    "html": html_body,
                    "text": text_body,
                }
                
                logger.info(f"Sending email via Resend with params: from={RESEND_FROM_EMAIL}, to={ENTERPRISE_CONTACT_EMAIL}")
                result = resend.Emails.send(params)
                
                # Log the full result for debugging
                logger.info(f"Resend API response: {result}")
                
                # Resend returns a dict with 'id' if successful
                if result and isinstance(result, dict):
                    if result.get('id'):
                        logger.info(f"Resend email sent successfully. Email ID: {result.get('id')}")
                        logger.info(f"Email sent to: {ENTERPRISE_CONTACT_EMAIL}, from: {RESEND_FROM_EMAIL}")
                        return True
                    elif result.get('error'):
                        error_msg = result.get('error', {}).get('message', 'Unknown error')
                        error_type = result.get('error', {}).get('type', 'Unknown')
                        logger.error(f"Resend API error: {error_type} - {error_msg}")
                        logger.error(f"Full error response: {result}")
                        # Check for specific error types
                        if 'domain' in error_msg.lower() or 'not verified' in error_msg.lower():
                            logger.error(f"Domain verification issue. Check if {RESEND_FROM_EMAIL.split('@')[1]} is verified in Resend")
                        elif 'unauthorized' in error_msg.lower() or '401' in str(result):
                            logger.error("API key authorization issue. Check RESEND_API_KEY")
                        return False
                    else:
                        logger.error(f"Resend returned unexpected response (no 'id' or 'error' field): {result}")
                        return False
                elif result is None:
                    logger.error("Resend API returned None - this should not happen")
                    return False
                else:
                    logger.error(f"Resend returned unexpected response type: {type(result)}, value: {result}")
                    return False
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.error(f"Exception sending email via Resend: {error_type}: {error_msg}")
                logger.exception(e)
                
                # Check for common error patterns
                if 'unauthorized' in error_msg.lower() or '401' in error_msg:
                    logger.error("Possible issue: Invalid Resend API key or API key not authorized")
                elif 'domain' in error_msg.lower() or 'from' in error_msg.lower() or 'testing emails' in error_msg.lower() or 'verify a domain' in error_msg.lower():
                    logger.error(f"DOMAIN VERIFICATION REQUIRED: FROM email ({RESEND_FROM_EMAIL}) is not verified in Resend")
                    logger.error("SOLUTION: 1) Go to https://resend.com/domains and verify your domain")
                    logger.error(f"         2) Update RESEND_FROM_EMAIL in .env to use your verified domain (e.g., noreply@yourbankstatementconverter.com)")
                    logger.error(f"         3) Current FROM email '{RESEND_FROM_EMAIL}' can only send to your account email")
                elif 'rate limit' in error_msg.lower():
                    logger.error("Possible issue: Rate limit exceeded for Resend API")
                
                return False
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, send_sync)
        
        if success:
            logger.info(f"Enterprise contact email sent successfully to {ENTERPRISE_CONTACT_EMAIL} via Resend")
        else:
            logger.error(f"Failed to send enterprise contact email to {ENTERPRISE_CONTACT_EMAIL} via Resend")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in send_enterprise_contact_email: {str(e)}")
        logger.exception(e)
        return False


def format_enterprise_contact_email_html(contact_data: dict) -> str:
    """Format enterprise contact email as HTML"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0;">New Enterprise Inquiry</h1>
        </div>
        <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
            <h2 style="color: #667eea; margin-top: 0;">Contact Information</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; font-weight: bold; width: 150px;">Name:</td>
                    <td style="padding: 8px 0;">{contact_data['name']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Company:</td>
                    <td style="padding: 8px 0;">{contact_data['company_name']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Email:</td>
                    <td style="padding: 8px 0;"><a href="mailto:{contact_data['email']}">{contact_data['email']}</a></td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Phone:</td>
                    <td style="padding: 8px 0;"><a href="tel:{contact_data['phone']}">{contact_data['phone']}</a></td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Website:</td>
                    <td style="padding: 8px 0;">{contact_data.get('website', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Submitted:</td>
                    <td style="padding: 8px 0;">{contact_data.get('submitted_at_str', contact_data.get('submitted_at', datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(contact_data.get('submitted_at'), datetime) else str(contact_data.get('submitted_at', 'N/A')))}
                </tr>
            </table>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <h2 style="color: #667eea;">Message</h2>
            <div style="background: white; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0;">
                <p style="white-space: pre-wrap; margin: 0;">{contact_data['message']}</p>
            </div>
        </div>
    </body>
    </html>
    """


def format_enterprise_contact_email_body(contact_data: dict) -> str:
    """Format enterprise contact email as plain text"""
    submitted_at_str = contact_data.get('submitted_at_str', 
                                       contact_data.get('submitted_at', datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC") 
                                       if isinstance(contact_data.get('submitted_at'), datetime) 
                                       else str(contact_data.get('submitted_at', 'N/A')))
    return f"""
New Enterprise Contact Form Submission

Contact Information:
Name: {contact_data['name']}
Company: {contact_data['company_name']}
Email: {contact_data['email']}
Phone: {contact_data['phone']}
Website: {contact_data.get('website', 'N/A')}
Submitted at: {submitted_at_str}

Message:
{contact_data['message']}
"""
