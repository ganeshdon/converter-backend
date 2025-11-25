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

from dodo_payments import get_dodo_client, get_product_id, get_dodo_api_base_url
from models import PaymentSessionRequest, PaymentSessionResponse
from auth import verify_jwt_token

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

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
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

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
        
        # If subscription is active, update database
        if subscription.status == "active":
            # MongoDB setup
            mongo_client = AsyncIOMotorClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
            db_name = os.getenv("DB_NAME", "test_database")
            db = mongo_client[db_name]
            
            # Get subscription from database
            db_subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
            
            if db_subscription:
                user_id = db_subscription["user_id"]
                plan = normalize_plan_name(db_subscription["plan"])
                
                # Determine pages based on plan (must match SUBSCRIPTION_PACKAGES in server.py)
                pages_limit_map = {
                    "starter": 400,
                    "professional": 1000,
                    "business": 4000,
                    "enterprise": -1  # -1 means unlimited
                }
                pages_limit = pages_limit_map.get(plan, 400)
                pages_remaining = pages_limit if pages_limit != -1 else -1
                
                # Get billing_interval from subscription
                billing_interval = db_subscription.get("billing_interval", "monthly")
                
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
            else:
                return {"status": "not_found", "message": "Subscription not found in database"}
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
            await handle_subscription_active(event_data)
        
        # Handle subscription.renewed event
        elif event_type == "subscription.renewed":
            await handle_subscription_renewed(event_data)
        
        # Handle subscription.on_hold event
        elif event_type == "subscription.on_hold":
            await handle_subscription_on_hold(event_data)
        
        # Handle subscription.cancelled event
        elif event_type == "subscription.cancelled":
            await handle_subscription_cancelled(event_data)
        
        # Handle subscription.failed event
        elif event_type == "subscription.failed":
            await handle_subscription_failed(event_data)
        
        # Handle payment.succeeded event
        elif event_type == "payment.succeeded":
            await handle_payment_succeeded(event_data)
        
        else:
            logger.warning(f"Unhandled webhook event type: {event_type}")
        
        logger.info(f"Successfully processed webhook event: {event_type}")
        return {"status": "success", "event_type": event_type}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
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
            "starter": 400,
            "professional": 1000,
            "business": 4000,
            "enterprise": -1  # -1 means unlimited
        }
        pages_limit = pages_limit_map.get(plan, 400)
        pages_remaining = pages_limit if pages_limit != -1 else -1
        
        # Get billing_interval from subscription
        billing_interval = subscription.get("billing_interval", "monthly")
        
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
                    
                    await db.payment_transactions.insert_one(tx_doc)
                    logger.info(f"Recorded invoice for subscription activation: {subscription_id}, user: {subscription['user_id']}")
            except Exception as e:
                logger.error(f"Failed to record invoice for subscription activation: {e}")


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
            "starter": 400,
            "professional": 1000,
            "business": 4000,
            "enterprise": -1  # -1 means unlimited
        }
        pages_limit = pages_limit_map.get(plan, 400)
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
                
                await db.payment_transactions.insert_one(tx_doc)
                logger.info(f"Recorded invoice for subscription renewal: {subscription_id}, user: {subscription['user_id']}, amount: ${invoice_amount}")
        except Exception as e:
            logger.error(f"Failed to record invoice for subscription renewal: {e}")


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
    
    logger.info(f"Payment succeeded: {payment_id} for subscription {subscription_id}, amount: {amount}")
    logger.info(f"Payment webhook data: {data}")  # Debug log
    
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
            # Still record the transaction but log the issue
            user_id = "unknown"

        # Get amount - use from webhook or calculate from plan
        if not amount and package_id:
            plan_prices = {
                "starter": {"monthly": 15.0, "annual": 12.0},
                "professional": {"monthly": 30.0, "annual": 24.0},
                "business": {"monthly": 50.0, "annual": 40.0},
                "enterprise": {"monthly": 100.0, "annual": 80.0}
            }
            price_map = plan_prices.get(package_id, {})
            amount = price_map.get(billing_interval or "monthly", price_map.get("monthly", 0.0))
            logger.info(f"Calculated amount from plan: {amount} for {package_id} ({billing_interval})")

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
        else:
            await db.payment_transactions.insert_one(tx_doc)
            logger.info(f"Recorded payment transaction for user: {user_id}, tx: {tx_doc['transaction_id']}, amount: ${amount}")
    except Exception as e:
        logger.error(f"Failed to record payment transaction: {e}")
        logger.exception(e)  # Log full traceback


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
                    payments_list = payments_data.get("data", []) if isinstance(payments_data, dict) else (payments_data if isinstance(payments_data, list) else [])
                    
                    logger.info(f"Found {len(payments_list)} payments for subscription {subscription_id}")
                    
                    for payment in payments_list:
                        payment_id = payment.get("id") or payment.get("payment_id")
                        if not payment_id:
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
                        try:
                            invoice_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                            invoice_response = await client.get(invoice_url, headers=headers)
                            
                            invoice_data = None
                            if invoice_response.status_code == 200:
                                # Check if response is JSON or PDF
                                content_type = invoice_response.headers.get("content-type", "")
                                if "application/json" in content_type:
                                    invoice_data = invoice_response.json()
                                else:
                                    # It's a PDF, we'll store the URL
                                    invoice_data = {"pdf_url": invoice_url}
                            
                            # Store payment transaction/invoice in database
                            amount = payment.get("amount", 0.0)
                            if invoice_data and invoice_data.get("amount"):
                                amount = invoice_data.get("amount")
                            
                            tx_doc = {
                                "transaction_id": payment_id,
                                "payment_id": payment_id,
                                "subscription_id": subscription_id,
                                "user_id": user_id_str,
                                "package_id": subscription.get("plan"),
                                "amount": amount,
                                "currency": payment.get("currency", "usd"),
                                "payment_status": payment.get("status", "succeeded"),
                                "subscription_status": subscription.get("status", "active"),
                                "billing_interval": subscription.get("billing_interval", "monthly"),
                                "payment_provider": "dodo",
                                "metadata": {
                                    "synced_from_dodo": True,
                                    "sync_date": datetime.utcnow().isoformat()
                                },
                                "invoice_url": invoice_data.get("invoice_url") if invoice_data else None,
                                "invoice_pdf_url": invoice_data.get("pdf_url") if invoice_data else invoice_url,
                                "invoice_number": invoice_data.get("invoice_number") if invoice_data else None,
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
                raise HTTPException(status_code=500, detail=f"Failed to sync invoices: {str(e)}")
        
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
        invoices = []
        cursor = db.payment_transactions.find(
            {
                "user_id": user_id_str,
                "payment_status": {"$in": ["succeeded", "completed"]}
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
                                payments_list = payments_data.get("data", []) if isinstance(payments_data, dict) else (payments_data if isinstance(payments_data, list) else [])
                                
                                for payment in payments_list:
                                    payment_id = payment.get("id") or payment.get("payment_id")
                                    if payment_id:
                                        try:
                                            invoice_url = f"{dodo_base_url}/invoices/payments/{payment_id}"
                                            invoice_response = await client.get(invoice_url, headers=headers)
                                            
                                            invoice_data = None
                                            if invoice_response.status_code == 200:
                                                content_type = invoice_response.headers.get("content-type", "")
                                                if "application/json" in content_type:
                                                    invoice_data = invoice_response.json()
                                                else:
                                                    invoice_data = {"pdf_url": invoice_url}
                                            
                                            invoices.append({
                                                "invoice_id": payment_id,
                                                "payment_id": payment_id,
                                                "subscription_id": subscription_id,
                                                "amount": payment.get("amount", invoice_data.get("amount", 0.0) if invoice_data else 0.0),
                                                "currency": payment.get("currency", "usd"),
                                                "status": payment.get("status", "succeeded"),
                                                "package_id": subscription.get("plan"),
                                                "billing_interval": subscription.get("billing_interval", "monthly"),
                                                "created_at": payment.get("created_at"),
                                                "updated_at": payment.get("updated_at"),
                                                "invoice_url": invoice_data.get("invoice_url") if invoice_data else None,
                                                "invoice_pdf_url": invoice_data.get("pdf_url") if invoice_data else invoice_url,
                                                "invoice_number": invoice_data.get("invoice_number") if invoice_data else None,
                                                "invoice_date": invoice_data.get("invoice_date") if invoice_data else None
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
        contact_data = {
            "name": data.get("name"),
            "company_name": data.get("company_name"),
            "website": data.get("website", ""),
            "phone": data.get("phone"),
            "email": data.get("email"),
            "message": data.get("message"),
            "submitted_at": datetime.utcnow(),
            "status": "pending"
        }
        
        await db.enterprise_contacts.insert_one(contact_data)
        
        # Send email notification
        try:
            send_enterprise_contact_email(contact_data)
        except Exception as email_error:
            logger.error(f"Failed to send email notification: {str(email_error)}")
            # Don't fail the request if email fails
        
        return {"status": "success", "message": "Your request has been submitted. We'll contact you soon!"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enterprise contact: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process contact form")


def send_enterprise_contact_email(contact_data: dict):
    """
    Send email notification for enterprise contact form
    This is a simple implementation - you may want to use a proper email service
    """
    try:
        # Email content
        subject = f"Enterprise Inquiry from {contact_data['company_name']}"
        body = f"""
New Enterprise Contact Form Submission

Name: {contact_data['name']}
Company: {contact_data['company_name']}
Website: {contact_data.get('website', 'N/A')}
Email: {contact_data['email']}
Phone: {contact_data['phone']}

Message:
{contact_data['message']}

Submitted at: {contact_data['submitted_at']}
"""
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = "noreply@yourbankstatementconverter.com"
        msg['To'] = "info@yourbankstatementconverter.com"
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Note: This is a basic implementation
        # In production, you should use a proper email service like SendGrid, AWS SES, etc.
        logger.info(f"Enterprise contact email prepared for: {contact_data['email']}")
        logger.info(f"Email body:\n{body}")
        
        # For now, just log the email
        # You can implement actual SMTP sending or use an email service
        
    except Exception as e:
        logger.error(f"Error sending enterprise contact email: {str(e)}")
        raise
