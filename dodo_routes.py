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
        
        # Get all invoices for this user
        cursor = db.payment_transactions.find({
            "user_id": user_id_str,
            "payment_status": {"$in": ["succeeded", "completed"]},
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
