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
from datetime import datetime
import uuid
from standardwebhooks.webhooks import Webhook
from motor.motor_asyncio import AsyncIOMotorClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dodo_payments import get_dodo_client, get_product_id
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
                
                # Determine pages based on plan
                pages_limit_map = {
                    "starter": 50,
                    "professional": 200,
                    "business": 500,
                    "enterprise": -1  # -1 means unlimited
                }
                pages_limit = pages_limit_map.get(plan, 50)
                pages_remaining = pages_limit if pages_limit != -1 else -1
                
                # Update user with subscription details
                await db.users.update_one(
                    {"_id": user_id},
                    {
                        "$set": {
                            "subscription_status": "active",
                            "subscription_tier": plan,
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
    """
    try:
        # Get webhook secret from environment
        webhook_secret = os.getenv("DODO_PAYMENTS_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.error("DODO_PAYMENTS_WEBHOOK_SECRET not configured")
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
        
        # Get raw body and headers
        body = (await request.body()).decode('utf-8')
        headers = {
            "webhook-id": request.headers.get("webhook-id"),
            "webhook-signature": request.headers.get("webhook-signature"),
            "webhook-timestamp": request.headers.get("webhook-timestamp")
        }
        
        # Verify webhook signature
        wh = Webhook(webhook_secret)
        try:
            payload = wh.verify(body, headers)
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
        
        # Process event based on type
        event_type = payload.get("type")
        event_data = payload.get("data", {})
        
        logger.info(f"Processing Dodo webhook event: {event_type}")
        
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
        
        return {"status": "success"}
        
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
    
    # Update user's subscription status
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if subscription:
        await db.users.update_one(
            {"_id": subscription["user_id"]},
            {
                "$set": {
                    "subscription_status": "active",
                    "subscription_tier": subscription["plan"],
                    "updated_at": datetime.utcnow()
                }
            }
        )


async def handle_subscription_renewed(data: dict):
    """Handle subscription.renewed event"""
    subscription_id = data.get("subscription_id")
    logger.info(f"Subscription renewed: {subscription_id}")
    
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
    
    logger.info(f"Payment succeeded: {payment_id} for subscription {subscription_id}")
    
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

        # If we have a subscription_id but no user_id, look it up in our DB
        if subscription_id and not user_id:
            sub = await db.subscriptions.find_one({"subscription_id": subscription_id})
            if sub:
                user_id = sub.get("user_id")
                package_id = package_id or sub.get("plan")
                billing_interval = billing_interval or sub.get("billing_interval")
                subscription_status = sub.get("status")

        tx_doc = {
            "transaction_id": payment_id or str(uuid.uuid4()),
            "payment_id": payment_id,
            "subscription_id": subscription_id,
            "user_id": user_id,
            "package_id": package_id,
            "amount": amount,
            "currency": data.get("currency", "usd"),
            "payment_status": data.get("status") or "succeeded",
            "subscription_status": subscription_status or data.get("subscription_status"),
            "billing_interval": billing_interval or data.get("billing_interval"),
            "payment_provider": "dodo",
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        await db.payment_transactions.insert_one(tx_doc)
        logger.info(f"Recorded payment transaction for user: {user_id}, tx: {tx_doc['transaction_id']}")
    except Exception as e:
        logger.error(f"Failed to record payment transaction: {e}")


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
