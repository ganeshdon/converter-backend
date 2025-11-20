from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    # Legacy / app-internal tiers
    DAILY_FREE = "daily_free"
    BASIC = "basic"
    PREMIUM = "premium"
    PLATINUM = "platinum"
    ENTERPRISE = "enterprise"

    # Dodo Payments plan names (keep for compatibility with payment webhook and check flows)
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"

class BillingInterval(str, Enum):
    MONTHLY = "monthly"
    ANNUAL = "annual"

class UserSignup(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    subscription_tier: SubscriptionTier
    pages_remaining: int
    pages_limit: int
    billing_cycle_start: Optional[datetime]
    daily_reset_time: Optional[datetime]
    language_preference: str = "en"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    file_size: int
    page_count: int
    pages_deducted: int
    conversion_date: datetime
    download_count: int
    status: str

class PagesCheckRequest(BaseModel):
    page_count: int

class PagesCheckResponse(BaseModel):
    can_convert: bool
    pages_remaining: int
    pages_limit: int
    reset_date: Optional[datetime]
    message: str

class SubscriptionPlan(BaseModel):
    tier: SubscriptionTier
    name: str
    price_monthly: float
    price_annual: float
    pages_limit: int
    features: List[str]
    is_popular: bool = False

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    language_preference: Optional[str] = None

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

# Google OAuth Models
class GoogleUserData(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    session_token: str

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime

# Anonymous Conversion Tracking Models
class AnonymousConversionCheck(BaseModel):
    browser_fingerprint: str
    ip_address: Optional[str] = None

class AnonymousConversionResponse(BaseModel):
    can_convert: bool
    conversions_used: int
    conversions_limit: int = 1
    message: str
    requires_signup: bool = False

class AnonymousConversionRecord(BaseModel):
    browser_fingerprint: str
    ip_address: Optional[str] = None
    filename: str
    file_size: int
    page_count: int
    conversion_date: datetime
    user_agent: Optional[str] = None

# Stripe Payment Models
class SubscriptionPackage(BaseModel):
    package_id: str
    name: str
    price: float
    pages_limit: int
    billing_interval: str  # monthly or annual
    features: List[str] = []

class PaymentSessionRequest(BaseModel):
    package_id: str
    billing_interval: str = "monthly"  # monthly or annual

class PaymentSessionResponse(BaseModel):
    checkout_url: str
    session_id: str

class PaymentTransaction(BaseModel):
    transaction_id: str
    session_id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    package_id: str
    amount: float
    currency: str = "usd"
    payment_status: str  # pending, completed, failed, cancelled
    subscription_status: str  # active, inactive, cancelled
    billing_interval: str
    created_at: datetime
    updated_at: datetime
    stripe_customer_id: Optional[str] = None
    metadata: Optional[dict] = None

class WebhookEventResponse(BaseModel):
    event_type: str
    event_id: str
    session_id: str
    payment_status: str
    metadata: Optional[dict] = None