import os
from dodopayments import AsyncDodoPayments

# Product IDs for subscription tiers
PRODUCT_IDS = {
    "starter_monthly": "pdt_3De8vbY7VeAL5N14U1u9v",
    "professional_monthly": "pdt_OkGQdqYOj4gbCELn16bFV",
    "business_monthly": "pdt_764AlaHI1InsTdvpe9sKF",
    "starter_annual": "pdt_qfKfB9Vld2Ju87p8gbw4u",
    "professional_annual": "pdt_K3LvXF5i6e4hu3bdTbfxw",
    "business_annual": "pdt_C3OegZyvFVMfXUeaL67oI",
}

def get_dodo_api_base_url():
    """Get Dodo Payments API base URL based on environment"""
    environment = os.getenv("DODO_PAYMENTS_ENVIRONMENT", "test_mode")
    if environment == "live_mode":
        return "https://live.dodopayments.com"
    else:
        return "https://test.dodopayments.com"

def get_dodo_client():
    """Initialize and return Async Dodo Payments client"""
    api_key = os.getenv("DODO_PAYMENTS_API_KEY")
    environment = os.getenv("DODO_PAYMENTS_ENVIRONMENT", "test_mode")
    
    if not api_key:
        raise ValueError("DODO_PAYMENTS_API_KEY environment variable is required")
    
    # Environment must be either 'test_mode' or 'live_mode'
    return AsyncDodoPayments(
        bearer_token=api_key,
        environment=environment
    )

def get_product_id(plan: str, billing_cycle: str) -> str:
    """Get product ID based on plan and billing cycle"""
    key = f"{plan.lower()}_{billing_cycle.lower()}"
    product_id = PRODUCT_IDS.get(key)
    
    if not product_id:
        raise ValueError(f"Invalid plan or billing cycle: {plan}, {billing_cycle}")
    
    return product_id
