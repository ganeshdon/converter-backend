import os
from dodopayments import AsyncDodoPayments

# Product IDs for subscription tiers
PRODUCT_IDS = {
    "starter_monthly": "pdt_tfooh1hgdtu28iMdXSRl3",
    "professional_monthly": "pdt_q0ZUGAq69LZ4vNUZGYjFS",
    "business_monthly": "pdt_FInGMoMySf6lYrxia8qgq",
    "starter_annual": "pdt_uO0U9F22GbAd87C6S7CzG",
    "professional_annual": "pdt_eewWmwQNJ26eMyvhRoG62",
    "business_annual": "pdt_rQiqTXDkiarEO0HW4WrIS",
}

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
