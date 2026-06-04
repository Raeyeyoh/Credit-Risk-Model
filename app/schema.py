from pydantic import BaseModel


class TransactionData(BaseModel):

    Amount: float
    Value: int
    PricingStrategy: int
    FraudResult: int
