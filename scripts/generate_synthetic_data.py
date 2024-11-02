import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(num_records=1000):
    np.random.seed(0)
    user_ids = np.random.randint(1000, 1100, num_records)
    transaction_amounts = np.round(np.random.normal(50, 25, num_records), 2)
    transaction_dates = [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(num_records)]
    fraud_flags = np.random.choice([0, 1], num_records, p=[0.95, 0.05])

    data = {
        "user_id": user_ids,
        "transaction_amount": transaction_amounts,
        "transaction_date": transaction_dates,
        "is_fraud": fraud_flags
    }
    
    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_transactions.csv", index=False)
    print("Synthetic data generated at data/synthetic_transactions.csv")

if __name__ == "__main__":
    generate_data()
