Python
import pandas as pd
import numpy as np
import os

def generate_synthetic_data(filepath: str, n_records: int = 10000, n_arms: int = 10):
    """Generates synthetic ad click data with hidden true conversion rates."""
    np.random.seed(42)
    # Define hidden true click-through rates (CTR) for 10 ad variants
    # Ad 4 (index 4) will be the secret "best" ad with a 25% CTR
    true_ctrs = [0.05, 0.11, 0.08, 0.15, 0.25, 0.04, 0.09, 0.12, 0.07, 0.06]
    
    data = []
    for _ in range(n_records):
        user_row = [np.random.binomial(1, ctr) for ctr in true_ctrs]
        data.append(user_row)
        
    df = pd.DataFrame(data, columns=[f'Ad_{i+1}' for i in range(n_arms)])
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset created at {filepath}")

if __name__ == "__main__":
    generate_synthetic_data("../data/Ads_CTR_Optimisation.csv")
