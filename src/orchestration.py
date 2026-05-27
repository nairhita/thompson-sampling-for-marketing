Python
import pandas as pd
import matplotlib.pyplot as plt
import os
from thompson_sampling import ThompsonSamplingAgent
from data_generator import generate_synthetic_data

def run_simulation(data_path: str):
    # Ensure data exists; if not, generate it
    if not os.path.exists(data_path):
        print("Kaggle data not found. Generating synthetic equivalent...")
        generate_synthetic_data(data_path)

    df = pd.read_csv(data_path)
    n_records, n_arms = df.shape
    
    agent = ThompsonSamplingAgent(n_arms=n_arms)
    
    total_reward = 0
    ads_selected = []
    
    print(f"Starting simulation for {n_records} user impressions...")
    
    for user_round in range(n_records):
        # 1. Agent chooses an ad variant to show the user
        chosen_ad = agent.select_arm()
        ads_selected.append(chosen_ad)
        
        # 2. We check the dataset to see if this user actually clicked this specific ad
        reward = df.values[user_round, chosen_ad]
        
        # 3. Agent updates its Beta distributions
        agent.update(chosen_ad, reward)
        total_reward += reward
        
    print(f"Simulation Complete. Total Clicks Generated: {total_reward}")
    
    # 4. Generate Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(ads_selected, bins=n_arms, align='left', rwidth=0.7, color='#007acc')
    plt.title('Thompson Sampling: Ad Selection Frequency', fontsize=14)
    plt.xlabel('Ad Variant Index', fontsize=12)
    plt.ylabel('Number of Times Selected', fontsize=12)
    plt.xticks(range(n_arms), [f'Ad {i+1}' for i in range(n_arms)])
    
    output_img = "../ad_selection_histogram.png"
    plt.savefig(output_img)
    print(f"Saved results visualization to {output_img}")

if __name__ == "__main__":
    run_simulation("../data/Ads_CTR_Optimisation.csv")
