# Dynamic Marketing Optimization using Thompson Sampling

Traditional A/B testing wastes money by forcing equal traffic to poor-performing campaigns while the test is running. This project demonstrates a **Multi-Armed Bandit** approach using **Thompson Sampling** to dynamically allocate marketing budget to the best-performing ad variants in real-time.

## The Objective
Given 10 different ad creatives and 10,000 user sessions, find the ad with the highest Click-Through Rate (CTR) while minimizing "regret" (the lost clicks from showing inferior ads during the exploration phase).

## How it Works
The algorithm uses Bayesian Inference. Each ad's unknown conversion rate is modeled as a Beta distribution. 
- **Exploration:** It samples from these distributions to occasionally try uncertain ads.
- **Exploitation:** As it gathers more clicks, the distributions narrow, and the algorithm aggressively routes traffic to the winning ad.

## Running the Project
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the orchestration script: `cd src && python orchestration.py`

*(Note: The script will automatically generate synthetic Kaggle-style data if the original `Ads_CTR_Optimisation.csv` is not present in the `data/` folder).*
