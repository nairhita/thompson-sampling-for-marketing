Python
import numpy as np

class ThompsonSamplingAgent:
    """
    A Multi-Armed Bandit agent using Thompson Sampling to maximize cumulative reward.
    """
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Alpha: Count of successes (clicks) + prior
        self.successes = np.ones(n_arms) 
        # Beta: Count of failures (no clicks) + prior
        self.failures = np.ones(n_arms)  

    def select_arm(self) -> int:
        """
        Samples from the Beta distribution of each arm and selects the one with the highest expected value.
        """
        sampled_theta = [
            np.random.beta(self.successes[i], self.failures[i]) 
            for i in range(self.n_arms)
        ]
        return int(np.argmax(sampled_theta))

    def update(self, chosen_arm: int, reward: int):
        """
        Updates the success or failure count for the chosen arm based on the observed reward.
        """
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
