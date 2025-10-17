import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from local_training.rl_models.environment import TradingEnv

class PPOTrainer:
    """
    A trainer for the PPO trading agent.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Wrap the environment in a vectorized environment
        self.env = DummyVecEnv([lambda: TradingEnv(self.df)])
        
        # Initialize the PPO model
        self.model = PPO(
            "MlpPolicy", 
            self.env, 
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/"
        )
        
    def train(self, total_timesteps=1_000_000):
        """
        Train the PPO model.
        """
        self.model.learn(total_timesteps=total_timesteps)
        
    def save_model(self, path="ppo_trader_model"):
        """
        Save the trained model.
        """
        self.model.save(path)

if __name__ == '__main__':
    # Create a dummy dataframe for testing
    data = {'feature_{}'.format(i): range(100) for i in range(50)}
    df = pd.DataFrame(data)

    # Initialize and train the PPO trader
    trainer = PPOTrainer(df=df)
    trainer.train(total_timesteps=1000) # Using a small number for testing
    trainer.save_model()
