# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from car_env import CarEnv
import os
import sys

# Create environment
env = DummyVecEnv([lambda: CarEnv(render=False)])
check_env(env.envs[0])

# Logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
sys.stdout = open(os.path.join(log_dir, "log.txt"), "w")

print("Starting training...")

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train model
model.learn(total_timesteps=20000)

# Save model
model.save("ppo_car_model")

print("Training completed and model saved.")
sys.stdout.close()
