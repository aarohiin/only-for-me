# watch_agent.py
from stable_baselines3 import PPO
from car_env import CarEnv
import time

# Load the environment with GUI enabled
env = CarEnv(render=True)

# Load the trained model
model = PPO.load("ppo_car_obstacle")

# Run one episode
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    time.sleep(1.0 / 30.0)  # 30 FPS

print(f"🎯 Episode finished. Total reward: {total_reward}")
env.close()
