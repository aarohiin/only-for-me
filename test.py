import time
import numpy as np
from car_env import CarEnv

def test_environment():
    # Initialize the environment
    env = CarEnv(render=True)

    # Reset the environment to start a new episode
    obs = env.reset()
    print("Initial Observation:", obs)

    # Run the simulation for a fixed number of steps
    num_steps = 100
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        # Take a random action
        action = env.action_space.sample()
        print(f"Action Taken: {action}")

        # Step the environment
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        # Wait for a short time to visualize the simulation
        time.sleep(0.1)

        # If the episode is done, reset the environment
        if done:
            print("Episode finished. Resetting environment...")
            obs = env.reset()

    # Close the environment
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    test_environment()