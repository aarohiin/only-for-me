import pybullet as p
import pybullet_data
import gym
import numpy as np
from gym import spaces

class CarEnv(gym.Env):
    def __init__(self, render=True):
        super(CarEnv, self).__init__()
        self.render_sim = render
        if self.render_sim:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.time_step = 1.0 / 60.0
        p.setTimeStep(self.time_step)
        
        self.boundary = 10
        self.goal_pos = np.array([4, 4])
        self.home_pos = np.array([-4, -4])
        self.car = None
        self.wheels = [2, 3]            # Wheel joint indices
        self.steering_links = [4, 6]      # Steering joint indices
        
        self.num_lidar_rays = 12
        self.lidar_range = 6.0
        
        self.action_space = spaces.Discrete(5)  # forward, left, right, backward, stop
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 + self.num_lidar_rays,), dtype=np.float32
        )
        self.reset()
    
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        
        p.loadURDF("plane.urdf")
        self._draw_boundary()
        self._draw_home_and_goal()
        self._generate_obstacles()
        
        self.car = p.loadURDF(
            "racecar/racecar.urdf",
            basePosition=self.home_pos.tolist() + [0.2],
            useFixedBase=False
        )
        # Reset all joint motors
        for joint in range(p.getNumJoints(self.car)):
            p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, force=0)
        for wheel in self.wheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=300)
        
        self.prev_dist_to_goal = np.linalg.norm(self.home_pos - self.goal_pos)
        return self._get_obs()
    
    def _draw_boundary(self):
        for i in range(-self.boundary, self.boundary + 1):
            for j in [-self.boundary, self.boundary]:
                p.loadURDF("cube_small.urdf", [i, j, 0])
                p.loadURDF("cube_small.urdf", [j, i, 0])
    
    def _draw_home_and_goal(self):
        green = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=green, basePosition=self.home_pos.tolist() + [0.05])
        red = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=red, basePosition=self.goal_pos.tolist() + [0.05])
    
    def _generate_obstacles(self, num_obstacles=15):
        for _ in range(num_obstacles):
            while True:
                pos = np.random.uniform(-self.boundary + 1, self.boundary - 1, size=2)
                if (np.linalg.norm(pos - self.goal_pos) > 2.0 and 
                    np.linalg.norm(pos - self.home_pos) > 2.0):
                    break
            size = np.random.uniform(0.3, 0.6)
            z_height = 0.2  # Lower obstacles to avoid being climbed
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, z_height])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, z_height], rgbaColor=[0.2, 0.2, 0.2, 1])
            p.createMultiBody(
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos.tolist() + [z_height],
            )
    
    def _get_lidar_readings(self):
        pos, _ = p.getBasePositionAndOrientation(self.car)
        pos = np.array(pos)
        rays_from, rays_to = [], []
        for i in range(self.num_lidar_rays):
            angle = i * 2 * np.pi / self.num_lidar_rays
            dx, dy = self.lidar_range * np.cos(angle), self.lidar_range * np.sin(angle)
            rays_from.append(pos)
            rays_to.append(pos + np.array([dx, dy, 0]))
        results = p.rayTestBatch(rays_from, rays_to)
        distances = [self.lidar_range if r[0] == -1 else r[2] * self.lidar_range for r in results]
        
        # Draw LiDAR rays if rendering is enabled
        if self.render_sim:
            for i in range(self.num_lidar_rays):
                p.addUserDebugLine(
                    lineFromXYZ=rays_from[i],
                    lineToXYZ=rays_to[i],
                    lineColorRGB=[1, 0, 0],
                    lifeTime=self.time_step
                )
        return np.array(distances)
    
    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.car)
        lin_vel, _ = p.getBaseVelocity(self.car)
        lidar = self._get_lidar_readings()
        return np.array([pos[0], pos[1], lin_vel[0], lin_vel[1]] + list(lidar), dtype=np.float32)
    
    def step(self, action):
        throttle = 50
        steering = 0
        pos = np.array(self._get_obs()[:2])
        goal_vector = self.goal_pos - pos
        goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
        current_orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.car)[1])[2]
        angle_diff = goal_angle - current_orientation
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        steering = np.clip(angle_diff, -0.5, 0.5)
        
        # Use LiDAR to avoid obstacles
        lidar = self._get_lidar_readings()
        min_distance = np.min(lidar)
        if min_distance < 1.0:
            obstacle_angle = np.argmin(lidar) * (2 * np.pi / self.num_lidar_rays)
            if obstacle_angle > np.pi:
                obstacle_angle -= 2 * np.pi
            steering -= np.clip(obstacle_angle, -0.5, 0.5)
        
        vel = throttle if action in [0, 1, 2] else -throttle if action == 3 else 0
        for wheel in self.wheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=vel, force=300)
        for steer in self.steering_links:
            p.setJointMotorControl2(self.car, steer, p.POSITION_CONTROL, targetPosition=steering)
        
        p.stepSimulation()
        obs = self._get_obs()
        reward = -0.1
        done = False
        
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)
        progress_reward = (self.prev_dist_to_goal - dist_to_goal) * 10
        reward += progress_reward
        self.prev_dist_to_goal = dist_to_goal
        
        if dist_to_goal < 1.0:
            reward += 1000
            done = True
        if np.abs(pos[0]) > self.boundary or np.abs(pos[1]) > self.boundary:
            reward -= 100
            done = True
        if min_distance < 0.5:
            reward -= 50
        
        return obs, reward, done, {}
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        p.disconnect()
