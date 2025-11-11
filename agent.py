"""
Agent Training Module with PPO and Environment Wrappers
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn
import supersuit as ss
from gymnasium import spaces


class VecEnvWrapper(VecEnv):
    """
    Wrapper to convert gymnasium API (multi-value returns) to gym API (single-value returns)
    for compatibility with Stable-Baselines3
    """
    
    def __init__(self, venv):
        """
        Initialize wrapper
        
        Args:
            venv: The vectorized environment to wrap
        """
        self.venv = venv
        
        # Get observation and action spaces from wrapped env
        observation_space = venv.observation_space
        action_space = venv.action_space
        
        # Initialize parent VecEnv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space,
            action_space=action_space
        )
        
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        
    def reset(self):
        """Reset the environment - convert gymnasium to gym format"""
        obs, info = self.venv.reset()
        return obs
    
    def step_async(self, actions):
        """Store actions to be executed"""
        self.actions = actions
    
    def step_wait(self):
        """Execute stored actions and return results in gym format"""
        # Call the wrapped environment's step
        obs, rews, dones, truncated, infos = self.venv.step(self.actions)
        
        # Combine terminated and truncated into single done flag (gym format)
        dones = np.logical_or(dones, truncated)
        
        return obs, rews, dones, infos
    
    def close(self):
        """Close the environment"""
        return self.venv.close()
    
    def get_attr(self, attr_name, indices=None):
        """Get attribute from wrapped environment"""
        return self.venv.get_attr(attr_name, indices)
    
    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in wrapped environment"""
        return self.venv.set_attr(attr_name, value, indices)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on wrapped environment"""
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped with specific wrapper"""
        # Handle different calling patterns from VecMonitor
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        
        # Check if this wrapper is the one we're looking for
        if isinstance(self, wrapper_class):
            return [True] * len(indices)
        
        # For nested wrappers, always return False since we're the outermost wrapper
        # This prevents issues with incompatible env_is_wrapped signatures
        return [False] * len(indices)
    
    def seed(self, seed=None):
        """Set random seed"""
        if hasattr(self.venv, 'seed'):
            return self.venv.seed(seed)
        return [None] * self.num_envs


class TrainingCallback(BaseCallback):
    """Callback to monitor training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        return True
    
    def _on_training_end(self):
        """Called at the end of training"""
        if self.verbose > 0:
            if len(self.episode_rewards) > 0:
                print(f"\nTraining completed!")
                print(f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}")
                print(f"Average Episode Length: {np.mean(self.episode_lengths):.2f}")


class TradingAgent:
    """Manages the PPO agent training"""
    
    def __init__(self, env, learning_rate=0.0001, gamma=0.995, n_steps=2048, ent_coef=0.01):
        """
        Initialize trading agent
        
        Args:
            env: The trading environment
            learning_rate (float): Learning rate for PPO
            gamma (float): Discount factor
            n_steps (int): Number of steps per update
            ent_coef (float): Entropy coefficient for exploration
        """
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.model = None
        
    def create_model(self):
        """Create PPO model with custom architecture"""
        # Custom policy network architecture: 2 hidden layers of 256 neurons
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],  # Policy network
                vf=[256, 256]   # Value network
            ),
            activation_fn=nn.ReLU
        )
        
        print("Creating PPO model with custom architecture...")
        print(f"Policy network: [256, 256] neurons")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Gamma: {self.gamma}")
        print(f"N steps: {self.n_steps}")
        print(f"Entropy coefficient: {self.ent_coef}")
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            ent_coef=self.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=None
        )
        
        return self.model
    
    def train(self, total_timesteps, callback=None):
        """
        Train the agent
        
        Args:
            total_timesteps (int): Total number of timesteps to train
            callback: Optional callback for monitoring
        """
        if self.model is None:
            self.create_model()
        
        print(f"\nStarting training for {total_timesteps} timesteps...")
        
        if callback is None:
            callback = TrainingCallback(verbose=1)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("Training completed!")
        
        return self.model
    
    def save(self, path):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
        return self.model


def create_training_environment(trading_env):
    """
    Create a training-ready environment with all necessary wrappers
    
    Args:
        trading_env: PettingZoo parallel environment
        
    Returns:
        Wrapped environment compatible with SB3
    """
    print("Wrapping environment for training...")
    
    # Step 1: Convert PettingZoo to vectorized environment
    vec_env = ss.pettingzoo_env_to_vec_env_v1(trading_env)
    
    # Step 2: Apply custom wrapper for gymnasium -> gym API compatibility
    wrapped_env = VecEnvWrapper(vec_env)
    
    # Step 3: Add VecMonitor for episode statistics
    wrapped_env = VecMonitor(wrapped_env)
    
    print(f"✓ Environment wrapped successfully")
    print(f"  Number of environments: {wrapped_env.num_envs}")
    print(f"  Observation space: {wrapped_env.observation_space}")
    print(f"  Action space: {wrapped_env.action_space}")
    
    return wrapped_env