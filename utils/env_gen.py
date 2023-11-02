import gym
from gym import spaces
import jax.numpy as jnp

class Env(gym.Env):

    def __init__(self):
        self.img_dim = 128
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(low=0.,high=1.,shape=(self.img_dim,self.img_dim,1),dtype=jnp.float32),
            'ref': spaces.Box(low=0.,high=1.,shape=(self.img_dim,self.img_dim,1),dtype=jnp.float32)
        })

        self.create_dummies()
    
    def create_dummies(self):
        img = jnp.zeros((self.img_dim,self.img_dim,1))
        ref = jnp.zeros((self.img_dim,self.img_dim,1))
        self.obs = {'obs':img,'ref':ref}
        self.reward = 0.
        self.done = False
        self.info = {}

    def seed(self,seed):
        pass
    
    def step(self,action):
        return self.obs, self.reward, self.done, self.info
    
    def reset(self):
        return self.obs