import numpy as np
from gym import utils
from envs import mujoco_env
import math
from copy import deepcopy

class HalfCheetahOODEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        self.init_qpos = np.array([0., -0.5, -2.879790833, 1.047196667, 0., 0., 0., 0., 0.])
        self.qpos = deepcopy(self.init_qpos)
        self.return_to_in_distribution = False
        
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
            
        xposafter = self.sim.data.qpos[0]
        self.qpos = deepcopy(self.sim.data.qpos)
        if self.qpos[2] < -math.pi:
            self.qpos[2] += 2*math.pi
        elif self.qpos[2] > math.pi:
            self.qpos[2] -= 2*math.pi

        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        original_reward = reward
        if self.qpos[2]>math.pi/2.0 or self.qpos[2]<-math.pi/2.0:
            reward = 0
            if self.return_to_in_distribution:
                done = True
        else:
            self.return_to_in_distribution = True
            
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, original_reward=original_reward)

    def _get_obs(self):
        return np.concatenate([
            self.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.return_to_in_distribution = False
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.qpos = deepcopy(self.sim.data.qpos)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
