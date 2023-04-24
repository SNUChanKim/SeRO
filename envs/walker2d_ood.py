import numpy as np
from envs import mujoco_env
from gym import utils
import math

class Walker2dOODEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.return_to_in_distribution = False
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        self.init_qpos = np.array([-1.27287638, 0.10, 1.5740085, -0.122172944, 0, -0.7853975, -0.122172944, 0, -0.7853975])
        
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
            
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False
        ob = self._get_obs()
        original_reward = reward
        if not (height > 0.9 and height < 2.0 and
                    ang > -0.3 and ang < 0.3):
            reward = 0.0
            if self.return_to_in_distribution:
                done = True
        else:
            self.return_to_in_distribution = True
        return ob, reward, done, dict(original_reward=original_reward)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.return_to_in_distribution = False
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
