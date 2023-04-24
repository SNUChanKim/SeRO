import numpy as np
from gym import utils
from envs import mujoco_env
import math
from utils.utils import euler_from_quaternion, euler_to_quaternion
class AntOODEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.return_to_in_distribution = False
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        
        ob = self._get_obs()
        yaw, pitch, roll = euler_from_quaternion(ob[1], ob[2], ob[3], ob[4])
        original_reward = reward
        if not (np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.5) or self._check_flip(roll, pitch):
            reward = 0.0
            if self.return_to_in_distribution:
                done = True
        else:
            self.return_to_in_distribution = True

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            original_reward=original_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def _check_flip(self, roll, pitch):
        flip = False
        if roll > math.pi/2.0 or roll < -math.pi/2.0:
            flip = True
        if pitch > math.pi/2.0 or pitch < -math.pi/2.0:
            flip = True
        return flip

    def reset_model(self):
        self.return_to_in_distribution = False
        qx, qy, qz, qw = euler_to_quaternion(0, math.pi, 0)
        qpos = self.init_qpos
        qpos[3] = qx
        qpos[4] = qy
        qpos[5] = qz
        qpos[6] = qw
        
        qpos[2] = 0.25
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
