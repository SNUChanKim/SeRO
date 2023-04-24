from gym.envs.registration import registry, register, make, spec
try:
    from envs.ant_ood import AntOODEnv
    from envs.ant_normal import AntNormalEnv
    from envs.walker2d_normal import Walker2dNormalEnv
    from envs.walker2d_ood import Walker2dOODEnv
    from envs.half_cheetah_normal import HalfCheetahNormalEnv
    from envs.half_cheetah_ood import HalfCheetahOODEnv
    from envs.hopper_normal import HopperNormalEnv
    from envs.hopper_ood import HopperOODEnv
except ImportError:
    Box2D = None

register(
    id='AntOOD-v2',
    entry_point='envs:AntOODEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='AntNormal-v2',
    entry_point='envs:AntNormalEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Walker2dNormal-v2",
    entry_point="envs:Walker2dNormalEnv",
    max_episode_steps=1000,
)

register(
    id="Walker2dOOD-v2",
    entry_point="envs:Walker2dOODEnv",
    max_episode_steps=1000,
)

register(
    id="HalfCheetahNormal-v2",
    entry_point="envs:HalfCheetahNormalEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetahOOD-v2",
    entry_point="envs:HalfCheetahOODEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperNormal-v2',
    entry_point='envs:HopperNormalEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperOOD-v2',
    entry_point='envs:HopperOODEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)