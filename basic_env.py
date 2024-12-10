from typing import Dict, Tuple, Union, Optional

import numpy as np
from numpy._typing import NDArray

import sim_util as util
import opt_utils

from gymnasium import utils

# import utils
# from gymnasium.envs.mujoco import MujocoEnv
from mujoco_env import MujocoEnv
import mujoco
from gymnasium.spaces import Box

# from gym.utils import seeding # Todo: need to address this
# from gymasium.utils import seeding
from gymnasium.utils import seeding


# viewer.cam.distance = 10
# viewer.cam.elevation = -10
# viewer.cam.azimuth = 180

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    # "distance": 4.0,
    "distance": 10,
    "lookat": np.array((0.0, 0.0, 1.15)),
    # "elevation": -20.0,
    "elevation": -10.0,
    "azimuth": 180,
}


class BasicEnv(MujocoEnv, utils.EzPickle):
    r"""
    | Parameter                                    | Type      | Default           | Description                                                                                                                                                                                         |
    | -------------------------------------------- | --------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   |`"walker2d_v5.xml"`| Path to a MuJoCo model                                                                                                                                                                              |
    | `frame_skip`                                 | **int**   |`"walker2d_v5.xml"`| Used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.                                                                             |
    | `keyframe_name`                              | **str**   | ``                | Name of mujoco model keyframe to use.                                                                                                                                                               |
    | `reset_noise_scale`                          | **float** | `5e-3`            | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple", "None"],
    }

    def __init__(
        self,
        # xml_file: str = "walker2d_v5.xml",
        xml_file: str = "./humanoid_and_baseball.xml",
        frame_skip: int = 4,
        keyframe_name: Optional[str] = None,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 5e-3,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reset_noise_scale,
            **kwargs,
        )

        self._reset_noise_scale = reset_noise_scale
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        if keyframe_name is not None:
            keyframe_id = self.model.keyframe(keyframe_name).id
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

        self.keyframe_name = keyframe_name

    def _get_obs(self) -> NDArray[np.float64]:
        position = self.data.qpos.flatten()
        # velocity = np.clip(self.data.qvel.flatten(), -10, 10)
        velocity = self.data.qvel.flatten()

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action, render=True):
        self.do_simulation(action, self.frame_skip)
        # contacts = util.get_contact_pairs(self.model, self.data)
        # for cp in contacts:
        #     if "ball" in cp and "target" in cp:
        #         self.terminated = True

        observation = self._get_obs()
        reward = 0.0
        info = {}

        if self.render_mode == "human" and render:
            self.render()
        return observation, reward, self.terminated, False, info

    def reward_fn(self, x1, x2, sig=1):
        diff = x1 - x2
        dist = np.linalg.norm(diff)
        # Gaussian function of dist
        fact = sig * np.sqrt(2 * np.pi)
        return_val = np.exp(-(dist**2) / (2 * sig**2)) / fact
        return return_val

    def _reset_simulation(self) -> None:
        if self.keyframe_name is not None:
            key_id = self.model.keyframe(self.keyframe_name).id
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

    def reset_sim_time_counter(self):
        self.mujoco_renderer.data.time = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if options is not None:
            n_steps = options["n_steps"] if "n_steps" in options else 0
            # render = options["render"] if "render" in options else True
        else:
            n_steps = 0
            # render = True
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        # ob = self.reset_model(n_steps=n_steps, render=render)
        ob = self.reset_model(n_steps=n_steps)

        info = self._get_reset_info()

        if self.render_mode == "human":
            self.reset_sim_time_counter()
            self.render()
        return ob, info

    def reset_model(self, n_steps: int = 0):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        self.terminated = False

        zctrl = np.zeros(self.action_space.shape)
        for _ in range(n_steps):
            self.do_simulation(zctrl, self.frame_skip)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
