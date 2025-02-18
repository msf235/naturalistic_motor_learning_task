from collections import abc
import copy
from typing import Any
import numpy as np
import basic_env
import imageio
import mujoco as mj


class targetRender:
    def __init__(self, env, target_data_list, sites) -> None:
        self.counter = 0
        self.target_data_list = target_data_list
        self.env = env
        self.sites = sites

    def render(self):
        self.env.mujoco_renderer.viewer._markers = []
        for k, target_data in enumerate(self.target_data_list):
            marker_pos = target_data[self.counter]
            self.env.mujoco_renderer.viewer.add_marker(
                size=np.array([0.05, 0.05, 0.05]),
                pos=marker_pos,
                matid=0,
                rgba=(1, 1, 0, 0.5),
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label="targ",
                emission=0,
                specular=0.1,
                shininess=0.1,
                reflectance=0,
            )
            marker_pos = self.env.data.site(self.sites[k]).xpos
            self.env.mujoco_renderer.viewer.add_marker(
                size=np.array([0.05, 0.05, 0.05]),
                pos=marker_pos,
                matid=0,
                rgba=(1, 1, 0, 0.5),
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label="hand",
                emission=0,
                specular=0.1,
                shininess=0.1,
                reflectance=0,
            )
        self.counter += 1
        return self.env.render()

    def reset_counter(self):
        self.counter = 0


def make_video_of_motion(
    model_file, qposs, output_file, traj_targs=None, site_names=None, speed_factor=1
):
    """Make video of mujoco positions as stored in qpos."""
    DEFAULT_CAMERA_CONFIG = {
        "trackbodyid": 2,
        "distance": 5,
        "lookat": np.array((0.0, 0.0, 1.15)),
        "elevation": -10.0,
        "azimuth": 180,
    }

    # render_mode = "rgb_array"
    render_mode = "human"

    env = basic_env.BasicEnv(
        render_mode=render_mode,
        frame_skip=1,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        xml_file=model_file,
    )
    env.reset()
    env.render()
    if traj_targs is not None:
        render_class = targetRender(env, traj_targs, site_names)
        render_fn = render_class.render
    else:
        render_fn = env.render

    # Example: Create a sequence of 100 frames (random colors)
    # frames = [
    #     np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)
    # ]

    # Save as video
    fps = int(speed_factor / env.model.opt.timestep)

    frames = []
    for qpos in qposs:
        env.set_state(qpos, np.zeros(env.model.nv))
        rgb_mat = render_fn()
        frames.append(rgb_mat)

    with imageio.get_writer(output_file, fps=fps, format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Video saved as {output_file}")


def propagate_singleton_points(data):
    """
    Modifies a (T, d) numpy array such that singleton points (values surrounded by np.nan)
    are copied into their adjacent time points.

    Parameters:
    data (np.ndarray): A (T, d) numpy array containing time series data.

    Returns:
    np.ndarray: Modified data with singleton values propagated to adjacent NaN values.
    """
    T, d = data.shape
    new_data = data.copy()

    for t in range(T):
        for dim in range(d):
            if np.isnan(data[t, dim]):
                continue  # Skip already NaN values

            # Check if it's a singleton point
            left_nan = t == 0 or np.isnan(data[t - 1, dim])
            right_nan = t == T - 1 or np.isnan(data[t + 1, dim])

            if left_nan and right_nan:
                # If singleton, propagate value to neighbors
                if t > 0 and np.isnan(new_data[t - 1, dim]):
                    new_data[t - 1, dim] = data[t, dim]
                if t < T - 1 and np.isnan(new_data[t + 1, dim]):
                    new_data[t + 1, dim] = data[t, dim]

    return new_data


class RightEndpointDict(abc.MutableMapping):
    def __init__(self, inp_dict: dict[float, Any]) -> None:
        """
        If inp_dict = {30: 'A', 50: 'B'} then:
            key = -1 -> return 'A'
            key = 30 -> return 'A'
            key = 35 -> return 'B'
            key = 50 -> return 'B'
            key = 51 -> raise KeyError
        """
        self.dict = copy.deepcopy(inp_dict)
        self.right_endpoints = sorted(list(inp_dict.keys()))
        self.intervals = []
        self.intervals.append([-np.inf, self.right_endpoints[0]])
        for k in range(len(self.right_endpoints) - 1):
            self.intervals.append(
                [self.right_endpoints[k], self.right_endpoints[k + 1]]
            )
        # self.intervals.append([self.right_endpoints[-1], np.inf])
        #

    def keys(self):
        return self.right_endpoints

    def values(self):
        return [
            self.dict[key] for key in self.right_endpoints
        ]  # List is in correct order

    def __getitem__(self, key):
        if key > self.right_endpoints[-1]:
            raise KeyError(
                f"key must be less than rightmost endpoint {self.right_endpoints[-1]}"
            )
        endpoint = -1
        for endpoint in self.right_endpoints:
            if key <= endpoint:
                break
        return self.dict[endpoint]

    def __setitem__(self, key, val):
        if key not in self.right_endpoints:
            raise KeyError(f"key must be in {self.right_endpoints}")
        self.dict[key] = val

    def __delitem__(self, key):
        if key not in self.right_endpoints:
            raise KeyError(f"key must be in {self.right_endpoints}")
        del self.dict[key]

    def get_interval(self, key):
        for interval in self.intervals:
            if key > interval[0] and key <= interval[1]:
                return interval
        return None

    def __repr__(self):
        str1 = (
            f"(-∞, {self.intervals[0][1]}"
            + "]:\n"
            + str(self.dict[self.intervals[0][1]])
            + "\n\n"
        )
        for interval in self.intervals[1:-1]:
            str1 += (
                "("
                + str(interval[0])
                + ","
                + str(interval[1])
                + "]:\n"
                + str(self.dict[interval[1]])
                + "\n\n"
            )
        str1 += (
            "("
            + str(self.intervals[-1][0])
            + ","
            + str(self.intervals[-1][1])
            + "]:\n"
            + str(self.dict[self.intervals[-1][1]])
            + "\n\n"
        )
        str1 += f"RightEndpointDict with keys {self.right_endpoints}"
        return str1

    def __iter__(self):
        return iter(self.right_endpoints)

    def __len__(self):
        return len(self.right_endpoints)


class LeftEndpointDict(abc.MutableMapping):
    def __init__(self, inp_dict: dict[float, Any]) -> None:
        """
        If inp_dict = {0: 'A', 30: 'B'} then:
            key = -1 -> raise KeyError
            key = 0 -> return 'A'
            key = 10 -> return 'A'
            key = 30 -> return 'B'
            key = 35 -> return 'B'
        """
        self.dict = copy.deepcopy(inp_dict)
        self.left_endpoints = sorted(list(inp_dict.keys()))
        self.intervals = []
        for k in range(len(self.left_endpoints) - 1):
            self.intervals.append([self.left_endpoints[k], self.left_endpoints[k + 1]])
        self.intervals.append([self.left_endpoints[-1], np.inf])
        # self.intervals.append([self.left_endpoints[-1], np.inf])
        #

    def keys(self):
        return self.left_endpoints

    def values(self):
        return [
            self.dict[key] for key in self.left_endpoints
        ]  # List is in correct order

    def __getitem__(self, key):
        if key < self.left_endpoints[0]:
            raise KeyError(
                f"key must be greater than or equal to leftmost endpoint {self.left_endpoints[0]}"
            )
        if len(self.left_endpoints) > 0:
            for endpoint in self.left_endpoints[::-1]:
                if key >= endpoint:
                    break
            return self.dict[endpoint]
        else:
            return {}

    def __setitem__(self, key, val):
        if key not in self.left_endpoints:
            raise KeyError(f"key must be in {self.left_endpoints}")
        self.dict[key] = val

    def __delitem__(self, key):
        if key not in self.left_endpoints:
            raise KeyError(f"key must be in {self.left_endpoints}")
        del self.dict[key]

    def get_interval(self, key):
        for interval in self.intervals:
            if key >= interval[0] and key < interval[1]:
                return interval
        return None

    def __repr__(self):
        str1 = ""
        for interval in self.intervals[:-1]:
            str1 += (
                "["
                + str(interval[0])
                + ","
                + str(interval[1])
                + "):\n"
                + str(self.dict[interval[0]])
                + "\n\n"
            )
        str1 += (
            f"[{self.intervals[-1][1]}, ∞)"
            + ":\n"
            + str(self.dict[self.intervals[-1][0]])
            + "\n\n"
        )
        str1 += f"RightEndpointDict with keys {self.left_endpoints}"
        return str1

    def __iter__(self):
        return iter(self.left_endpoints)

    def __len__(self):
        return len(self.left_endpoints)
