from collections import abc
import copy
from typing import Any
import numpy as np


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
