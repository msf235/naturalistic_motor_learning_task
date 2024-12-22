from typing import Any
import arm_targ_traj
import masks


def test_arm_targ_traj__get_from_interv_dict():
    print("Testing test_arm_targ_traj.get_from_interv_dict")
    test_interv_dict: dict[int | float, Any] = {0: "A", 30: "B", 50: "C"}
    test_keys = [0, 10, 30, 60]
    results = [
        arm_targ_traj.get_from_interv_dict(test_interv_dict, key) for key in test_keys
    ]
    print(
        f'Using test dictionary "{test_interv_dict}" with keys "{test_keys}",',
        f'get the values "{results}".',
    )
    assert results == ["A", "A", "B", "C"]
    print("Test passed.")


def test_arm_targ_traj__get_from_right_endpoint_interv_dict():
    print("Testing test_arm_targ_traj.get_from_right_endpoint_interv_dict")
    test_interv_dict: dict[int | float, Any] = {30: "A", 50: "B"}
    test_keys = [0, 30, 40, 60]
    results = [
        arm_targ_traj.get_from_right_endpoint_interv_dict(test_interv_dict, key)
        for key in test_keys
    ]
    print(
        f'Using test dictionary "{test_interv_dict}" with keys "{test_keys}",',
        f'get the values "{results}".',
    )
    assert results == ["A", "B", "B", "B"]
    print("Test passed.")


def test_arm_targ_traj__RightEndpointDict():
    print("Testing test_arm_targ_traj.RightEndpointDict")
    test_interv_dict: dict[int | float, Any] = {30: "A", 50: "B"}
    red = arm_targ_traj.RightEndpointDict(test_interv_dict)
    test_keys = [-1, 30, 35, 50]
    results = [red[key] for key in test_keys]
    print(
        f'Using test dictionary "{test_interv_dict}" with keys "{test_keys}",',
        f'get the values "{results}".',
    )
    assert results == ["A", "A", "B", "B"]
    try:  # Test that the RightEndpointDict object behaves as expected
        red[51]
    except KeyError:
        print("Using key=51 throws a KeyError as it should. Test passed.")
        return
    print("Test failed -- error not thrown for key 51.")


def masks_generate_decaying_intervals():
    interval_end_times = [2, 5, 7, 10]
    decay_factor = 0.5
    # matrix = masks.generate_decaying_intervals(interval_end_times, Tk, decay_factor)
    matrix = masks.generate_decaying_intervals(interval_end_times, decay_factor)
    for row in matrix:
        print(row)
    assert matrix == [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0],
    ]
    print("Test passed.")


if __name__ == "__main__":
    # test_arm_targ_traj__get_from_interv_dict()
    test_arm_targ_traj__get_from_right_endpoint_interv_dict()
    # masks_generate_decaying_intervals()
