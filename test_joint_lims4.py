import mujoco

# from mujoco import viewer
import numpy as np
from matplotlib import pyplot as plt
from mujoco_rendering import WindowViewer


def get_xml(margin=0.0, d0=0.95, width=0.001):
    # Read from tmp3.xml and put contents into a string.
    with open("tmp3.xml", "r") as f:
        xml_str = f.read()
    xml_str = xml_str.format(margin, d0, width)
    return xml_str


def plot_forces(xml, ax, label=None, x_offset=0.0, x_min=0, x_max=0.0002):
    model = mujoco.MjModel.from_xml_string(xml)
    # jnt_dofs = [60, 61, 62]
    jnt_qposadr = 61
    # jnt_velid = 59
    jnt_id = 55
    jnt_lims = model.joint(jnt_id).range
    data = mujoco.MjData(model)
    breakpoint()
    # renderer = mujoco.Renderer(model)
    viewer = WindowViewer(model, data)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    # mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.qacc = 0  # Assert that there is no the acceleration.
    # data.qpos[jnt_dof] += jnt_lims[1] - 0.1
    # viewer.launch(model, data)
    # data.qpos[jnt_dof] += jnt_lims[1] - 0.1
    plt.close("all")
    while True:
        # data.qpos[jnt_dof] += 1e-2
        data.qpos[jnt_qposadr] += 1e-2
        data.qacc = 0  # Assert that there is no the acceleration.
        mujoco.mj_forward(model, data)
        viewer.render()
        plt.pause(0.01)
    # vopt = mujoco.MjvOption()
    # renderer.update_scene(data)
    # renderer.render()
    # viewer.launch(model, data)
    breakpoint()

    mujoco.mj_inverse(model, data)
    tmp = data.qfrc_inverse
    print(tmp)
    viewer.launch(model, data)
    breakpoint()

    # angle_offsets = np.linspace(x_min, x_max, 2001)
    angle_offsets = np.linspace(jnt_lims[0] - 0.1, jnt_lims[1] + 0.1, 20001)
    print(jnt_lims)
    # angle_offsets = np.linspace(3, jnt_lims[1] + 0.05, 20001)
    vertical_forces = []
    for offset in angle_offsets:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        data.qpos[jnt_dof] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[jnt_velid])

    # Find the height-offset at which the vertical force is smallest.
    # idx = np.argmin(np.abs(vertical_forces))
    # best_offset = angle_offsets[idx]

    # Plot the relationship.
    ax.plot(angle_offsets - x_offset, vertical_forces, linewidth=3, label=label)
    # Red vertical line at offset corresponding to smallest vertical force.
    # ax.axvline(x=best_offset, color="red", linestyle="--")
    # ax.axvline(x=jnt_lims[0], color="blue", linestyle="--")
    # ax.axvline(x=jnt_lims[1], color="red", linestyle="--")
    # Green horizontal line at the humanoid's weight.
    weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
    # ax.axhline(y=weight, color="green", linestyle="--")
    ax.set_xlabel("angle offset (radians)")
    ax.set_ylabel("torque (N)")
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax.minorticks_on()
    # ax.set_title(
    #     f"Smallest vertical force " f"found at offset {best_offset*1000:.4f}mm."
    # )


# solimp=".9 .99 .003" solref=".015 1"
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
# for k0, width in enumerate([0.001, 0.1]):
x_min = 0.01
x_max = 0.025
# x_min = 0
# for k0, width in enumerate([0.005, 20]):
for k0, width in enumerate([0.5]):
    # for k, d0 in enumerate(np.linspace(0, 0.1, 4)):
    for k, d0 in zip([0], [0]):
        ax = axs[k0, k]
        xml = get_xml(0, d0, width)
        plot_forces(xml, ax, label="no margin", x_min=x_min, x_max=x_max)
        # xml = get_xml(0.001, d0, width)
        # plot_forces(xml, ax, label="margin", x_min=x_min, x_max=x_max)
        # xml = get_xml(0, d0, width)
        # plot_forces(xml, ax, label="offset", x_offset=0.00025, x_max=x_max)
        # ax.legend()
        ax.set_title(f"d0 = {d0:.1f}, width = {width:.3f}")
        # ax.set_ylim([-0.05, 0.8])
fig.tight_layout()
plt.show()
