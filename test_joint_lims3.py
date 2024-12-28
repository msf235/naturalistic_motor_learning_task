import mujoco
import numpy as np
from matplotlib import pyplot as plt


def get_xml(margin=0.0, d0=0.95, width=0.001):
    return r"""
    <mujoco model="test">
        <visual>
        <map force="0.1" zfar="30"/>
        <rgba haze="0.15 0.25 0.35 1"/>
            <!--offwidth="2560" offheight="1440"-->
        <global
            elevation="20"
            azimuth="10"
            />
        <headlight ambient=".8 .8 .8" specular=".8 .8 .8"/>
        </visual>
        <option gravity="0 0 0"/>
        <default class="main">
            <joint type="hinge" limited="true" range="-50 1"/>
        <default class="arm_upper">
            <geom type="capsule" size=".04"/>
        </default>
        </default>
        <worldbody>
        <body>
            <geom name="upper_arm_right" fromto="0 0 0 0 -.16 -.16" class="arm_upper"/>
            <joint type="hinge" margin="{0}" solimplimit="{1:.3f} 0.95 {2:.8f} 0.5 1.5"/>
        </body>
        </worldbody>
    </mujoco>
    """.format(margin, d0, width)


def plot_forces(xml, ax, label=None, x_offset=0.0, x_min=0, x_max=0.0002):
    model = mujoco.MjModel.from_xml_string(xml)
    jnt_lim = model.joint(0).range[1].item()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0  # Assert that there is no the acceleration.
    mujoco.mj_inverse(model, data)

    angle_offsets = np.linspace(x_min, x_max, 2001)
    vertical_forces = []
    for offset in angle_offsets:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        # Offset the height by `offset`.
        data.qpos[0] += offset
        print(data.qpos[0])
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[0])

    # Find the height-offset at which the vertical force is smallest.
    # idx = np.argmin(np.abs(vertical_forces))
    # best_offset = angle_offsets[idx]

    # Plot the relationship.
    ax.plot(angle_offsets - x_offset, vertical_forces, linewidth=3, label=label)
    # Red vertical line at offset corresponding to smallest vertical force.
    # ax.axvline(x=best_offset, color="red", linestyle="--")
    ax.axvline(x=jnt_lim, color="red", linestyle="--")
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
for k0, width in enumerate([0.005, 20]):
    for k, d0 in enumerate(np.linspace(0, 0.1, 4)):
        ax = axs[k0, k]
        xml = get_xml(0, d0, width)
        plot_forces(xml, ax, label="no margin", x_min=x_min, x_max=x_max)
        xml = get_xml(0.001, d0, width)
        plot_forces(xml, ax, label="margin", x_min=x_min, x_max=x_max)
        # xml = get_xml(0, d0, width)
        # plot_forces(xml, ax, label="offset", x_offset=0.00025, x_max=x_max)
        ax.legend()
        ax.set_title(f"d0 = {d0:.1f}, width = {width:.3f}")
        # ax.set_ylim([-0.05, 0.8])
fig.tight_layout()
plt.show()
