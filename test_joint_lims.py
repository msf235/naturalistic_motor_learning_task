import mujoco
import numpy as np
from matplotlib import pyplot as plt

# solimp=".9 .99 .003" solref=".015 1"
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()
for k, d0 in enumerate(np.linspace(0, 0.9, 8)):
    xml = """
    <mujoco model="test">
        <default class="main">
            <joint type="hinge" margin="0" solimplimit="{0:.2f} 0.95 0.001 0.5 2" limited="true" range="0 0.1"/>
        <default class="arm_upper">
            <geom size=".04"/>
        </default>
        </default>
        <worldbody>
            <body>
            <geom name="upper_arm_right" fromto="0 0 0 0 -.16 -.16" class="arm_upper"/>
            <joint/>
            </body>
        </worldbody>
    </mujoco>
    """.format(d0)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0  # Assert that there is no the acceleration.
    mujoco.mj_inverse(model, data)
    print(data.qfrc_inverse)

    height_offsets = np.linspace(-0.001, 0.001, 2001)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        # Offset the height by `offset`.
        data.qpos[2] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]

    # Plot the relationship.
    ax = axs[k]
    ax.plot(height_offsets * 1000, vertical_forces, linewidth=3)
    # Red vertical line at offset corresponding to smallest vertical force.
    ax.axvline(x=best_offset * 1000, color="red", linestyle="--")
    # Green horizontal line at the humanoid's weight.
    weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
    ax.axhline(y=weight, color="green", linestyle="--")
    ax.set_xlabel("Height offset (mm)")
    ax.set_ylabel("Vertical force (N)")
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax.minorticks_on()
    # ax.set_title(
    #     f"Smallest vertical force " f"found at offset {best_offset*1000:.4f}mm."
    # )
    ax.set_title(f"d0 = {d0:.2f}")
fig.tight_layout()
plt.show()
