import numpy as np
import mujoco as mj
import humanoid2d

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

def get_contact_pairs(model, data):
    contact_pairs = [[model.geom(c.geom[0]).name, model.geom(c.geom[1]).name]
                     for c in data.contact]
    # contact_pairs = np.stack(contact_pairs)
    return contact_pairs

def test_target_hit(ctrls):
    env = humanoid2d.Humanoid2dEnv(
        render_mode='human',
        frame_skip=1,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        reset_noise_scale=0,
        xml_file='./humanoid_and_baseball.xml',
        keyframe_name='wide',)
    model = env.model
    data = env.data
    # model_file="./humanoid_and_baseball.xml"
    # model = mj.MjModel.from_xml_path(model_file)
    # data = mj.MjData(model)

    for k, ctrl in enumerate(ctrls):
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrl
        mj.mj_step2(model, data)
        cps = get_contact_pairs(model, data)
        for cp in cps:
            if 'ball' in cp and 'floor' in cp:
                return 0
            if 'ball' in cp and 'target' in cp:
                return 1
    return 0

if __name__ == '__main__':
    ctrls = np.load('ball_throw_1_ctrls.npy')
    ctrls_end = np.zeros((10, ctrls.shape[1]))
    ctrls_full = np.vstack((ctrls, ctrls_end))
    hit = test_target_hit(ctrls_full)
    print(hit)
