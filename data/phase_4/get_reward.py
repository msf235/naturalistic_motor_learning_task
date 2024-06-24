import mujoco as mj

def get_contact_pairs(model, data):
    contact_pairs = [[model.geom(c.geom[0]).name, model.geom(c.geom[1]).name]
                     for c in data.contact]
    return contact_pairs

def get_reward(ctrls, prev_dist=None):
    model_file="./humanoid_and_baseball.xml"
    model = mj.MjModel.from_xml_path(model_file)
    data = mj.MjData(model)
    keyframe="wide"
    key_id = model.keyframe(keyframe).id
    mj.mj_resetDataKeyframe(model, data, key_id)

    for k, ctrl in enumerate(ctrls):
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrl
        mj.mj_step2(model, data)
        cps = get_contact_pairs(model, data)
        for cp in cps:
            if 'ball' in cp and 'floor' in cp:
                return 0, None
            if 'ball' in cp and 'target' in cp:
                ball_z = data.site('ball').xpos[-1]
                new_dist = abs(data.site('bullseye').xpos[-1] - ball_z)
                if prev_dist is not None and new_dist >= prev_dist:
                    return 0, new_dist
                else:
                    return 1, new_dist
    return 0, None

