import mujoco as mj

def get_contact_pairs(model, data):
    contact_pairs = [[model.geom(c.geom[0]).name, model.geom(c.geom[1]).name]
                     for c in data.contact]
    return contact_pairs

def test_target_hit(ctrls):
    model_file="./humanoid_and_tennis.xml"
    model = mj.MjModel.from_xml_path(model_file)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.keyframe('wide_tennis_pos').id)

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

