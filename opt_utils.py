import mujoco as mj
import time
import sys
import numpy as np
import scipy
# import control_logic as cl
import sim_util as util
import humanoid2d as h2d
import copy
import sim_util
import optimizers as opts

epsilon_grad = 5e-9

body_root_keys = ['Pelvis']
body_without_root_keys = ['Hip', 'Knee', 'Ankle', 'Toe', 'Torso',
             'Spine', 'Chest', 'Neck', 'Head', 'Thorax', 'Shoulder',
             'Elbow', 'Wrist', 'Hand'] 
body_keys = body_root_keys + body_without_root_keys
# body_keys = ['human', 'shoulder', 'hand', 'torso', 'hip', 'knee', 'ankle',
             # 'abdomen', 'elbow', 'arm', 'wrist'] 
abd_keys = ['Torso', 'Spine', 'Chest']
leg_keys = ['Hip', 'Knee', 'Ankle', 'Toe']
arm_keys = ['Thorax', 'Shoulder', 'Elbow', 'Wrist', 'Hand']


### LQR
def get_ctrl0(model, data, qpos0, stable_jnt_ids, ctrl_act_ids):
    # data = copy.deepcopy(data)
    # data.qpos[:] = qpos0.copy()
    mj.mj_forward(model, data)
    data.qacc[:] = 0
    data.qvel[:] = 0
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    # breakpoint()
    qfrc0 = qfrc0[stable_jnt_ids]
    # M = data.actuator_moment[:, stable_jnt_ids]
    M = data.actuator_moment[ctrl_act_ids][:, stable_jnt_ids]
    # Probably much better way to do this
    # ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(M)
    ctrl0 = np.linalg.lstsq(M.T, qfrc0, rcond=None)[0]
    # ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    return ctrl0


def key_match(key, key_list):
    for k in key_list:
        if k in key:
            return True
    return False

def batch_differentiatePos(model, dt, qpos1_list, qpos2_list):
    res = []
    qvel = np.zeros(model.nv)
    assert len(qpos1_list) == len(qpos2_list)
    for qpos1, qpos2 in zip(qpos1_list, qpos2_list):
        mj.mj_differentiatePos(model, qvel, dt, qpos1, qpos2)
        res.append(qvel.copy())
    return np.array(res)


def convert_dofadr(model, data=None, joint_ids=None, concat=False):
    """Convert joint ids to dofadr. A list of dofadr is associated with each
    joint id; hence, this function returns a list of lists of dofard."""
    # WARNING: I'm not totally sure that len(bodyid) is always the number of
    # dofs for a joint, which I assume here.
    if joint_ids is None:
        joint_ids = range(model.njnt)
    if not hasattr(joint_ids, '__len__'):
        joint = model.joint(joint_ids)
        dof_st = joint.dofadr.item()
        n_dofs = len(joint.bodyid)
        dofadr = list(range(dof_st, dof_st+n_dofs))
        return dofadr
    dofadrs = []
    for id in joint_ids:
        joint = model.joint(id)
        dof_st = joint.dofadr.item()
        n_dofs = len(joint.bodyid)
        dofadr = list(range(dof_st, dof_st+n_dofs))
        if concat:
            dofadrs.extend(dofadr)
        else:
            dofadrs.append(dofadr)
    return dofadrs

def get_body_joints(model, data=None):
    """Get joint names for body."""
    jntname = lambda k: model.joint(k).name
    dof_conv = lambda li: convert_dofadr(model, None, li, True)

    bod_jnts = {}
    bod_jnts['ids'] = [k for k in range(model.njnt)
                       if key_match(jntname(k), body_keys)]
    bod_jnts['ids_without_root'] = [k for k in range(model.njnt)
                       if key_match(jntname(k), body_without_root_keys)]
    body_ids = bod_jnts['ids_without_root'] # Exclude root here
    bod_jnts['dofadrs'] = dof_conv(bod_jnts['ids'])
    bod_jnts['dofadrs_without_root'] = dof_conv(bod_jnts['ids_without_root'])
    body_dofs = bod_jnts['dofadrs_without_root'] # Exclude root here
    bod_jnts['root_ids'] = [k for k in bod_jnts['ids']
                             if key_match(jntname(k), body_root_keys)]
    bod_jnts['root_dofadrs'] = dof_conv(bod_jnts['root_ids'])

    bod_jnts['abdomen_ids'] = []
    bod_jnts['abdomen_dofadrs'] = []
    for id, dof in zip(body_ids, body_dofs):
        if key_match(jntname(id), abd_keys):
            bod_jnts['abdomen_ids'].append(id)
            bod_jnts['abdomen_dofadrs'].append(dof)
    bod_jnts['leg_ids'] = []
    bod_jnts['leg_dofadrs'] = []
    for id, dof in zip(body_ids, body_dofs):
        if key_match(jntname(id), leg_keys):
            bod_jnts['leg_ids'].append(id)
            bod_jnts['leg_dofadrs'].append(dof)
    bod_jnts['balance_ids'] = bod_jnts['abdomen_ids'] + bod_jnts['leg_ids']
    bod_jnts['balance_ids'].sort()
    bod_jnts['balance_dofadrs'] = bod_jnts['abdomen_dofadrs'] + bod_jnts['leg_dofadrs']
    bod_jnts['balance_dofadrs'].sort()
    bod_jnts['other_ids'] = [k for k in body_ids if k not in bod_jnts['balance_ids']]
    bod_jnts['other_dofadrs'] = dof_conv(bod_jnts['other_ids'])
    bod_jnts['right_arm_ids'] = []
    bod_jnts['right_arm_dofadrs'] = []
    bod_jnts['left_arm_ids'] = []
    bod_jnts['left_arm_dofadrs'] = []
    for id, dof in zip(body_ids, body_dofs):
        if key_match(jntname(id), arm_keys):
            if 'R_' in jntname(id):
                bod_jnts['right_arm_ids'].append(id)
                bod_jnts['right_arm_dofadrs'].append(dof)
            if 'L_' in jntname(id):
                bod_jnts['left_arm_ids'].append(id)
                bod_jnts['left_arm_dofadrs'].append(dof)
    bod_jnts['not_right_arm_ids'] = [i for i in body_ids if i not in
                                   bod_jnts['right_arm_ids']]
    bod_jnts['not_right_arm_dofadrs'] = dof_conv(bod_jnts['not_right_arm_ids'])
    bod_jnts['not_left_arm_ids'] = [i for i in body_ids if i not in
                                   bod_jnts['left_arm_ids']]
    bod_jnts['not_left_arm_dofadrs'] = dof_conv(bod_jnts['not_left_arm_ids'])

    return bod_jnts

def get_joint_ids(model, data=None):
    jntn = lambda k: model.joint(k).name
    joints = {}
    joints['names'] = [jntn(k) for k in range(model.njnt)]
    joints['all_id_dict'] = {jntn(k): k for k in range(model.njnt)}
    # joints['all_dofadr_dict'] = convert_dofadr(model, data,
                                      # joints['all_id_dict'].values())
    joints['body'] = get_body_joints(model, data)
    joints['ball'] = [
        k for k in range(model.njnt) if 'ball' in jntn(k)
    ]
    joints['tennis'] = [
        k for k in range(model.njnt) if 'tennis' in jntn(k)
    ]
    return joints

def get_act_ids(model, data=None):
    acts = {}
    acts['act_names'] = [model.actuator(i).name for i in range(model.nu)]
    acts['all'] = [i for i in range(model.nu)]
    acts.update(get_act_names_left_or_right(model, data, 'right'))
    acts.update(get_act_names_left_or_right(model, data, 'left'))
    acts['adh'] = acts[f'adh_left_hand'] + acts[f'adh_right_hand']
    acts['adh'].sort()
    acts[f'not_adh'] = [k for k in acts['all'] if k not in acts['adh']]
    return acts

def get_act_names_left_or_right(model, data=None, left_or_right='right'):
    actn = lambda k: model.actuator(k).name
    act_names = [actn(k) for k in range(model.nu)]
    direct_dict = {'right': 'R_', 'left': 'L_'}
    acts = {}
    acts[f'{left_or_right}_arm'] = [
        model.actuator(name).id for name in act_names
        if key_match(name, arm_keys) and direct_dict[left_or_right] in name
    ]
    acts[f'not_{left_or_right}_arm'] = [
        model.actuator(name).id for name in act_names
        if model.actuator(name).id not in acts[f'{left_or_right}_arm']
    ]
    acts[f'adh_{left_or_right}_hand'] = [
        k for k in acts[f'{left_or_right}_arm'] if 'adh' in actn(k)
    ]
    acts[f'{left_or_right}_arm_without_adh'] = [
        k for k in acts[f'{left_or_right}_arm'] if k not in
        acts[f'adh_{left_or_right}_hand']
    ]
    return acts

class AdhCtrl:
    def __init__(self, t_zero_thrs=[], t_zero_ids=[], n_steps=10,
                 contact_check_list=[], adh_ids=[]):
        self.t_zero_thrs = t_zero_thrs
        self.tk = 0
        self.t_zero_thrs = t_zero_thrs
        self.t_zero_ids = t_zero_ids
        self.n_steps = n_steps
        self.contact_check_list = contact_check_list
        self.adh_ids = adh_ids
        self.ks = {k: 1 for k in adh_ids}

    def get_ctrl(self, model, data, ctrl):
        if len(self.adh_ids) == 0 or len(self.contact_check_list) == 0:
            return ctrl, None, None
        ctrl = ctrl.copy()
        ccl = self.contact_check_list
        adh_ids = self.adh_ids
        act = get_act_ids(model)
        contact_pairs = util.get_contact_pairs(model, data)
        adh_contact_ids = []
        for cp in contact_pairs:
            for k in range(len(ccl)):
                if ccl[k][0] in cp and ccl[k][1] in cp:
                    adh_id = adh_ids[k] 
                    if adh_id not in adh_contact_ids:
                        adh_contact_ids.append(adh_id)
                        ctrl[adh_id] = 1/self.n_steps * self.ks[adh_id]
                        if self.ks[adh_id] < self.n_steps:
                            self.ks[adh_id] += 1
        for k in range(len(self.t_zero_thrs)):
            if self.t_zero_thrs[k] is not None and self.tk >= self.t_zero_thrs[k]:
                ctrl[self.t_zero_ids[k]] = 0
            self.tk += 1
        return ctrl, None, None

    def reset(self):
        self.k_right = 0
        self.k_left = 0

def get_Q_balance(model, data, balance_cost, foot_cost):
    center_chest = 'Chest'
    left_foot = 'L_Ankle'
    right_foot = 'R_Ankle'
    nv = model.nv
    jac_com = np.zeros((3, nv))
    mj.mj_jacSubtreeCom(model, data, jac_com, model.body(center_chest).id)
    # Get the Jacobian for the left foot.
    jac_lfoot = np.zeros((3, nv))
    mj.mj_jacBodyCom(model, data, jac_lfoot, None,
                     model.body(left_foot).id)
    jac_rfoot = np.zeros((3, nv))
    mj.mj_jacBodyCom(model, data, jac_rfoot, None,
                     model.body(right_foot).id)
    jac_base = (jac_lfoot + jac_rfoot) / 2
    jac_diff = jac_com - jac_base
    Qbalance = balance_cost * jac_diff.T @ jac_diff 
    tmp = foot_cost * (jac_lfoot.T @ jac_lfoot + jac_rfoot.T @ jac_rfoot)
    Qbalance += tmp
    return Qbalance

def get_Q_joint(model, data=None, balance_joint_cost=3, other_joint_cost=.3,
                root_cost=0, excluded_acts=[]):
    joint_ids = get_joint_ids(model)
    joints = joint_ids['body']
    # z_joint = joint_ids['all']['human_z_root']
    # Construct the Qjoint matrix.
    Qjoint = np.eye(model.nv)
    # Qjoint[joints['root_dofs'], joints['root_dofs']] *= 0  # Don't penalize free joint directly.
    Qjoint[joints['root_dofadrs'], joints['root_dofadrs']] *= root_cost
    # Qjoint[z_joint, z_joint] = 100
    Qjoint[joints['balance_dofadrs'], joints['balance_dofadrs']] *= balance_joint_cost
    Qjoint[joints['other_dofadrs'], joints['other_dofadrs']] *= other_joint_cost
    Qjoint[excluded_acts, excluded_acts] *= 0
    return Qjoint

def get_Q_matrix(model, data, excluded_state_inds=[], balance_cost=1000,
                 joint_cost=100, root_cost=0, foot_cost=1000):
    # Cost coefficients.
    # balance_cost        = 1000  # Balancing.

    Qbalance = get_Q_balance(model, data, balance_cost, foot_cost)
    Qjoint = get_Q_joint(model, data, root_cost=root_cost,
                         excluded_acts=excluded_state_inds)
    # Construct the Q matrix for position DoFs.
    # Qpos = balance_cost * Qbalance + Qjoint
    # Qpos = balance_cost * Qbalance + 500*Qjoint
    Qpos = Qbalance + joint_cost*Qjoint
    # Qpos = 1000*Qjoint

    # No explicit penalty for velocities.
    nv = model.nv
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                  [np.zeros((nv, 2*nv))]])
    return Q

def get_feedback_ctrl_matrix_from_QR(model, data, Q, R, stable_jnt_ids,
                                     active_ctrl_ids):
    # Assumes that data.ctrl has been set to ctrl0 and data.qpos has been set
    # to qpos0.
    # data = copy.deepcopy(data)
    qvel = data.qvel.copy()
    nv = model.nv
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, model.nu))
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon_grad, flg_centered, A, B,
                        None, None)
    stable_ids = stable_jnt_ids + [i+nv for i in stable_jnt_ids]
    A = A[stable_ids][:, stable_ids]
    B = B[stable_ids][:, active_ctrl_ids]
    Q = Q[stable_ids][:, stable_ids]
    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

def get_feedback_ctrl_matrix(model, data, ctrl0, stable_jnt_ids,
                             active_ctrl_ids, balance_cost=1000,
                             joint_cost=100, root_cost=0, foot_cost=1000,
                             ctrl_cost=1):
    # What about data.qpos, data.qvel, data.qacc?
    # data = copy.deepcopy(data)
    data.ctrl[active_ctrl_ids] = ctrl0
    nq = model.nq
    nu = model.nu
    R = ctrl_cost*np.eye(len(active_ctrl_ids))
    Q = get_Q_matrix(model, data, balance_cost=balance_cost,
                     joint_cost=joint_cost, root_cost=root_cost,
                     foot_cost=foot_cost)
    K = get_feedback_ctrl_matrix_from_QR(model, data, Q, R, stable_jnt_ids,
                                         active_ctrl_ids)
    return K

def get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0, stable_jnt_ids):
    dq = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dq = dq[stable_jnt_ids]
    qvel = data.qvel[stable_jnt_ids]
    dx = np.concatenate((dq, qvel))
    return ctrl0 - K @ dx

def get_stabilized_ctrls(model, data, Tk, noisev, qpos0, ctrl_act_ids_dict,
                         stable_jnt_dofadrs_dict,
                         free_ctrls_dict=None,
                         K_update_interv=None, free_ctrl_fn=None,
                         balance_cost=1000, joint_cost=100,
                         root_cost=0,
                         foot_cost=1000,
                         ctrl_cost=1,
                         let_go_times=[], let_go_ids=[],
                         n_steps_adh=20,
                         contact_check_list=[], adh_ids=[], mask=None):
    """Get stabilized controls.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        Tk: Number of time steps.
        noisev: Noise vector.
        qpos0: Initial position.
        stable_jnt_dofadrs_dict: DOF adrs for joints that will be stabilized (kept
        from moving). Stored as values in a dict with keys corresponding to the
        values in mask.
        ctrl_act_ids_dict: IDs for actuators that will be used for stabilization
            control. Stored as values in a dict with keys corresponding to the
            values in mask.
        free_act_ids: IDs for actuators that will not be used for stabilization
            control.
        free_ctrls_dict: Free controls.
        K_update_interv: Update interval for K.
        """

    adh_ctrl = AdhCtrl(let_go_times, let_go_ids, n_steps_adh,
                       contact_check_list, adh_ids)

    data0 = copy.deepcopy(data)
    free_act_idx_dict = {}
    free_jnt_dofadrs_dict = {}
    for key in stable_jnt_dofadrs_dict:
        # idx = stable_jnt_dofadrs_dict[key]
        # free_jnt_dofadrs_dict[key] = [k for k in range(model.njnt) if
                                      # k not in idx]
        idx = ctrl_act_ids_dict[key]
        free_act_idx_dict[key] = [k for k in range(model.nu) if k not in idx]
    joints = get_joint_ids(model)
    acts = get_act_ids(model)
    # bodyj_id = joints['body']['body_dofs']
    # body_dof = convert_dofadr(model, None, bodyj_id, concat=True)
    if free_ctrls_dict is None:
        free_ctrls_dict = {key: np.zeros((Tk, len(item))) for key, item in free_act_idx_dict}
    if K_update_interv is None:
        K_update_interv = Tk+1
    qpos0n = qpos0.copy()
    qs = np.zeros((Tk, model.nq))
    qs[0] = data.qpos.copy()
    qvels = np.zeros((Tk, model.nv))
    qvels[0] = data.qvel.copy()
    ctrls = np.zeros((Tk-1, model.nu))
    if mask is None:
        mask = np.ones(Tk-1, dtype=bool)
    k_K_update = 0
    prev_bit = mask[0].item()
    for k in range(Tk-1):
        curr_bit = mask[k].item()
        idx1 = stable_jnt_dofadrs_dict[curr_bit]
        idx2 = ctrl_act_ids_dict[curr_bit]
        if curr_bit != prev_bit: # If we switch to new type of control
            k_K_update = 0 # Reset K_update counter
        if k_K_update % K_update_interv == 0:
            datak0 = copy.deepcopy(data)
            tmp = data.qpos.copy()
            tmp[idx1] = qpos0n[idx1].copy()
            # idx = free_jnt_dofadrs_dict[curr_bit]
            # qpos0n[idx] = data.qpos[idx].copy()
            ctrl0 = get_ctrl0(model, data, qpos0n, idx1, idx2)
            util.reset_state(model, data, datak0)
            K = get_feedback_ctrl_matrix(model, data, ctrl0, idx1, idx2,
                                         balance_cost, joint_cost, root_cost,
                                         foot_cost, ctrl_cost)
            util.reset_state(model, data, datak0)
            k_K_update += 1
        ctrl = get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0, idx1)
        ctrls[k][idx2] = ctrl
        # if free_ctrl_fn is not None:
            # ctrls[k][free_act_ids] = free_ctrl_fn(model, data, free_ctrls[k])
        # else:
            # ctrls[k][free_act_ids] = free_ctrls[k]
        ctrls[k][free_act_idx_dict[curr_bit]] = free_ctrls_dict[curr_bit][k]
        ctrls[k], __, __ = adh_ctrl.get_ctrl(model, data, ctrls[k])
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrls[k] + noisev[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
        prev_bit = curr_bit
    return ctrls, K, qs, qvels

def cum_mat_prod(tuple_of_mat):
    t = np.eye(tuple_of_mat[0].shape[0])
    rs = []
    for m in tuple_of_mat:
        t = t @ m
        rs.append(t)
    return rs

def cum_sum(tuple_of_mat):
    t = np.zeros(tuple_of_mat[0].shape[0])
    rs = []
    for m in tuple_of_mat:
        t = t + m
        rs.append(t)
    return rs

def traj_deriv_new2(model, data, ctrls, targ_trajs, targ_traj_masks,
                   q_targs, q_targ_masks,
                   grad_trunc_tk, deriv_sites, deriv_id_lists,
                   update_every=1, update_phase=0, grad_filter=True,
                   grab_time=None, let_go_time=None, n_steps_adh=10):
    """deriv_inds specifies the indices of the actuators that will be
    updated (for instance, the actuators related to the right arm)."""
    # data = copy.deepcopy(data)
    assert update_phase < update_every
    n = len(deriv_sites)
    Tk = ctrls.shape[0]+1
    grad_range = range(update_phase, Tk, update_every)
    Tkn = grad_range[-1]
    Bs = []
    fixed_act_id_list = []
    for k in range(n):
        nuderiv = len(deriv_id_lists[k])
        Bs.append(np.zeros((Tk-1, 2*model.nv, nuderiv)))
        fixed_act_id_list.append([i for i in range(model.nu) if i not in
                                  deriv_id_lists[k]])
    As = np.zeros((Tk-1, 2*model.nv, 2*model.nv))
    B = np.zeros((2*model.nv, model.nu))
    C = np.zeros((3, model.nv))
    dldqs = np.zeros((n, Tk, 2*model.nv))
    lams = np.zeros((n, Tk, 2*model.nv))

    adh_ctrl = AdhCtrl(let_go_time, contact_check_list, adh_ids, n_steps_adh)

    # q_targ_mask_flat = np.sum(q_targ_mask, axis=1) > 0
    targ_traj_mask_any = np.any(np.stack(targ_traj_masks), axis=0)

    for tk in range(Tk):
        mj.mj_forward(model, data)
        if tk in grad_range and targ_traj_mask_any[tk]:
            if tk < Tk-1:
                mj.mjd_transitionFD(
                    model, data, epsilon_grad, True, As[tk], B, None, None
                )
        for k, deriv_site in enumerate(deriv_sites):
            if tk in grad_range and targ_traj_masks[k][tk]:
                mj.mj_jacSite(
                    model, data, C, None, site=data.site(deriv_site).id)
                site_xpos = data.site(deriv_site).xpos
                dlds = site_xpos - targ_trajs[k][tk]
                dldq = C.T @ dlds
                dldqs[k, tk, :model.nv] = dldq
                Bs[k][tk] = np.delete(B, fixed_act_id_list[k], axis=1)
        # if tk in grad_range and q_targ_mask_flat[tk]:
        if tk in grad_range:
            qnow = np.concatenate((data.qpos[:], data.qvel[:]))
            for k in range(n):
                q_targ_mask = q_targ_masks[k]
                dldq = qnow - q_targs[k][tk]
                dldqs[k, tk] += dldq * q_targ_mask[tk]
        if tk < Tk-1:
            ctrls[tk] = adh_ctrl.get_ctrl(model, data, ctrls[tk],
                                          contact_check_list, act_ids)[0]
            sim_util.step(model, data, ctrls[tk])
    # print(As[1630])
    # print(Bs[0][1630])
    # print(dldqs[0, 1630])
    grads = np.zeros((n, Tk-1, nuderiv))
    loss_us = []
    for k in range(n):
        loss_us.append(np.delete(ctrls, fixed_act_id_list[k], axis=1))
        lams[k, tk] = dldqs[k, tk]
    
    tau_loss_factor = 1e-9
    for tk in reversed(grad_range[1:]):
        tks = tk - update_every # Shifted by one update
        term_lists = [[]]*n
        for k in range(n):
            terms = term_lists[k]
            terms.insert(0, dldqs[k, tk])
            while len(terms) > grad_trunc_tk:
                terms.pop()
            At = As[tks].T
            terms = [At @ term for term in terms]
            if grad_filter:
                if targ_traj_masks[k][tk]:
                    grads[k, tks] = tau_loss_factor*loss_us[k][tks] \
                        + Bs[k][tks].T @ lams[k, tk]
            else:
                grads[k, tks] = tau_loss_factor*loss_us[k][tks] \
                    + Bs[k][tks].T @ lams[k, tk]
        # if np.sum(np.abs(lams[k, tks])) > 0:
    # fig, ax = plt.subplots()
    # nrms = np.linalg.norm(grads, axis=1)
    # ax.plot(nrms)
    # plt.show()

    mat_block = np.zeros((update_every, update_every+1))
    dk = 1/update_every
    v = np.arange(dk, 1+dk, dk)
    mat_block[:, 0] = v[::-1]
    mat_block[1:, update_every] = v[:-1]
    # if update_phase > 0:
    n_complete_blocks = (Tk-1) // update_every + 1
    last_block_size = (Tk-1) % update_every - update_phase
    first_block_size = update_phase
    # As = n_complete_blocks * update_every + last_block_size + first_block_size
    An = Tk-1
    A = np.zeros((An, An))
    A[:update_phase, update_phase] = 1
    for k in range(0, An-update_every-update_phase, update_every):
        ks = k + update_phase
        A[ks:ks+update_every, ks:ks+update_every+1] = mat_block
    A[ks+update_every:, ks+update_every] = 1

    grads_interp = np.zeros((n, Tk-1, nuderiv))
    for k in range(n):
        grads_interp[k] = A @ grads[k]

    return grads_interp

def centered_diff_2_norm_der(u, dt, axis=0):
    """Derivative of the 2-norm of the centered difference derivative
    approximation of u, with respect to u. TODO: change to work with axis other
    than 0."""
    assert axis == 0
    u_r = np.roll(u, 1, axis=axis)
    u_l = np.roll(u, -1, axis=axis)
    u_diff = u_l - u_r
    # Derivative away from the boundaries
    deriv = np.roll(u_diff, 1, axis=axis) - np.roll(u_diff, -1, axis=axis)
    deriv = deriv / (2*dt)
    # Now to deal with the boundaries TODO: change to work with axis other than
    # 0
    deriv[0] = -(u[1]-u[0]) - (u[2]-u[0])
    deriv[0] = deriv[0] / dt
    deriv[1] = -(u[1]-u[0]) - (u[3]-u[1])
    deriv[1] = deriv[1] / dt
    deriv[-2] = u[-2]-u[-4] - (u[-1]-u[-2])
    deriv[-2] = deriv[-2] / dt
    deriv[-1] = u[-1]-u[-3] + u[-1]-u[-2]
    deriv[-1] = deriv[-1] / dt
    return deriv



### Gradient descent
def traj_deriv_new(model, data, ctrls, targ_traj, targ_traj_mask,
                   q_targ, q_targ_mask,
                   grad_trunc_tk,
                   deriv_ids=[], deriv_site='hand_right',
                   update_every=1, update_phase=0, grad_filter=True,
                   grab_time=None, let_go_times=None,
                   let_go_ids=None, n_steps_adh=10,
                   contact_check_list=None, adh_ids=None,
                   ctrl_reg_weight=None
                  ):
    """deriv_inds specifies the indices of the actuators that will be
    updated (for instance, the actuators related to the right arm)."""
    # data = copy.deepcopy(data)
    assert update_phase < update_every
    nq = model.nq
    nv = model.nv
    syssize1 = nv + nq
    syssize2 = 2*nv + model.na
    syssize3 = 2*nq
    nuderiv = len(deriv_ids)
    if ctrl_reg_weight is None:
        ctrl_reg_weight = np.ones((ctrls.shape[0], nuderiv))
    Tk = ctrls.shape[0]+1
    grad_range = range(update_phase, Tk, update_every)
    Tkn = grad_range[-1]
    As = np.zeros((Tk-1, syssize2, syssize2))
    Bs = np.zeros((Tk-1, syssize2, nuderiv))
    B = np.zeros((syssize2, model.nu))
    C = np.zeros((3, nv))
    dq = np.zeros(nv)
    dldqs = np.zeros((Tk, syssize2))
    dldss = np.zeros((Tk, 3))
    lams = np.zeros((Tk, syssize2))
    lams2 = np.zeros((Tk, syssize2))
    lams3 = np.zeros((Tk, syssize2))
    fixed_act_ids = [i for i in range(model.nu) if i not in deriv_ids]
    hxs = np.zeros((Tk, 3))

    adh_ctrl = AdhCtrl(let_go_times, let_go_ids, n_steps_adh,
                       contact_check_list, adh_ids)

    q_targ_mask_flat = np.sum(q_targ_mask, axis=1) > 0

    qs = np.zeros((Tk, syssize1))
    for tk in range(Tk):
        if tk in grad_range and targ_traj_mask[tk]:
            mj.mj_forward(model, data)
            mj.mj_jacSite(
                model, data, C, None, site=data.site(f'{deriv_site}').id)
            site_xpos = data.site(f'{deriv_site}').xpos
            dlds = site_xpos - targ_traj[tk]
            dldss[tk] = dlds
            hxs[tk] = site_xpos
            dldq = C.T @ dlds
            dldqs[tk, :nv] = dldq
            if tk < Tk-1:
                mj.mjd_transitionFD(
                    model, data, epsilon_grad, True, As[tk], B, None, None
                )
                Bs[tk] = np.delete(B, fixed_act_ids, axis=1)
        # if tk in grad_range and q_targ_mask_flat[tk]:
        qnow = np.concatenate((data.qpos[:], data.qvel[:]))
        qs[tk] = qnow
        # dldq = qnow - q_targ[tk]
        mj.mj_differentiatePos(model, dq, 1,
                               data.qpos*q_targ_mask[tk, :nq],
                               q_targ[tk, :nq]*q_targ_mask[tk, :nq])
        dqvel = (qnow[nq:] - q_targ[tk, nq:]) * q_targ_mask[tk, nq:]
        dqfull = np.concatenate((dq, dqvel))
        dldqs[tk] += dqfull
        
        if tk < Tk-1:
            if contact_check_list is not None:
                ctrls[tk], __, __ = adh_ctrl.get_ctrl(
                    model, data, ctrls[tk],
                )
            sim_util.step(model, data, ctrls[tk])
    # qdots = qs[:, nq:]
    # qaccs = np.gradient(qdots, axis=0)
    qsfft = np.fft.fft(qs[:, :nq], axis=0)
    freqs = np.tile(np.fft.fftfreq(Tk).reshape(-1, 1), (1, nq))
    # regularizer = np.sum((freqs**2) * np.abs(qsfft)**2)
    grad_regularizer = np.fft.ifft(2 * (freqs**2) * qsfft, axis=0).real
    # dldqs[:, :nq] += 1e3*grad_regularizer
    lams[tk] = dldqs[tk]
    grads = np.zeros((Tk-1, nuderiv))
    # tau_loss_factor = 1e-7
    ctrls_clip = np.delete(ctrls, fixed_act_ids, axis=1)
    loss_u = 1e-6*ctrls_clip.copy() * ctrl_reg_weight

    ufft = np.fft.fft(ctrls_clip, axis=0)
    # freqs = np.tile(np.fft.fftfreq(Tk).reshape(-1, 1), (1, nq))
    # regularizer = np.sum((freqs**2) * np.abs(ufft)**2)
    freqs = np.tile(np.fft.fftfreq(Tk-1).reshape(-1, 1), (1, nuderiv))
    grad_regularizer = np.fft.ifft(2 * (freqs**2) * ufft, axis=0).real
    # loss_u += grad_regularizer # endpoints large -- maybe remove

    dt = model.opt.timestep
    
    # shift ctrls two places
    deriv = centered_diff_2_norm_der(ctrls_clip, dt) * ctrl_reg_weight
    loss_u += 1e-5 * deriv

    # deriv = centered_diff_2_norm_der(qs[:, :nq], dt)
    # deriv *= (q_targ_mask[:, :nq]>0)
    # dldqs[:, :nq] += 1e2 * deriv
    
    terms = []
    for tk in reversed(grad_range[1:]):
        tks = tk - update_every # Shifted by one update
        terms.insert(0, dldqs[tk])
        while len(terms) > grad_trunc_tk:
            terms.pop()
        At = As[tks].T
        terms = [At @ term for term in terms]
        lams[tks] = dldqs[tks] + np.sum(terms, axis=0)
        if grad_filter and targ_traj_mask[tk]:
            grads[tks] = loss_u[tks] + Bs[tks].T @ lams[tk]

    mat_block = np.zeros((update_every, update_every+1))
    dk = 1/update_every
    v = np.arange(dk, 1+dk, dk)
    mat_block[:, 0] = v[::-1]
    mat_block[1:, update_every] = v[:-1]
    # if update_phase > 0:
    n_complete_blocks = (Tk-1) // update_every + 1
    last_block_size = (Tk-1) % update_every - update_phase
    first_block_size = update_phase
    # As = n_complete_blocks * update_every + last_block_size + first_block_size
    An = Tk-1
    A = np.zeros((An, An))
    A[:update_phase, update_phase] = 1
    for k in range(0, An-update_every-update_phase, update_every):
        ks = k + update_phase
        A[ks:ks+update_every, ks:ks+update_every+1] = mat_block
    A[ks+update_every:, ks+update_every] = 1

    grads_interp = A @ grads

    return grads_interp

def reset_with_lqr(env, seed, nsteps1, nsteps2, balance_cost, joint_cost,
                   root_cost, foot_cost, ctrl_cost):
    env.reset(seed=seed, options={'n_steps': nsteps1, 'render': False})
    model = env.model
    data = env.data
    noisev = np.zeros((nsteps2, model.nu))
    joints = get_joint_ids(model)
    acts = get_act_ids(model)
    body_dof = joints['body']['dofadrs_without_root']
    # body_dof = joints['body']['dofadrs'] # This is unstable for some reason
    # breakpoint()
    mask = np.ones((nsteps2,), dtype=bool)
    ctrls = get_stabilized_ctrls(
        model, data, nsteps2, noisev, data.qpos.copy(), {1: acts['not_adh']},
        {1: body_dof}, free_ctrls_dict={1: np.ones((nsteps2, len(acts['adh'])))},
        balance_cost=balance_cost, joint_cost=joint_cost, root_cost=root_cost,
        foot_cost=foot_cost, ctrl_cost=ctrl_cost, mask=mask
    )[0]
    ctrls = np.vstack((np.zeros((nsteps1, model.nu)), ctrls))
    return ctrls
