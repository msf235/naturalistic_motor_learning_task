import mujoco as mj
import time
import sys
import numpy as np
import scipy
import control_logic as cl
import sim_util as util
import humanoid2d as h2d
import copy
import sim_util
import optimizers as opts

epsilon_grad = 5e-9

class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        ## update weights and biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w, b

### LQR
def get_ctrl0(model, data, qpos0, stable_jnt_ids, ctrl_act_ids):
    # data = copy.deepcopy(data)
    # data.qpos[:] = qpos0.copy()
    mj.mj_forward(model, data)
    data.qacc[:] = 0
    data.qvel[:] = 0
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()[stable_jnt_ids]
    # M = data.actuator_moment[:, stable_jnt_ids]
    M = data.actuator_moment[ctrl_act_ids][:, stable_jnt_ids]
    # Probably much better way to do this
    # ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(M)
    ctrl0 = np.linalg.lstsq(M.T, qfrc0, rcond=None)[0]
    # ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    # breakpoint()
    return ctrl0

body_keys = ['human', 'shoulder', 'hand', 'torso', 'hip', 'knee', 'ankle',
             'abdomen', 'elbow', 'arm', 'wrist'] 

def key_match(key, key_list):
    for k in key_list:
        if k in key:
            return True
    return False

def get_body_joints(model, data=None):
    """Get joint names for body."""
    body_keys = ['human', 'shoulder', 'torso', 'hip', 'knee', 'ankle',
                 'abdomen', 'elbow', 'arm', 'wrist'] 
    jntn = lambda k: model.joint(k).name

    joints = {}
    joints['all'] = [k for k in range(model.nq) if
                      key_match(jntn(k), body_keys)]
    jba = joints['all']
    # Get indices into relevant sets of joints.
    joints['root_dofs'] = [k for k in jba if 'root' in jntn(k)]
    joints['body_dofs'] = [k for k in jba if k not in joints['root_dofs']]
    jb = joints['body_dofs']

    joints['abdomen_dofs'] = [
        k for k in jb if 'abdomen' in jntn(k) and not 'z' in jntn(k)
    ]
    joints['leg_dofs'] = [
        k for k in jb if key_match(jntn(k), ['hip', 'knee', 'ankle'])
    ]
    joints['balance_dofs'] = joints['abdomen_dofs'] + joints['leg_dofs']
    joints['balance_dofs'].sort()
    joints['other_dofs'] = [k for k in jb if k not in joints['balance_dofs']]
    joints['right_arm'] = [
        k for k in jb if key_match(jntn(k), ['shoulder', 'elbow', 'wrist'])
        and 'right' in jntn(k)
    ]
    joints['left_arm'] = [
        k for k in jb if key_match(jntn(k), ['shoulder', 'elbow', 'wrist'])
        and 'left' in jntn(k)
    ]

    joints['not_right_arm'] = [i for i in jb if i not in joints['right_arm']]
    joints['not_left_arm'] = [i for i in jb if i not in joints['left_arm']]

    return joints

def get_joint_ids(model, data=None):
    jntn = lambda k: model.joint(k).name
    joints = {}
    joints['joint_names'] = [jntn(k) for k in range(model.njnt)]
    joints['all'] = {jntn(k): k for k in range(model.njnt)}
    joints['body'] = get_body_joints(model, data)
    joints['ball'] = [
        k for k in range(model.nq) if 'ball' in jntn(k)
    ]
    joints['tennis'] = [
        k for k in range(model.nq) if 'tennis' in jntn(k)
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
    acts = {}
    acts[f'{left_or_right}_arm'] = [
        model.actuator(name).id
        for name in act_names
        if ('shoulder' in name or 'elbow' in name or 'hand' in name or 'wrist'
            in name)
        and left_or_right in name
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
    def __init__(self, t_zero_thr):
        self.k_right = 1
        self.k_left = 1
        self.tk = 0
        self.t_zero_thr = t_zero_thr
        self.n_steps = 10

    def get_ctrl(self, model, data, ctrl):
        act = get_act_ids(model)
        # adh = opt_utils.get_act_ids(model)['adh']
        contact_pairs = util.get_contact_pairs(model, data)
        contact1 = False
        contact2 = False
        for cp in contact_pairs:
            if 'racket_handle' in cp and ('hand_right1' in cp or 'hand_right2' in cp):
                contact1 = True
                ctrl[act['adh_right_hand']] = 1/self.n_steps * self.k_right
                if self.k_right < self.n_steps:
                    self.k_right += 1
            if 'ball' in cp and ('hand_left1' in cp or 'hand_left2' in cp):
                contact2 = True
                ctrl[act['adh_left_hand']] = 1/self.n_steps * self.k_left
                if self.k_left < self.n_steps:
                    self.k_left += 1
        if self.t_zero_thr is not None and self.tk >= self.t_zero_thr:
            ctrl[act['adh_left_hand']] = 0
        self.tk += 1
        return ctrl, contact1, contact2

    def reset(self):
        self.k_right = 0
        self.k_left = 0

def get_Q_balance(model, data):
    nq = model.nq
    jac_com = np.zeros((3, nq))
    mj.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)
    # Get the Jacobian for the left foot.
    jac_lfoot = np.zeros((3, nq))
    mj.mj_jacBodyCom(model, data, jac_lfoot, None,
                     model.body('foot_left').id)
    jac_rfoot = np.zeros((3, nq))
    mj.mj_jacBodyCom(model, data, jac_rfoot, None,
                     model.body('foot_right').id)
    jac_base = (jac_lfoot + jac_rfoot) / 2
    jac_diff = jac_com - jac_base
    Qbalance = jac_diff.T @ jac_diff
    return Qbalance

def get_Q_joint(model, data=None, excluded_acts=[]):
    balance_joint_cost  = 3     # Joints required for balancing.
    other_joint_cost    = .3    # Other joints.
    joint_ids = get_joint_ids(model)
    joints = joint_ids['body']
    # z_joint = joint_ids['all']['human_z_root']
    # Construct the Qjoint matrix.
    Qjoint = np.eye(model.nq)
    # Qjoint[joints['root_dofs'], joints['root_dofs']] *= 0  # Don't penalize free joint directly.
    Qjoint[joints['root_dofs'], joints['root_dofs']] *= 3
    # Qjoint[z_joint, z_joint] = 100
    Qjoint[joints['balance_dofs'], joints['balance_dofs']] *= balance_joint_cost
    Qjoint[joints['other_dofs'], joints['other_dofs']] *= other_joint_cost
    Qjoint[excluded_acts, excluded_acts] *= 0
    return Qjoint

def get_Q_matrix(model, data, excluded_state_inds=[]):
    # Cost coefficients.
    balance_cost        = 1000  # Balancing.

    Qbalance = get_Q_balance(model, data)
    Qjoint = get_Q_joint(model, data, excluded_state_inds)
    # Construct the Q matrix for position DoFs.
    Qpos = balance_cost * Qbalance + Qjoint
    Qpos = balance_cost * Qbalance + 500*Qjoint
    # Qpos = 1000*Qjoint

    # No explicit penalty for velocities.
    nq = model.nq
    Q = np.block([[Qpos, np.zeros((nq, nq))],
                  [np.zeros((nq, 2*nq))]])
    return Q

def get_feedback_ctrl_matrix_from_QR(model, data, Q, R, stable_jnt_ids,
                                     active_ctrl_ids):
    # Assumes that data.ctrl has been set to ctrl0 and data.qpos has been set
    # to qpos0.
    # data = copy.deepcopy(data)
    qvel = data.qvel.copy()
    nq = model.nq
    A = np.zeros((2*nq, 2*nq))
    B = np.zeros((2*nq, model.nu))
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon_grad, flg_centered, A, B,
                        None, None)
    stable_ids = stable_jnt_ids + [i+nq for i in stable_jnt_ids]
    A = A[stable_ids][:, stable_ids]
    B = B[stable_ids][:, active_ctrl_ids]
    Q = Q[stable_ids][:, stable_ids]
    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

def get_feedback_ctrl_matrix(model, data, ctrl0, stable_jnt_ids,
                             active_ctrl_ids):
    # What about data.qpos, data.qvel, data.qacc?
    # data = copy.deepcopy(data)
    data.ctrl[active_ctrl_ids] = ctrl0
    nq = model.nq
    nu = model.nu
    R = np.eye(len(active_ctrl_ids))
    Q = get_Q_matrix(model, data)
    K = get_feedback_ctrl_matrix_from_QR(model, data, Q, R, stable_jnt_ids,
                                         active_ctrl_ids)
    return K

def get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0, stable_jnt_ids):
    dq = np.zeros(model.nq)
    mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dq = dq[stable_jnt_ids]
    qvel = data.qvel[stable_jnt_ids]
    dx = np.concatenate((dq, qvel))
    return ctrl0 - K @ dx

def get_stabilized_ctrls(model, data, Tk, noisev, qpos0, ctrl_act_ids,
                         stable_jnt_ids, free_ctrls=None,
                         K_update_interv=None, free_ctrl_fn=None,
                         let_go_time=None):
    """Get stabilized controls.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        Tk: Number of time steps.
        noisev: Noise vector.
        qpos0: Initial position.
        ctrl_act_ids: IDs for actuators that will be used for stabilization
            control.
        free_act_ids: IDs for actuators that will not be used for stabilization
            control.
        stable_jnt_ids: IDs for joints that will be stabilized (kept from
            moving).
        free_ctrls: Free controls.
        K_update_interv: Update interval for K.
        """

    adh_ctrl = AdhCtrl(let_go_time)

    data0 = copy.deepcopy(data)
    free_act_ids = [k for k in range(model.nu) if k not in ctrl_act_ids]
    free_jnt_ids = [k for k in range(model.njnt) if k not in stable_jnt_ids]
    if free_ctrls is None:
        free_ctrls = np.zeros((Tk, len(free_act_ids)))
    if K_update_interv is None:
        K_update_interv = Tk+1
    qpos0n = qpos0.copy()
    qs = np.zeros((Tk, model.nq))
    qs[0] = data.qpos.copy()
    qvels = np.zeros((Tk, model.nq))
    qvels[0] = data.qvel.copy()
    ctrls = np.zeros((Tk-1, model.nu))
    for k in range(Tk-1):
        if k % K_update_interv == 0:
            datak0 = copy.deepcopy(data)
            qpos0n[free_jnt_ids] = data.qpos[free_jnt_ids]
            ctrl0 = get_ctrl0(model, data, qpos0n, stable_jnt_ids,
                              ctrl_act_ids)
            util.reset_state(model, data, datak0)
            K = get_feedback_ctrl_matrix(model, data, ctrl0, stable_jnt_ids,
                                         ctrl_act_ids)
            util.reset_state(model, data, datak0)
        ctrl = get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0,
                                   stable_jnt_ids)
        ctrls[k][ctrl_act_ids] = ctrl
        # if free_ctrl_fn is not None:
            # ctrls[k][free_act_ids] = free_ctrl_fn(model, data, free_ctrls[k])
        # else:
            # ctrls[k][free_act_ids] = free_ctrls[k]
        ctrls[k][free_act_ids] = free_ctrls[k]
        ctrls[k], __, __ = adh_ctrl.get_ctrl(model, data, ctrls[k])
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrls[k] + noisev[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
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

### Gradient descent
def traj_deriv_new(model, data, ctrls, targ_traj, targ_traj_mask,
                   grad_trunc_tk, deriv_ids=[], deriv_site='hand_right',
                   update_every=1, update_phase=0, grad_filter=True,
                   grab_time=None, let_go_time=None):
    """deriv_inds specifies the indices of the actuators that will be
    updated (for instance, the actuators related to the right arm)."""
    # data = copy.deepcopy(data)
    assert update_phase < update_every
    Tk = ctrls.shape[0]+1
    grad_range = range(update_phase, Tk, update_every)
    Tkn = grad_range[-1]
    nuderiv = len(deriv_ids)
    As = np.zeros((Tk-1, 2*model.nv, 2*model.nv))
    Bs = np.zeros((Tk-1, 2*model.nv, nuderiv))
    B = np.zeros((2*model.nv, model.nu))
    C = np.zeros((3, model.nv))
    dldqs = np.zeros((Tk, 2*model.nv))
    dldss = np.zeros((Tk, 3))
    lams = np.zeros((Tk, 2*model.nv))
    lams2 = np.zeros((Tk, 2*model.nv))
    lams3 = np.zeros((Tk, 2*model.nv))
    fixed_act_ids = [i for i in range(model.nu) if i not in deriv_ids]
    hxs = np.zeros((Tk, 3))

    adh_ctrl = AdhCtrl(let_go_time)

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
            dldqs[tk, :model.nv] = dldq
            if tk < Tk-1:
                mj.mjd_transitionFD(
                    model, data, epsilon_grad, True, As[tk], B, None, None
                )
                Bs[tk] = np.delete(B, fixed_act_ids, axis=1)
        
        if tk < Tk-1:
            ctrls[tk], __, __ = adh_ctrl.get_ctrl(model, data, ctrls[tk])
            sim_util.step(model, data, ctrls[tk])
    # Ast = [A.T for A in As]
    # Aprods = cum_mat_prod(As)
    # Aprods.insert(0, np.eye(As[0].shape[0]))
    # terms = [Aprods[k]@dldqs[k] for k in range(Tk)]
    # lams2 = cum_sum(terms)
    # n = 1*(len(deriv_site) < 5)
    # print(deriv_site + '\t\t' + '\t'*n + str(np.max(np.abs(As))))
    # ttm = targ_traj_mask.reshape(-1, 1)
    # dldqs = dldqs * ttm
    # lams[-1] = dldqs[-1]
    lams[tk] = dldqs[tk]
    # lams2[tk] = dldqs[tk]
    # lams3[tk] = dldqs[tk]
    grads = np.zeros((Tk-1, nuderiv))
    tau_loss_factor = 1e-9
    # tau_loss_factor = 0
    loss_u = np.delete(ctrls, fixed_act_ids, axis=1)
    
    # time.tic()
    # terms = [dldqs[tk]]
    # terms2 = [dldqs[tk]]
    # from matplotlib import pyplot as plt
    # plt.close('all')
    # fig, ax = plt.subplots()
    # fig.show()
    terms = []
    for tk in reversed(grad_range[1:]):
        tks = tk - update_every # Shifted by one update
        # lams[tks] = dldqs[tks] + As[tks].T @ lams[tk]
        terms.insert(0, dldqs[tk])
        while len(terms) > grad_trunc_tk:
            terms.pop()
        At = As[tks].T
        terms = [At @ term for term in terms]
        # print(np.linalg.norm(terms[0]), np.linalg.norm(terms[-1]))
        # nrm_terms = np.linalg.norm(terms, axis=1)
        # ax.cla()
        # ax.plot(nrm_terms)
        # plt.pause(1)
        # mprods = cum_mat_prod(Ast[tks:])
        # terms = [mat@lam for mat, lam in zip(mprods, dldqs[tk:])]
        # lams2[tks] = dldqs[tks] + np.sum(terms, axis=0)
        lams[tks] = dldqs[tks] + np.sum(terms, axis=0)
        # print(np.linalg.norm(lams[tks]))
        if grad_filter and targ_traj_mask[tk]:
            grads[tks] = tau_loss_factor*loss_u[tks] + Bs[tks].T @ lams[tk]
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

    grads_interp = A @ grads

    return grads_interp
    # return grads, hxs, dldss

def reset(model, data, nsteps1, nsteps2, keyframe_name=None):
    if keyframe_name is not None:
        keyframe_id = model.keyframe(keyframe_name).id
        mj.mj_resetDataKeyframe(model, data, keyframe_id)
    else:
        mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    for k in range(nsteps1):
        mj.mj_step(model, data)
    noisev = np.zeros((nsteps2, model.nu))
    joints = get_joint_ids(model)
    acts = get_act_ids(model)
    bodyj = joints['body']['body_dofs']
    get_stabilized_ctrls(
        model, data, nsteps2, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj, free_ctrls=np.ones((nsteps2, len(acts['adh'])))
    )[0]
