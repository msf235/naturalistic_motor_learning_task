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

def get_joint_names(model, data=None):
    """Get joint names for body."""
    joints = {}
    joints['joint_names'] = [model.joint(i).name for i in range(model.njnt)]

    # Get indices into relevant sets of joints.
    joints['root_dofs'] = range(3)
    joints['body_dofs'] = range(3, model.nq)
    joints['abdomen_dofs'] = [
        model.joint(name).dofadr[0]
        for name in joints['joint_names']
        if 'abdomen' in name
        and not 'z' in name
    ]
    joints['leg_dofs'] = [
        model.joint(name).dofadr[0]
        for name in joints['joint_names']
        if ('hip' in name or 'knee' in name or 'ankle' in name)
        and not 'z' in name
    ]
    joints['balance_dofs'] = joints['abdomen_dofs'] + joints['leg_dofs']
    joints['other_dofs'] = np.setdiff1d(joints['body_dofs'],
                                        joints['balance_dofs'])
    joints['right_arm'] = [
        model.joint(name).dofadr[0] for name in joints['joint_names']
        if ('shoulder' in name or 'elbow' in name) and 'right' in name
    ]
    joints['left_arm'] = [
        model.joint(name).dofadr[0] for name in joints['joint_names']
        if ('shoulder' in name or 'elbow' in name) and 'left' in name
    ]
    ball_jnts = [
        model.joint(name).dofadr[0] for name in joints['joint_names']
        if 'ball' in name
    ]

    joints['ball_jnts'] = ball_jnts

    joints['non_right_arm'] = [i for i in range(model.nq) if i not
                                in joints['right_arm']+ball_jnts]
    joints['non_left_arm'] = [i for i in range(model.nq) if i not
                                in joints['left_arm']+ball_jnts]
    joints['body'] = [k for k in range(model.nq) if k not in ball_jnts]
    return joints

def get_act_names(model, data=None):
    acts = {}
    acts['act_names'] = [model.actuator(i).name for i in range(model.nu)]
    acts.update(get_act_names_left_or_right(model, data, 'right'))
    acts.update(get_act_names_left_or_right(model, data, 'left'))
    acts[f'non_adh'] = [
        i for i in range(model.nu) if i not in
        acts[f'adh_left_hand'] + acts[f'adh_right_hand']
    ]
    return acts

def get_act_names_left_or_right(model, data=None, left_or_right='right'):
    act_names = [model.actuator(i).name for i in range(model.nu)]
    acts = {}
    acts[f'{left_or_right}_arm'] = [
        model.actuator(name).id
        for name in act_names
        if ('shoulder' in name or 'elbow' in name) and left_or_right in name
    ]
    try:
        acts[f'adh_{left_or_right}_hand'] = [
            model.actuator(name).id
            for name in act_names
            if 'hand' in name and left_or_right in name and 'adh' in name
        ]
        acts[f'{left_or_right}_arm_with_adh'] = \
            acts[f'{left_or_right}_arm'] + acts[f'adh_{left_or_right}_hand']
    except KeyError:
        acts[f'adh_{left_or_right}_hand'] = []
    acts[f'non_adh_{left_or_right}_hand'] = [
        i for i in range(model.nu) if i not in
        acts[f'adh_{left_or_right}_hand']
    ]
    acts[f'non_{left_or_right}_arm_non_adh'] = [
        i for i in range(model.nu) if i not in acts[f'{left_or_right}_arm'] and
        i not in acts[f'adh_{left_or_right}_hand']]
    acts[f'non_{left_or_right}_arm'] = [i for i in range(model.nu) if i not in
                               acts[f'{left_or_right}_arm']]
    return acts

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
    joints = get_joint_names(model)
    # Construct the Qjoint matrix.
    Qjoint = np.eye(model.nq)
    Qjoint[joints['root_dofs'], joints['root_dofs']] *= 0  # Don't penalize free joint directly.
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
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B,
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
                         K_update_interv=None,):
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
            util.reset_state(data, datak0)
            K = get_feedback_ctrl_matrix(model, data, ctrl0, stable_jnt_ids,
                                         ctrl_act_ids)
            util.reset_state(data, datak0)
        ctrl = get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0,
                                   stable_jnt_ids)
        ctrls[k][ctrl_act_ids] = ctrl
        ctrls[k][free_act_ids] = free_ctrls[k]
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrls[k] + noisev[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
    return ctrls, K, qs, qvels

### Gradient descent
def traj_deriv(model, data, ctrls, targ_traj, targ_traj_mask,
               grad_trunc_tk, fixed_act_inds=[], right_or_left='right'):
    """fixed_act_inds specifies the indices of the actuators that will NOT be
    updated (for instance, the actuators not related to the right arm)."""
    # data = copy.deepcopy(data)
    nufree = model.nu - len(fixed_act_inds)
    Tk = ctrls.shape[0]
    As = np.zeros((Tk+1, 2*model.nv, 2*model.nv))
    Bs = np.zeros((Tk+1, 2*model.nv, nufree))
    B = np.zeros((2*model.nv, model.nu))
    C = np.zeros((3, model.nv))
    dldqs = np.zeros((Tk, 2*model.nv))
    dldss = np.zeros((Tk, 3))
    lams = np.zeros((Tk, 2*model.nv))
    not_fixed_act_inds = [i for i in range(model.nu) if i not in
                          fixed_act_inds]
    epsilon = 1e-6
    hxs = np.zeros((Tk, 3))

    for tk in range(Tk):
        mj.mj_forward(model, data)
        mj.mjd_transitionFD(model, data, epsilon, True, As[tk], B, None, None)
        Bs[tk] = np.delete(B, fixed_act_inds, axis=1)
        mj.mj_jacSite(
            model, data, C, None, site=data.site(f'hand_{right_or_left}').id)
        dlds = data.site(f'hand_{right_or_left}').xpos - targ_traj[tk]
        dldss[tk] = dlds
        hxs[tk] = data.site(f'hand_{right_or_left}').xpos
        dldq = C.T @ dlds
        dldqs[tk, :model.nv] = dldq
        
        if tk < Tk-1:
            sim_util.step(model, data, ctrls[tk])

    ttm = targ_traj_mask.reshape(-1, 1)
    dldqs = dldqs * ttm
    lams[-1] = dldqs[-1]
    grads = np.zeros((Tk, nufree))
    # tau_loss_factor = 1e-9
    tau_loss_factor = 0
    loss_u = np.delete(ctrls, fixed_act_inds, axis=1)

    for tk in range(2, Tk): # Go backwards in time
        lams[Tk-tk] = dldqs[Tk-tk] + As[Tk-tk].T @ lams[Tk-tk+1]
        # grads[Tk-tk] = (tau_loss_factor/tk**.5)*loss_u[Tk-tk] \
        grads[Tk-tk] = tau_loss_factor*loss_u[Tk-tk] \
                + Bs[Tk-tk].T @ lams[Tk-tk+1]
    return grads, hxs, dldss

def get_final_loss(model, data, xpos1, xpos2):
    # I could put a forward sim here for safety (but less efficient)
    dlds = xpos1 - xpos2
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq # 11 and 12 are currently right shoulder and elbow
    loss = .5*np.mean(dlds**2)
    print(f'loss: {loss}', f'xpos1: {xpos1}', f'xpos2: {xpos2}')
    return loss, lams_fin

