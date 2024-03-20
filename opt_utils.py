import mujoco as mj
import time
import sys
import numpy as np
import scipy
import control_logic as cl
import sim_utils as util
import humanoid2d as h2d
import copy

### LQR
def get_ctrl0(model, data, qpos0):
    data = copy.deepcopy(data)
    data.qpos[:] = qpos0.copy()
    mj.mj_forward(model, data)
    data.qacc[:] = 0
    data.qvel[:] = 0
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
    ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
    return ctrl0

def get_joint_names(model, data=None):
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
    joints['right_arm_joint_inds'] = [8, 9]
    joints['non_right_arm_joint_inds'] = [i for i in range(model.nq) if i not
                                          in joints['right_arm_joint_inds']]
    joints['right_arm_act_inds'] = [5,6]
    joints['non_right_arm_act_inds'] = [i for i in range(model.nu) if i not in
                                        joints['right_arm_act_inds']]
    return joints


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

def get_feedback_ctrl_matrix_from_QR(model, data, Q, R):
    # Assumes that data.ctrl has been set to ctrl0 and data.qpos has been set
    # to qpos0.
    qvel = data.qvel.copy()
    data.qvel[:] = 0
    nq = model.nq
    A = np.zeros((2*nq, 2*nq))
    B = np.zeros((2*nq, model.nu))
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B,
                        None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    data.qvel[:] = qvel
    return K

def get_feedback_ctrl_matrix(model, data, ctrl0, excluded_state_inds=[], rv=None):
    # What about data.qpos, data.qvel, data.qacc?
    data = copy.deepcopy(data)
    data.ctrl[:] = ctrl0
    nq = model.nq
    nu = model.nu
    if rv is None:
        R = np.eye(nu)
    else:
        R = np.diag(rv)
    Q = get_Q_matrix(model, data, excluded_state_inds)
    K = get_feedback_ctrl_matrix_from_QR(model, data, Q, R)
    return K

def get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0):
    dq = np.zeros(model.nq)
    mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    # ctrl0 = get_ctrl0(model, data)
    return ctrl0 - K @ dx

def get_stabilized_ctrls(model, data, Tk, noisev,
                        qpos0, K_update_interv=None, free_act_ids=None,
                         free_ctrls=None):
    if free_act_ids is None or len(free_act_ids) == 0:
        assert free_ctrls is None
        free_ctrls = np.empty((Tk, 0))
        free_act_ids = []
        free_jnt_ids = []
    else:
        free_jnt_ids = [model.actuator(k).trnid[0] for k in free_act_ids]
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
            qpos0n[free_jnt_ids] = data.qpos[free_jnt_ids]
            ctrl0 = get_ctrl0(model, data, qpos0n)
            K = get_feedback_ctrl_matrix(model, data, ctrl0)
        ctrl = get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
        ctrls[k] = ctrl
        ctrl[free_act_ids] = free_ctrls[k]
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrl + noisev[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
    return ctrls, K, qs, qvels

### Gradient descent
def traj_deriv(model, data, qs, vs, us, lams_fin, losses,
               fixed_act_inds=[], k0=None):
    data = copy.deepcopy(data)
    nufree = model.nu - len(fixed_act_inds)
    Tk = qs.shape[0]
    As = np.zeros((Tk-1, 2*model.nv, 2*model.nv))
    Bs = np.zeros((Tk-1, 2*model.nv, nufree))
    B = np.zeros((2*model.nv, model.nu))
    # Cs = np.zeros((Tk, 3, model.nv))
    lams = np.zeros((Tk, 2*model.nv))
    not_fixed_act_inds = [i for i in range(model.nu) if i not in
                          fixed_act_inds]

    for tk in range(Tk-1):
        data.qpos[:] = qs[tk]
        data.qvel[:] = vs[tk]
        data.ctrl[:] = us[tk]
        epsilon = 1e-6
        mj.mjd_transitionFD(model, data, epsilon, True, As[tk], B, None, None)
        Bs[tk] = np.delete(B, fixed_act_inds, axis=1)
        # mj.mj_jacSite(model, data, Cs[tk], None, site=model.site('').id)

    lams[Tk-1, :model.nv] = lams_fin
    grads = np.zeros((Tk, nufree))
    # tau_loss_factor = 1e-9
    tau_loss_factor = 0
    loss_u = np.delete(us, fixed_act_inds, axis=1)

    for tk in range(2, Tk):
        lams[Tk-tk] = As[Tk-tk].T @ lams[Tk-tk+1]
        # grads[Tk-tk] = (tau_loss_factor/tk**.5)*loss_u[Tk-tk] \
        grads[Tk-tk] = tau_loss_factor*loss_u[Tk-tk] \
                + Bs[Tk-tk].T @ lams[Tk-tk+1]
    # breakpoint()
    return grads

