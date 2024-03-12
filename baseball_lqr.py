import mujoco as mj
import time
import sys
import numpy as np
import scipy
import control_logic as cl
import humanoid2d as h2d

# xml_file = 'humanoid_and_baseball.xml'
# with open(xml_file, 'r') as f:
  # xml = f.read()

def get_ctrl0(model, data):
    # Get initial stabilizing controls
    # THIS FUNCTION MODIFIES data!
    mj.mj_forward(model, data) # Necessary?
    data.qacc = 0
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

def get_Q_joint(model, data=None):
    balance_joint_cost  = 3     # Joints required for balancing.
    other_joint_cost    = .3    # Other joints.
    joints = get_joint_names(model)
    # Construct the Qjoint matrix.
    Qjoint = np.eye(model.nq)
    Qjoint[joints['root_dofs'], joints['root_dofs']] *= 0  # Don't penalize free joint directly.
    Qjoint[joints['balance_dofs'], joints['balance_dofs']] *= balance_joint_cost
    Qjoint[joints['other_dofs'], joints['other_dofs']] *= other_joint_cost
    return Qjoint

def get_Q_matrix(model, data):
    # Cost coefficients.
    balance_cost        = 1000  # Balancing.

    Qbalance = get_Q_balance(model, data)
    Qjoint = get_Q_joint(model, data)
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
    return K

def get_feedback_ctrl_matrix(model, data):
    # Assumes that data.ctrl has been set to ctrl0.
    nq = model.nq
    nu = model.nu
    R = np.eye(nu)
    Q = get_Q_matrix(model, data)
    K = get_feedback_ctrl_matrix_from_QR(model, data, Q, R)
    return K

def get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0):
    dq = np.zeros(model.nq)
    mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    # ctrl0 = get_ctrl0(model, data)
    return ctrl0 - K @ dx

# def get_lqr_ctrl(model, data, qpos0, ctrl0):
    # K = get_feedback_ctrl_matrix(model, data)
    # return get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)



