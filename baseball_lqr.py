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

class BaseballLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Get all joint names.
        self.joint_names = [model.joint(i).name for i in range(model.njnt)]

        # Get indices into relevant sets of joints.
        self.root_dofs = range(3)
        self.body_dofs = range(3, model.nq)
        self.abdomen_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if 'abdomen' in name
            and not 'z' in name
        ]
        self.leg_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if ('hip' in name or 'knee' in name or 'ankle' in name)
            and not 'z' in name
        ]
        self.balance_dofs = abdomen_dofs + leg_dofs
        self.other_dofs = np.setdiff1d(body_dofs, balance_dofs)

        # Cost coefficients.
        self.balance_cost        = 1000  # Balancing.
        self.balance_joint_cost  = 3     # Joints required for balancing.
        self.other_joint_cost    = .3    # Other joints.
        # Need to update this with burn-in.
        self.qpos0 = data.qpos.copy()  # Save the position setpoint.

            
    def set_model_data(self, model, data):
        self.model = model
        self.data = data
        self.qpos0 = data.qpos.copy()  # Save the position setpoint.

    def get_ctrl0(self):
        model = self.model
        data = self.data
        # Burn in: # Todo: move this out
        for k in range(10):
            # env.step(np.zeros(nu))
            mj.mj_step(model, data)

        # Get initial stabilizing controls
        data.qacc = 0
        mj.mj_inverse(model, data)
        qfrc0 = data.qfrc_inverse.copy()
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
        ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
        # mj.mj_resetData(model, data) # Maybe need to do this differently.
        return ctrl0

    def get_Q_balance(self):
        model = self.model
        data = self.data
        nq = self.model.nq
        jac_com = np.zeros((3, self.nq))
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

    def get_Q_joint(self):
        # Construct the Qjoint matrix.
        Qjoint = np.eye(self.model.nq)
        Qjoint[self.root_dofs, self.root_dofs] *= 0  # Don't penalize free joint directly.
        Qjoint[self.balance_dofs, self.balance_dofs] *= self.balance_joint_cost
        Qjoint[self.other_dofs, self.other_dofs] *= self.other_joint_cost
        return Qjoint


    def get_Q_matrices(self):
        # Construct the Q matrix for position DoFs.
        Qpos = self.balance_cost * self.Qbalance + self.Qjoint

        # No explicit penalty for velocities.
        nq = self.model.nq
        Q = np.block([[self.Qpos, np.zeros((nq, nq))],
                      [np.zeros((nq, 2*nq))]])
        return Q

    def get_feedback_ctrl_matrix_from_QR(self, Q, R):
        nq = self.model.nq
        nu = self.model.nu
        A = np.zeros((2*nq, 2*nq))
        B = np.zeros((2*nq, nu))
        epsilon = 1e-6
        flg_centered = True
        mj.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B,
                            None, None)

        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute the feedback gain matrix K.
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def get_feedback_ctrl_matrix(self):
        nq = self.model.nq
        nu = self.model.nu
        R = np.eye(nu)
        Q = self.get_Q_matrices()
        K = self.get_feedback_ctrl_matrix_from_QR(Q, R)
        return K

    def get_lqr_ctrl_from_K(self, K):
        dq = np.zeros(nq)
        mj.mj_differentiatePos(self.model, dq, 1, qpos0, self.data.qpos)
        dx = np.hstack((dq, self.data.qvel)).T
        ctrl0 = self.get_ctrl0()
        return ctrl0 - K @ dx

    def get_lqr_ctrl(self):
        K = self.get_feedback_ctrl_matrix()
        return self.get_lqr_ctrl_from_K(K)


# fact = 3
# Qupright = np.eye(nq, nq)
# # Qupright = np.zeros((nq, nq))
# Qupright[0,0] = fact
# Qupright[1,1] = fact

# Q = fact*np.block([[Qupright, np.zeros((nq, nq))],
              # [np.zeros((nq, nq)), np.eye(nq)]])






