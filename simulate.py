import model_output_manager_hash as mom
from PIL import Image
import signal
import copy
import cv2
import numpy as np
import numpy_ml as npml
import scipy.linalg
import mujoco as mj
# import mediapy as media
import matplotlib.pyplot as plt
import numpy_ml.neural_nets.optimizers as optimizers
import numpy_ml.neural_nets.schedulers as schedulers
import inspect 
import sys 

np.set_printoptions(precision=10, suppress=True, linewidth=100)

memory = mom.Memory('cache')

def reset_model(data, data0):
    data.qpos[:] = data0.qpos
    data.qvel[:] = data0.qvel
    data.qacc[:] = data0.qacc
    data.time = data0.time
    data.act[:] = data0.act

def get_initial_stabilize_ctrl(model, data):
    data0 = copy.deepcopy(data)
    qpos0 = data.qpos.copy()
    data.qacc = 0  # Assert that there is no the acceleration.
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = qfrc0 @ np.linalg.pinv(data.actuator_moment)
    return ctrl0

# Set the initial state and control.
def get_stabilize_ctrl(model, data, Tk):
    ctrl0 = get_initial_stabilize_ctrl(model, data)

    data0 = copy.deepcopy(data)
    reset_model(data, data0)
    qpos0 = data0.qpos.copy()

    nv = model.nv
    nu = model.nu

    data.ctrl[:] = ctrl0
    Q = 1000*np.eye(2*nv)
    Q[nv:,nv:] = 0
    R = np.eye(nu)
    
    # Allocate the A and B matrices, compute them.
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    reset_model(data, data0)

    taus = np.zeros((Tk, nu))
    taus[0] = ctrl0
    dq = np.zeros(nv)
    data.qpos[:] = qpos0
    for k in range(1,Tk):
        mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T
        ctrl = ctrl0 - K @ dx
        taus[k] = ctrl
        data.ctrl[:] = ctrl
        mj.mj_step(model, data)
    return taus


def forward_sim(model, data, taus, site1="tip", site2=None, renderer=None,
                frames=None, framerate=60):
    if renderer is None:
        renderer = mj.Renderer(model)
    def render():
        renderer.update_scene(data, camera=model.cam('top').id)
        return renderer.render()
    nv = model.nv
    nu = model.nu
    Tk = taus.shape[0]
    qs = np.zeros((Tk, nv))
    touchs = np.zeros(Tk)
    vs = np.zeros((Tk, nv))
    sites1_pos = np.zeros((Tk, 3))
    sites2_pos = np.zeros((Tk, 3))
    As = np.zeros((Tk, 2*nv, 2*nv)) # f_x
    Bs = np.zeros((Tk, 2*nv, nu)) # f_tau
    Cs = np.zeros((Tk, 3, nv)) # d(site_pos)/dq "end-effector jacobian"

    mj.mj_forward(model, data)
    qs[0] = data.qpos.copy()
    vs[0] = data.qvel.copy()
    touchs[0] = data.sensor('tiptouch').data.copy()
    sites1_pos[0] = data.site("tip").xpos.copy()
    data.ctrl[:] = taus[0]
    epsilon = 1e-6
    mj.mjd_transitionFD(model, data, epsilon, True, As[0],
                        Bs[0], None, None)  
    mj.mj_jacSite(model, data, Cs[0], None, site=model.site(site1).id)
    for tk in range(1, Tk):  # Step the simulation.
        mj.mj_step(model, data)
        qs[tk] = data.qpos.copy()
        vs[tk] = data.qvel.copy()
        touchs[tk] = data.sensor('tiptouch').data.copy()
        sites1_pos[tk] = data.site(site1).xpos.copy()
        if site2 is not None:
            sites2_pos[tk] = data.site(site2).xpos.copy()
        data.ctrl[:] = taus[tk]
        mj.mjd_transitionFD(model, data, epsilon, True, As[tk],
                            Bs[tk], None, None)  
        mj.mj_jacSite(model, data, Cs[tk], None, site=model.site(site1).id)
        if frames is not None and len(frames) < data.time * framerate:
            frames.append(render())
    outdata = dict(qs=qs, vs=vs, tiptouch=touchs, sites1=sites1_pos,
                   sites2=sites2_pos, As=As, Bs=Bs, Cs=Cs)
    return outdata

def contact_one_it(k, Tk, m, taus, dlds, dldtheta, opts, As, Bs, Cs,
                   theta_final, targ_factor=1, angle_factor=1,
                   tau_loss_factor=0):
    nv = int(As[0].shape[0]/2)
    nu = Bs[0].shape[1]
    lams = np.zeros((Tk, 2*nv))
    dldq = Cs[Tk-1].T @ dlds
    lams[Tk-1,:nv] = targ_factor * dldq
    lams[Tk-1,2] += dldtheta
    grads = np.zeros((Tk, nu))
    loss_taus = taus / Tk
    for k, tk in enumerate(range(Tk-2, Tk-m-1, -1)):
        lams[tk] = As[tk].T @ lams[tk+1]
        grads[tk] = (tau_loss_factor/(k+1)**.5)*loss_taus[tk] \
                + Bs[tk].T @ lams[tk+1]
        if hasattr(opts[k], "update"):
            taus[tk] = opts[k].update(taus[tk], grads[tk], "tau_k") 
        else:
            taus[tk] = taus[tk] - opts[k] * grads[tk]
    # for k, tk in enumerate(range(2, m+1)):
        # lams[Tk-tk] = As[Tk-tk].T @ lams[Tk-tk+1]
        # grads[Tk-tk] = (tau_loss_factor/tk**.5)*loss_taus[Tk-tk] \
                # + Bs[Tk-tk].T @ lams[Tk-tk+1]
        # if hasattr(opts[k], "update"):
            # taus[Tk-tk] = opts[k].update(taus[Tk-tk], grads[Tk-tk], "tau_k") 
        # else:
            # taus[Tk-tk] = taus[Tk-tk] - opts[k] * grads[Tk-tk]

def contact_target(model, data, K, m, taus, opt, theta_final, sites2,
                targ_factor=1, angle_factor=0, tau_loss_factor=0):
    Tk = taus.shape[0]
    curr_losses = []
    opts = []
    for k, tk in enumerate(range(2, m+1)):
        opts.append(copy.deepcopy(opt))

    data0 = copy.deepcopy(data)
    outdata = forward_sim(model, data, taus)
    for k in range(K+1):
        sites1 = outdata['sites1']
        qs = outdata['qs']
        dlds = sites1[Tk-1] - sites2[Tk-1]
        curr_loss = .5*np.linalg.norm(dlds)**2
        lever_loss = .5*(qs[Tk-1, 2] - theta_final)**2
        print(k, curr_loss, np.linalg.norm(taus)/Tk, lever_loss)
        curr_losses.append(curr_loss)
       
        if k == K:
            break # Don't do update

        dldtheta = angle_factor*(qs[Tk-1, 2] - theta_final)
        contact_one_it(k, Tk, m, taus, dlds, dldtheta, opts, outdata['As'],
                       outdata['Bs'], outdata['Cs'], theta_final,
                       targ_factor, angle_factor, tau_loss_factor)
        reset_model(data, data0)
        outdata = forward_sim(model, data, taus)
    return curr_losses

def optimize_traj_all(model, data, K, taus, opt, theta_final, q_targ_1,
                      q_targ_2, q_targ_3, tk_targ1, tk_targ2, tk_targ3,
                      targ_factor=1, angle_factor=0, tau_loss_factor=0,
                      m=None):
    nv = model.nv
    nu = model.nu
    Tk = taus.shape[0]
    if m is None:
        m = Tk

    curr_losses = []
    opts = []
    for k, tk in enumerate(range(2, m+1)):
        opts.append(copy.deepcopy(opt))

    data0 = copy.deepcopy(data)
    outdata = forward_sim(model, data, taus)
    for k in range(K+1):
        try:
            sites1 = outdata['sites1']
            qs = outdata['qs']
            As = outdata['As']
            Bs = outdata['Bs']
            Cs = outdata['Cs']
            
            if k == K:
                break # Don't do update

            grads = np.zeros((Tk-1, nu))
            lam_grads = np.zeros((Tk-1, nu))
            tau_grads = np.zeros((Tk-1, nu))
            lams = np.zeros((Tk, 2*nv))
            loss_taus = taus / Tk

            dldtheta = angle_factor*(qs[tk_targ1, 2] - theta_final)
            dlds = sites1[tk_targ1] - q_targ_1
            dldq = Cs[tk_targ1-1].T @ dlds
            lams[tk_targ1,:nv] += targ_factor * dldq
            lams[tk_targ1, 2] += dldtheta

            curr_loss = .5*np.linalg.norm(dlds)**2

            dlds = sites1[tk_targ2] - q_targ_2
            dldq = Cs[tk_targ2-1].T @ dlds
            lams[tk_targ2,:nv] += targ_factor * dldq

            curr_loss += .5*np.linalg.norm(dlds)**2

            dldtheta = angle_factor*(qs[tk_targ3, 2] - theta_final)
            dlds = sites1[tk_targ3] - q_targ_3
            dldq = Cs[tk_targ3-1].T @ dlds
            lams[tk_targ3,:nv] += targ_factor * dldq
            lams[tk_targ3, 2] += dldtheta

            curr_loss += .5*np.linalg.norm(dlds)**2
            curr_loss = curr_loss / 3

            lever_loss = .5*(qs[tk_targ1, 2] - theta_final)**2
            lever_loss += .5*(qs[tk_targ3, 2] - theta_final)**2
            lever_loss = angle_factor * lever_loss / 2

            print(k, curr_loss, np.linalg.norm(taus)/Tk, lever_loss)
            curr_losses.append(curr_loss)

            for k2, tk in enumerate(range(Tk-2, Tk-m-1, -1)):
                lams[tk] += As[tk].T @ lams[tk+1]
                lam_grad = Bs[tk].T @ lams[tk+1] 
                lam_grads[k2] = lam_grad
                tau_grad = (tau_loss_factor/(k2+1)**.5)*loss_taus[tk]
                tau_grads[k2] = tau_grad
                grads[tk] = tau_grad + lam_grad
                if hasattr(opts[k2], "update"):
                    taus[tk] = opts[k2].update(taus[tk], grads[tk], "tau_k") 
                else:
                    taus[tk] = taus[tk] - opts[k2] * grads[tk]
            reset_model(data, data0)
            outdata = forward_sim(model, data, taus)
        except KeyboardInterrupt:
            reset_model(data, data0)
            frames = []
            outdata = forward_sim(model, data, taus, frames=frames)
            save_video(frames, "debug.mp4")
            reset_model(data, data0)
            input("Press enter to continue.")
    return curr_losses

@memory.cache()
def double_tap(durations=(1,1,1,2), opt='adam', lrs=(1,1,1,1),
               num_its=(2000,2000,2000,2000), tau_loss_factor=1e-9,
               targ_factor=.1, theta_final=.4, angle_factor=0,
               xml_file='arm_and_lever.xml'):

    with open(xml_file, 'r') as f:
      xml = f.read()
    del f

    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    nu = model.nu
    nv = model.nv
    dt = model.opt.timestep
    # mj.mj_resetDataKeyframe(model, data, 0)
    # frames = []
    # od = forward_sim(model, data, Tk, taus1, True, frames)
    # save_video(frames, 'simple_before')

    Tk0 = int(durations[0] / dt)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    data.qacc = 0  # Assert that there is no the acceleration.
    mj.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    ctrl0 = qfrc0 @ np.linalg.pinv(data.actuator_moment)
    taus1 = np.ones((Tk0, nu)) * ctrl0

    target1 = data.site("target1").xpos.copy()
    target2 = data.site("target2").xpos.copy()

    sites2 = np.ones((Tk0, 3)) * target1
    mj.mj_resetDataKeyframe(model, data, 0)
    model.jnt_limited = [1,1,0]
    # opt = optimizers.SGD(lr=lrs[0], momentum=0)
    if opt == 'adam':
        optim = optimizers.Adam(lr=lrs[0])

    curr_losses = contact_target(model, data, num_its[0], Tk0, taus1, optim,
                                 theta_final, sites2, targ_factor,
                                 angle_factor, tau_loss_factor)

    # mj.mj_resetDataKeyframe(model, data, 0)
    # frames = []
    # forward_sim(Tk, taus, True, frames)
    # save_video(frames, 'after_target1')

    mj.mj_resetDataKeyframe(model, data, 0)
    model.jnt_limited = [1,1,1]
    # frames = []
    # forward_sim(Tk0, taus, True, frames)
    # save_video(frames, 'after_target1_restricted_lever1')

    Tk1 = int(durations[1] / dt)
    taus2 = np.ones((Tk1, nu)) * taus1[-1]
    # mj.mj_resetDataKeyframe(model, data, 0)
    # frames = []
    # forward_sim(Tk, taus, True, frames)
    # forward_sim(Tk, taus2, True, frames)
    # save_video(frames, 'after_target1_stabilize')

    sites2 = np.ones((Tk1, 3)) * target2
    mj.mj_resetDataKeyframe(model, data, 0)
    od = forward_sim(model, data, taus1)
    # opt = optimizers.SGD(lr=lrs[1], momentum=0)
    if opt == 'adam':
        optim = optimizers.Adam(lr=lrs[1])
    curr_losses = contact_target(model, data, num_its[1], Tk1, taus2, optim,
                              theta_final, sites2, targ_factor,
                              tau_loss_factor=tau_loss_factor)
    # forward_sim(Tk, taus2)
    # curr_losses, taus2[:] = contact_one(K1, Tk, Tk, taus2, lr/4, phase=3)

    # renderer = mj.Renderer(model)
    # mj.mj_resetDataKeyframe(model, data, 0)
    # frames = []
    # forward_sim(model, data, Tk0, taus1, frames=frames, renderer=renderer)
    # forward_sim(model, data, Tk1, taus2, frames=frames, renderer=renderer)
    # save_video(frames, 'after_target2')

    Tk2 = int(durations[2] / dt)
    mj.mj_resetDataKeyframe(model, data, 0)
    forward_sim(model, data, taus1)
    forward_sim(model, data, taus2)
    taus3 = get_stabilize_ctrl(model, data, Tk2)

    # mj.mj_resetDataKeyframe(model, data, 0)
    # frames = []
    # forward_sim(Tk, taus, True, frames)
    # forward_sim(Tk, taus2, True, frames)
    # forward_sim(Tk, taus3, True, frames)
    # save_video(frames, 'after_target2_stabilize')

    target = data.site("target1").xpos.copy()
    sites2 = np.ones((Tk2, 3)) * target1
    mj.mj_resetDataKeyframe(model, data, 0)
    forward_sim(model, data, taus1)
    forward_sim(model, data, taus2)
    model.jnt_limited = [1,1,0]
    # opt = optimizers.SGD(lr=lrs[2], momentum=0)
    if opt == 'adam':
        optim = optimizers.Adam(lr=lrs[2])
    curr_losses = contact_target(model, data, num_its[2], Tk2, taus3, optim,
                              theta_final, sites2, targ_factor,
                              tau_loss_factor=tau_loss_factor)

    mj.mj_resetDataKeyframe(model, data, 0)
    model.jnt_limited = [1,1,1]
    angle_factor = 1
    taus_all = np.concatenate((taus1, taus2, taus3), axis=0)
    mj.mj_resetDataKeyframe(model, data, 0)
    frames = []

    mj.mj_forward(model, data)
    if opt == 'adam':
        optim = optimizers.Adam(lr=lrs[3])
    t_targ1 = int(1 / dt) - 1
    t_targ2 = int(2 / dt) - 1
    t_targ3 = int(3 / dt) - 1
    optimize_traj_all(model, data, num_its[3], taus_all, optim, theta_final,
                      target1, target2, target1, t_targ1, t_targ2, t_targ3,
                      targ_factor, angle_factor, tau_loss_factor)

    # frames = []
    # mj.mj_resetDataKeyframe(model, data, 0)
    # model.jnt_limited = [1,1,1]
    # renderer = mj.Renderer(model)
    # outdata1 = forward_sim(model, data, taus1, renderer=renderer, frames=frames)
    # outdata2 = forward_sim(model, data, taus2, renderer=renderer, frames=frames)
    # outdata3 = forward_sim(model, data, taus3, renderer=renderer, frames=frames)
    # save_video(frames, 'test', 60)

    return taus_all

if __name__ == '__main__':
    xml_file = 'arm_and_lever.xml'
    model = mj.MjModel.from_xml_path(xml_file)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)

    taus_all = double_tap(num_its=(1000,1000,1000,2000), lrs=(1, 1, 1, 1),
                     tau_loss_factor=3e-6, angle_factor=1)

    frames = []
    forward_sim(model, data, taus_all, frames=frames)
    save_video(frames, "debug")
    breakpoint()


    out = double_tap(num_its=(1000,1000,1000))
    taus1, taus2, taus3, outdata1, outdata2, outdata3 = out
    taus = np.concatenate((taus1, taus2, taus3))
    qs = np.concatenate((outdata1['qpos'], outdata2['qpos'], outdata3['qpos']))
    contact = np.concatenate((outdata1['qpos'], outdata2['qpos'], outdata3['qpos']))
    Tk = taus.shape[0]
    # Tk1 = taus1.shape[0]
    xml_file = 'arm_and_lever.xml'
    model = mj.MjModel.from_xml_path(xml_file)
    data = mj.MjData(model)

    mj.mj_resetDataKeyframe(model, data, 0)
    renderer = mj.Renderer(model)
    frames = []
    dataout = forward_sim(model, data, Tk, taus, frames=frames, renderer=renderer)
    save_video(frames, 'test', 60)
    breakpoint()
