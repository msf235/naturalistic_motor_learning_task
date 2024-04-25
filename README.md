Python documentation: https://mujoco.readthedocs.io/en/stable/python.html
XML documentation: https://mujoco.readthedocs.io/en/stable/python.html

File to run: rl_throw_ball.py

Gymnasium environment: humanoid2d.py

xml files:
    humanoid: humanoid.xml
    Scene for baseball throw (without humanoid): baseball_pitch_scene.xml 
    Combine the two above (main file): humanoid_and_baseball.xml
    One note: the humanoid x position is set in rl_throw_ball.py (body_pos)

Target trajectory is computed by grab_ball.throw_traj

Before doing any optimization, I run the simulation with zero control for 10
timesteps. This is done by util.reset and in gymnasium environment's
(Humanoid2dEnv) reset method.

LQR controller is computed opt_utils.get_stabilized_ctrls.

Gradient updates are computed by opt_utils.traj_deriv.

Alternate between getting stabilized controlls (LQR controller) and doing
gradient updates.
