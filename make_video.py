import sys
import pickle as pkl
import util

args = sys.argv[1:]

with open(args[0], "rb") as f:
    data = pkl.load(f)

qposs = data["qpos"]
targ = data["trajectory_target"]
site_names = data["site_names"]
breakpoint()
util.make_video_of_motion(args[1], qposs, args[2], targ, site_names, float(args[3]))
