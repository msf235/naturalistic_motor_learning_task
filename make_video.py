import sys
import pickle as pkl
import util

args = sys.argv[1:]

with open(args[0], "rb") as f:
    data = pkl.load(f)

qposs = data["qpos"]
util.make_video_of_motion(args[1], qposs, args[2], float(args[3]))
