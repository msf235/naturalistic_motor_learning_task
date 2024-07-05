# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import yaml
import glob
import numpy as np
import random
import argparse

# from isaacgym import gymapi
# from isaacgym import gymutil

import torch


SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):

    cfg_path = f'embodied_pose/cfg/**/{args.cfg}.yaml'
    files = glob.glob(cfg_path, recursive=True)
    assert(len(files) == 1)
    cfg_file = files[0]

    with open(cfg_file, 'r') as f:
        cfg_all = yaml.load(f, Loader=yaml.SafeLoader)

    cfg = {'name': cfg_all['name'], 'test_name': cfg_all.get('test_name', 'HumanoidSMPLRefv1'), 'env': cfg_all['env'], 'sim': cfg_all['sim']}
    cfg_train = {'params': cfg_all['params']}

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    if args.task is not None:
        cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    cfg_dir = os.path.join('/tmp/embodied_pose' if args.tmp else args.results_dir, args.cfg)
    cfg['cfg_dir'] = cfg_dir

    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["config"]["load_checkpoint"] = True

    cfg_train["params"]["load_path"] = args.checkpoint
        
    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = cfg_train["params"]["config"]["args"] = args

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['network_path'] = network_path = os.path.join(cfg_dir, 'models') 
    cfg_train['params']['config']['log_path'] = log_path = os.path.join(cfg_dir, 'logs') 
    cfg_train['params']['config']['wandb_dir'] = cfg_dir
    cfg_train['params']['config']['train_dir'] = os.path.join(cfg_dir, 'train') 
    cfg_train['params']['config']['device'] = args.rl_device

    os.makedirs(network_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    if args.test:
        cfg['name'] = cfg['test_name']
    if args.motion_id is not None:
        cfg['env']['motion_id'] = args.motion_id
    if args.export_dataset is not None:
        cfg['env']['export_dataset'] = args.export_dataset
    
    cfg['env']['record'] = args.record
    if args.rec_fname is not None:
        cfg['env']['rec_fname'] = args.rec_fname
    if args.num_rec_frames is not None:
        cfg['env']['num_rec_frames'] = args.num_rec_frames
    if args.rec_fps is not None:
        cfg['env']['rec_fps'] = float(args.rec_fps)
    if args.camera is not None:
        cfg['env']['camera'] = args.camera

    if args.cpu_motion_lib:
        cfg['env']['gpu_motion_lib'] = not args.cpu_motion_lib

    if args.test_motion_file is not None:
        cfg['env']['test_motion_file'] = args.test_motion_file

    if args.motion_file_range is not None:
        new_range = [int(x) for x in args.motion_file_range.split('-')]
        if len(new_range) == 1:
            new_range.append(new_range[0] + 1)
        cfg['env']['motion_file_range'] = new_range

    if args.context_length is not None:
        cfg['env']['context_length'] = args.context_length

    return cfg, cfg_train


def get_arg_parser():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--rerun", action="store_true", default=False,
            # help="Retrain model instead of loading previous file")
    # parser.add_argument("--savefile", type=str, default=None,
            # help="Path to the saved data.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", type=str, default=None,
            help="Path to the configuration file (YAML formatted).")
    parser.add_argument("--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--rerun", action="store_true",
                        help="Retrain model instead of loading.")
    parser.add_argument("--savefile", type=str, default=None,
            help="Path to the datafile for previous runs.")
    parser.add_argument("--seed", type=int, default=2,
            help="Random seed.")
    parser.add_argument("--render", action="store_true",
            help="Render the agent.")
    return parser

def get_config(config_file):
    """Get configuration from yaml configuration file."""

    with open(config_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

