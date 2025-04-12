#!/bin/zsh
py run_experiment.py --configfile="exp_configs/basic_movements_right.yaml" \
  --plot_every=1 --render_every=1 --seed=8 --start-it=0 \
  --name="basic_movements_right" --savedir="/storage/naturalistic_motor_learning_task/output"
