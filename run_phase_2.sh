#!/bin/zsh
# commit_id=$(git rev-parse --short HEAD)
# echo $commit_id
py run_experiment.py --configfile="exp_configs/tennis_serve.yaml" \
  --plot_every=1 --render_every=1 --seed=8 --phase=2 \
  --name=phase_2_sep --savedir="/storage/naturalistic_motor_learning_task/output"
