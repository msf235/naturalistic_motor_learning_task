#!/bin/zsh
commit_id=$(git rev-parse --short HEAD)
echo $commit_id
py run_experiment.py --configfile="exp_configs/tennis_serve_phase_1.yaml" \
  --plot_every=1 --render_every=1 --seed=8 --task-phase=1 --rerun \
  --name="$commit_id" --savedir="/storage/naturalistic_motor_learning_task/output"
