if [ "$(uname)" != "Darwin" ]; then
    savedir="./output"
else;
    savedir="/storage/naturalistic_motor_learning_task/output"
fi
commit_id=$(git rev-parse --short HEAD)
py run_experiment.py --configfile="exp_configs/tennis_serve.yaml" \
  --plot_every=1 --render_every=1 --seed=8 --phase=1 --rerun \
  --name="phase_1" --savedir="$savedir"
