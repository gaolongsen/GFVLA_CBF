cd /workspaces/chenhao/code/Fast-in-Slow
export COPPELIASIM_ROOT=/workspaces/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine

N=0
Xvfb :$N -screen 0 1024x768x24 &  
export DISPLAY=:$N

export PYTHONPATH=/workspaces/chenhao/code/Fast-in-Slow:/workspaces/chenhao/code/Fast-in-Slow/transformers:/workspaces/chenhao/code/Fast-in-Slow/timm:$PYTHONPATH

models=("<fisvla_model_path>")
# tasks=("close_box" "close_laptop_lid")
# tasks=("toilet_seat_down" "sweep_to_dustpan")
# tasks=("close_fridge" "place_wine_at_rack_location")
# tasks=("water_plants" "phone_on_base")
# tasks=("take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

tasks=("close_box")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-2)}')
  action_steps=$(echo ${exp_name} | grep -oP 'window\K[0-9]+')
  for task in "${tasks[@]}"; do
    python scripts/sim.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name}_diff_layer-1_step4 \
      --replay-or-predict 'predict' \
      --result-dir /workspaces/chenhao/test_result/ch_test_0525 \
      --training_mode 'async' \
      --slow-fast-ratio 4 \
      --cuda $N \
      --training-diffusion-steps 100 \
      --llm_middle_layer 30 \
      --use-diff 1 \
      --use-ar 0 \
      --use_robot_state 1 \
      --model-action-steps 0 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 1 \
      --pointcloud-pos "fast" \
      --action-chunk 1 \
      --sparse 1 \
      --angle_delta 0 \
      --lang_subgoals_exist 1 \
      --ddim-steps 4   # 4, 6, 8
  done
done