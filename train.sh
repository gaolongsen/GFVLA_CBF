# parameter
export PYTHONPATH=/workspaces/longsen/code/dualarmvla:/workspaces/longsen/code/in/transformers:/workspaces/longsen/code/dualarmvla/timm:$PYTHONPATH
export HF_HOME=/workspaces/huggingface
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=1800

TRAINING_MODE='async'        # a very powerful control mode, see "models/vlms/prismatic.py"
LLM_MIDDLE_LAYER=30
FUTURE_ACTION_STEPS=0
LOAD_POINTCLOUD=true
LOAD_STATE=true
POINTCLOUD_POS='fast'        # use when LOAD_POINTCLOUD is true
DIFFUSION_STEPS=100
ACTION_TOKENIZER_EXIST=true  # if you dont want to use AR loss, set it to false
USE_DIFF=true
AR_DIFF_LOSS=true            # if you dont want to use AR loss, set it to false
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0
FREEZE_VISON=false
FREEZE_LLM=false
SLOW_FAST_RATIO=1_4
ACTION_CHUNK=1
LANG_SUBGOALS_EXIST=true

SETTING="test_multi_key_STATE_${LOAD_STATE}_ACTION_CHUNK_${ACTION_CHUNK}_SLOW_FAST_RATIO_${SLOW_FAST_RATIO}_ddim${DIFFUSION_STEPS}_PC${LOAD_POINTCLOUD}_POS${POINTCLOUD_POS}_${TRAINING_MODE}_withARloss${LOAD_POINTCLOUD}_slow_fast_[after]_[-1]_${LLM_MIDDLE_LAYER}_fisvla_pretrain_window${FUTURE_ACTION_STEPS}"

DATA_MIX=rlbench
TASK=12tasks_selected_keyframe_s1f4_pc_fast_sparsechunk_1_0518
BATCH_SIZE=6
EPOCHS=300
LEARNING_RATE=2e-5
ACTION_DIM=7
CAMERA_VIEW="head_slow,head_fast"

DATA_ROOT=/workspaces/rlds_data
EXP_ROOT=/workspaces/chenhao/code/Fast-in-Slow/exp
MODEL_SAVE_NUM=3

NUM_GPUS=8
NODES=1
MASTER_ADDR="172.31.0.3"
NODE_RANK=0

torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=$NODE_RANK --master_addr=${MASTER_ADDR} --master_port=29500 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --need_to_sub 0 \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project '<wandb_project>' \
  --wandb_entity '<wandb_entity>' \
  --save_interval 100 \
  --action_dim ${ACTION_DIM} \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --is_resume False \
  --llm_middle_layer ${LLM_MIDDLE_LAYER} \
  --camera_view ${CAMERA_VIEW} \
  --training_mode ${TRAINING_MODE} \
  --load_pointcloud ${LOAD_POINTCLOUD} \
  --diffusion_steps ${DIFFUSION_STEPS} \
  --model_save_num ${MODEL_SAVE_NUM} \
  --pointcloud_pos ${POINTCLOUD_POS} \
  --action_chunk ${ACTION_CHUNK} \
  --load_state ${LOAD_STATE} \
  --lang_subgoals_exist ${LANG_SUBGOALS_EXIST} \
  --pretrained_checkpoint "<pretrained_checkpoint>"