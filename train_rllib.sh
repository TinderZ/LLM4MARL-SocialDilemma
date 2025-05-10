#!/bin/bash

# Set experiment name
EXP_NAME="cleanup_PPObaseline_5-10"

# Set environment and model configurations
ENV="cleanup"
MODEL="baseline"
ALGORITHM="PPO"
NUM_AGENTS=5
STOP_TIMESTEPS=500000000
POLICY_MODE="centralized"  # choices=["centralized", "decentralized", "two_policies"]
# Set hyperparameters

NUM_WORKERS=1  # Optimized for multi-GPU parallelism
NUM_ENVS_PER_WORKER=16  # Increased parallelism for environment instances
ROLL_OUT_FRAGMENT_LENGTH=1000
TRAIN_BATCH_SIZE=$((NUM_WORKERS * NUM_ENVS_PER_WORKER * ROLL_OUT_FRAGMENT_LENGTH))
SGD_MINIBATCH_SIZE=2000
NUM_SGD_ITER=8          # ppo epochs
CHECKPOINT_FREQ=50      # save per N iter

ENTROPY_COEFF=0.00176
LR_SCHEDULE_STEPS="0 20000000"
LR_SCHEDULE_WEIGHTS="0.00126 0.000012"
GRAD_CLIP=40.0

#PPO Epochs 或 Optimization Epochs

# Set GPU configuration
CPUS_PER_WORKER=8  # Adjusted based on CPU core availability
GPUS_PER_WORKER=0.8  # Each worker uses one GPU
CPUS_FOR_DRIVER=12  # Number of CPU cores available for driver
GPUS_FOR_DRIVER=0.2  # Use one GPU for driver (since GPUs are powerful)

HORIZON=50 # short episode length, and use soft-horizon

# Set up Ray configuration
export RAY_MEMORY=90000000000  # 160000000000 大约是给了 Ray 149 GB 的共享内存 # Example memory allocation for Ray workers
export RAY_ALLOW_MULTI_GPU=1  # Enable multi-GPU mode for Ray

# Run training with optimizations for multi-GPU setup
python train_rllib.py \
  --exp_name $EXP_NAME \
  --env $ENV \
  --model $MODEL \
  --algorithm $ALGORITHM \
  --policy_mode $POLICY_MODE \
  --num_agents $NUM_AGENTS \
  --num_workers $NUM_WORKERS \
  --num_envs_per_worker $NUM_ENVS_PER_WORKER \
  --rollout_fragment_length $ROLL_OUT_FRAGMENT_LENGTH \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --sgd_minibatch_size $SGD_MINIBATCH_SIZE \
  --num_sgd_iter $NUM_SGD_ITER \
  --checkpoint_freq $CHECKPOINT_FREQ \
  --stop_timesteps $STOP_TIMESTEPS \
  --cpus_per_worker $CPUS_PER_WORKER \
  --gpus_per_worker $GPUS_PER_WORKER \
  --cpus_for_driver $CPUS_FOR_DRIVER \
  --gpus_for_driver $GPUS_FOR_DRIVER \
  --entropy_coeff $ENTROPY_COEFF \
  --lr_schedule_steps $LR_SCHEDULE_STEPS \
  --lr_schedule_weights $LR_SCHEDULE_WEIGHTS \
  --grad_clip $GRAD_CLIP \
  --clip_param 0.2 \
  --vf_loss_coeff 0.5 \
  --lstm_hidden_size 128