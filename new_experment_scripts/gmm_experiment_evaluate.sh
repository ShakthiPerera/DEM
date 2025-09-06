#!/bin/bash

# List of trained dimensions
DIMS="128"

# Common overrides (shared for all runs)
BASE_OVERRIDES="energy.n_mixes=8 energy.loc_scaling=8 energy.data_normalization_factor=11"
FIXED_OVERRIDES="model.nll_integration_method=euler model.tol=1e-3 model.use_ema=true \
model.init_from_prior=true model.use_buffer=true model.num_samples_to_save=10000 model.buffer.prioritize=false"

# Use GPU 0 for evaluation
export CUDA_VISIBLE_DEVICES=1

for DIM in $DIMS; do
  BASE_DIR="outputs/gmm_8modes_dim${DIM}"
  CKPT_DIR="${BASE_DIR}/train/checkpoints"

  if [ ! -d "$CKPT_DIR" ]; then
    echo "No checkpoints found for dim=${DIM}, skipping."
    continue
  fi

  # Loop through all checkpoint files in the folder
  for CKPT in "$CKPT_DIR"/*.ckpt; do
    CKPT_NAME=$(basename "$CKPT" .ckpt)
    OUT_FILE="${CKPT_DIR}/${CKPT_NAME}_test_metrics.txt"

    echo "Running evaluation for dim=${DIM}, checkpoint=${CKPT_NAME}"

    # Dimension-dependent overrides
    if [ $DIM -le 8 ]; then
      DIM_OVERRIDES="model.num_estimator_mc_samples=500 model.num_samples_to_generate_per_epoch=1000 model.num_integration_steps=200 model.eval_batch_size=1000"
    elif [ $DIM -le 32 ]; then
      DIM_OVERRIDES="model.num_estimator_mc_samples=700 model.num_samples_to_generate_per_epoch=1000 model.num_integration_steps=200 model.eval_batch_size=1000"
    elif [ $DIM -le 64 ]; then
      DIM_OVERRIDES="model.num_estimator_mc_samples=1000 model.num_samples_to_generate_per_epoch=512 model.num_integration_steps=200 model.eval_batch_size=512"
    else
      DIM_OVERRIDES="model.num_estimator_mc_samples=1000 model.num_samples_to_generate_per_epoch=512 model.num_integration_steps=300 model.eval_batch_size=512 model.buffer.max_length=20000"
    fi

    # Run evaluation and save results
    python dem/eval.py experiment=gmm_idem \
      ${BASE_OVERRIDES} energy.dimensionality=${DIM} model.partial_prior.dim=${DIM} \
      ${FIXED_OVERRIDES} ${DIM_OVERRIDES} \
      hydra.run.dir=${BASE_DIR}/eval_${CKPT_NAME} ckpt_path="${CKPT}" \
      | tee "${OUT_FILE}"
  done
done
