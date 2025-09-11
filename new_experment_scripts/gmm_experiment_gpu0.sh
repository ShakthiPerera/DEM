#!/bin/bash

# Dimensions for GPU 0
DIMS="8 16 32"

# Base overrides for 8 Gaussians on [-8,8] scale, normalized by 10
BASE_OVERRIDES="energy.n_mixes=8 energy.loc_scaling=8 energy.data_normalization_factor=11"

# Fixed overrides for faster experiments, adapted for 8 modes
FIXED_OVERRIDES="model.partial_prior.dim=\${energy.dimensionality} model.nll_integration_method=euler model.tol=1e-3 model.nll_batch_size=256 model.use_ema=true model.init_from_prior=true model.use_buffer=true model.num_samples_to_save=10000 model.buffer.prioritize=false "

# Loop over dimensions
for DIM in $DIMS; do
  # Dim-dependent overrides
  if [ $DIM -le 8 ]; then
    DIM_OVERRIDES="model.num_estimator_mc_samples=500 model.num_samples_to_generate_per_epoch=1000 model.num_integration_steps=200 model.eval_batch_size=1000"
  elif [ $DIM -le 32 ]; then
    DIM_OVERRIDES="model.num_estimator_mc_samples=700 model.num_samples_to_generate_per_epoch=1000 model.num_integration_steps=200 model.eval_batch_size=1000"
  elif [ $DIM -le 64 ]; then
    DIM_OVERRIDES="model.num_estimator_mc_samples=1000 model.num_samples_to_generate_per_epoch=512 model.num_integration_steps=200 model.eval_batch_size=512"
  elif [ $DIM -le 128 ]; then
    DIM_OVERRIDES="model.num_estimator_mc_samples=1000 model.num_samples_to_generate_per_epoch=512 model.num_integration_steps=300 model.eval_batch_size=512"
  else
    DIM_OVERRIDES="trainer.max_epochs=2000 data.n_train_batches_per_epoch=200 model.num_estimator_mc_samples=1000 model.num_samples_to_generate_per_epoch=512 model.num_integration_steps=300 model.eval_batch_size=512"
  fi

  # Combine all overrides
  ALL_OVERRIDES="${BASE_OVERRIDES} energy.dimensionality=${DIM} ${FIXED_OVERRIDES} ${DIM_OVERRIDES}"

  # Set CUDA_VISIBLE_DEVICES for GPU 0
  export CUDA_VISIBLE_DEVICES=0

  # Output base dir for this dim
  BASE_DIR="outputs_1/gmm_8modes_dim${DIM}"

  # Train: Override params and set Hydra dir
  TRAIN_DIR="${BASE_DIR}/train"
  python dem/train.py experiment=gmm_idem ${ALL_OVERRIDES} hydra.run.dir=${TRAIN_DIR}

  echo "Completed dim=${DIM} on GPU 0. Samples saved to ${BASE_DIR}. Check ${BASE_DIR} for logs/metrics."
done