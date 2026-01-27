# TODOs

Tasks for implementing a completed signal diffusion paper

- Apply label smoothing to classifiers <https://arxiv.org/abs/1512.00567>
- Rename mapping_cond to something more accurate and descriptive
- Train diff models for each
  - data type
  - conditioning

# Completed

- SOTA diffusion operations
  - stochastic sampling for flow matching?
  - Adjusted initial noise to avoid memorization <https://arxiv.org/pdf/2510.08625>
  - Chamfer guidance for class adherence? <https://arxiv.org/pdf/2508.10631>
  - [x] based on deep research results
  - [x] careful conditioning with text embeddings and cross-attention
  - [x] Rectified CFG++ <https://arxiv.org/pdf/2510.07631>
- Metrics
  - MMD overall against test set and per-class
  - "Improved Precision and Recall" to get at accuracy and diversity
  - [x] Eval for MMD during training
  - [x] save images to tb, wandb
- Utilities
  - Batch size finder based on model config settings
    - Max batch size possible
    - Best batch size based on wall time for 1 epoch using grad_accum to reach a baseline
  - [x] HPO for time-domain classification
  - HPO for diffusion models
    - Parameters: model depth, embed dims, lr, betas, dropout, wgt decay
- Datasets
  - [x] Longitudinal dataset
  - [x] Hyperparameter search with Optuna instead of exhaustive
  - [x] Both time-domain and STFT image domain
  - Compare mag-only STFT to
    - [x] db+iq
    - [x] db+phasor
- Classifiers for:
  - [x] Age
  - [x] Health
  - [x] Gender
  - [x] Versions for time-domain signals and STFT images
- Models for:
  - [x] Mamba diffusion
  - [x] Hourglass diffusion
  - [x] Stable diffusion fine tune
- Update training code
  - [x] bnb adam 8bit optimizer for reduced memory use
  - [WIP] evaluation with MMD and downstream classifier during training
  - [x] logging to wandb/mlflow endpoints
  - [x] checkpoint management and early stopping
  - [x] configuration management
  - [x] bf16 mixed precision
  - [x] gradient checkpointing and accumulation for reduced memory use
  - Metric: KID for average of last four 1000-step checkpoints (per OLMO 3)
  - [x] SWA for classifiers
