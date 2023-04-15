#!/usr/bin/env bash

cd ~/git/transpormer_
conda activate tsp

python3 main.py \
--model custom \
--positional_encoding sin \
--train_mode reinforce \
--loss reinforce_loss_entropy \
--reinforce_loss_entropy_alpha 0.6 \
--do_train \
--train_steps_per_epoch 2500 \
--epochs 100 \
--reinforce_baseline baseline \
--do_eval \
--eval_dataset tsp_data/validation \
--train_batch_size 512 \
--eval_batch_size 32 \
--dataloader_num_workers 2 \
--checkpoint_dir checkpoints \
--save_epochs 1 \
--learning_rate 1e-4 \
--num_hidden_encoder_layers 5 \
--d_model 128 \
--dim_feedforward 512 \
--nhead 8 \
--activation relu \
--norm custom_batch \
--device mps \
--sinkhorn_tau 0.02 \
--use_feedforward_block_ca \
--optimizer adam \
--metrics len_to_ref_len_ratio avg_tour_len avg_tour_len_ils \
--metric_for_best_checkpoint avg_tour_len \
--ils_n_restarts 5 \
--ils_n_iterations 10 \
--ils_k 5 \
--tb_comment custom
# --resume_from_checkpoint path_to_checkpoint
