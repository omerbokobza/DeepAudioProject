#!/bin/bash


BLOCK_SIZE=8
MODEL_LENGTH=768
checkpoint_dir="/home/tzlillev/LLadaSMDM/checkpoints/musicnet_32khz/TEST_DATALOADER"
mkdir -p "$checkpoint_dir/checkpoints"
python -u bd3lms/main.py \
     loader.global_batch_size=10 \
     loader.eval_global_batch_size=1 \
     loader.batch_size=10 \
     loader.eval_batch_size=1 \
     model=small \
     algo=bd3lm \
     algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
     data=musicnet_32khz \
     model.length=$MODEL_LENGTH \
     block_size=$BLOCK_SIZE \
     wandb.name=TEST_DATALOADER \
     mode=train \
     model.attn_backend=sdpa \
     training.resample=True \
     checkpointing.save_dir=${checkpoint_dir} \
     checkpointing.resume_from_ckpt=False \
     callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
     callbacks.checkpoint_every_n_steps.save_last=true \
     trainer.log_every_n_steps=10 \
     trainer.val_check_interval=0 \
     trainer.limit_val_batches=0 \
     data.cache_dir=/home/tzlillev/LLadaSMDM/ramdisk \
     music=True \
     data.insert_train_special=False \
     lr_scheduler.t_initial=200000 \
     lr_scheduler.warmup_t=1000 \
     lr_scheduler.warmup_lr_init=1e-6 \
     optim.lr=1e-3 \
     lr_scheduler=cosine_decay_warmup \
     +data.insert_train_eos=True \
     data.insert_train_special=True