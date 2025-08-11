#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################


##SBATCH --partition=l40s
##SBATCH --part=rtx6000						### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
##SBATCH --part rtx6000
##SBATCH --qos=eliyanac
##SBATCH --constraint=rtx_6000      		###gtx_1080 | rtx_2080 | rtx_3090 | rtx_4090 | rtx_6000 | titan_rtx | tesla_p100
##SBATCH --gpus=rtx_6000:1
##SBATCH --exclude=cs-6000-04,ise-cpu256-15
##SBATCH --nodelist=ise-cpu256-01 
#SBATCH --time 7-0:00:00					### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name MTG_GOLDEN  	    			### name of the job
#SBATCH --output MTG_GOLDEN.out				### output log for running job - %J for job number
##SBATCH --mem=48G							### ammount of RAM memory, allocating more than 60G requires IT team's permission
##SBATCH --nodes=1
##SBATCH --cpus-per-gpu=4
##SBATCH --gpus-per-node=3
##SBATCH --tmp=100G
#SBATCH --mail-user=tzlillev@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL						### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --ntasks-per-node=1
##SBATCH --nodelist=ee-l40s-01
##SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_6000:1
##SBATCH --cpus-per-task=16
# ---------------------------------------------------------------------------------------------
#  Environment setup
# ---------------------------------------------------------------------------------------------

module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate BD				### activate a conda environment, replace my_env with your conda environment
cd /home/tzlillev/ProjectAudioMTG       ### change to project root

# ---------------------------------------------------------------------------------------------
#  Hyper-parameters
# ---------------------------------------------------------------------------------------------
BLOCK_SIZE=8                   # BD3LM block size (keep small – affects mask logic)
MODEL_LENGTH=2048               # Context window trained on (~2.66 s @ 3072 tok/s)

# Directory for checkpoints & Hydra configs
checkpoint_dir="/home/tzlillev/LLadaSMDM/checkpoints/musicnet_32khz/MTG_GOLDEN"
mkdir -p "$checkpoint_dir/checkpoints"

echo "Checkpoints: $checkpoint_dir"

# ---------------------------------------------------------------------------------------------
#  Launch training
# ---------------------------------------------------------------------------------------------
python -u bd3lms/main.py \
    loader.global_batch_size=12 \
    loader.eval_global_batch_size=1 \
    loader.batch_size=12 \
    loader.eval_batch_size=1 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=mtg \
    model.length=$MODEL_LENGTH \
    block_size=$BLOCK_SIZE \
    wandb.name=MTG_GOLDEN \
    mode=train \
    model.attn_backend=sdpa \
    training.resample=True \
    checkpointing.save_dir=${checkpoint_dir} \
    checkpointing.resume_ckpt_path=${checkpoint_dir}/checkpoints/last-v1.ckpt \
    checkpointing.resume_from_ckpt=False \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=500 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    trainer.log_every_n_steps=5 \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    trainer.val_check_interval=0 \
    trainer.limit_val_batches=0 \
    data.cache_dir=/home/tzlillev/ProjectAudioMTG/ramdisk \
    music=True \
    trainer.max_steps=100000000 \
    lr_scheduler.t_initial=1500000 \
    lr_scheduler.warmup_t=2000 \
    lr_scheduler.warmup_lr_init=1e-6 \
    lr_scheduler.lr_min=1e-5 \
    optim.lr=1e-3 \
    lr_scheduler=cosine_decay_warmup
    #wandb.name=bd3lm-musicnet_32khz-len${MODEL_LENGTH}-block${BLOCK_SIZE}-b16h16-e13-cosine-2048-gold-1gpu-16bh-1e-3
    #wandb.name=bd3lm-musicnet_32khz-new-len${MODEL_LENGTH}-block${BLOCK_SIZE}-b16h16-e13-cosine-2048-golden-1gpu-TESTETST_16bh_1e-3\




# BLOCK_SIZE=8                   # BD3LM block size (keep small – affects mask logic)
# MODEL_LENGTH=2048               # Context window trained on (~2.66 s @ 3072 tok/s)

# python -u bd3lms/main.py \
#     loader.eval_batch_size=1 \
#     model=small \
#     algo=bd3lm \
#     algo.T=5000 \
#     data=musicnet_mtg_32khz \
#     model.length=1024 \
#     block_size=8 \
#     wandb=null \
#     mode=sample_eval \
#     eval.checkpoint_path=/home/tzlillev/LLadaSMDM/checkpoints/musicnet_32khz/MTG_GOLDEN/checkpoints/last-v1.ckpt \
#     model.attn_backend=sdpa \
#     sampling.nucleus_p=0.9 \
#     sampling.kv_cache=true \
#     music=True \
#     sampling.var_length=True