#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################


##SBATCH --partition=l40s
#SBATCH --partition=main
##SBATCH --part=rtx6000						### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
##SBATCH --part rtx6000
##SBATCH --qos=eliyanac
#SBATCH --constraint=rtx_6000      		###gtx_1080 | rtx_2080 | rtx_3090 | rtx_4090 | rtx_6000 | titan_rtx | tesla_p100
##SBATCH --gpus=0
##SBATCH --exclude=cs-6000-04,ise-cpu256-15
##SBATCH --nodelist=ise-cpu256-01 
##SBATCH --time 0-4:00:00					### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name MTG	    			### name of the job
#SBATCH --output MTG.out				### output log for running job - %J for job number
##SBATCH --mem=48G							### ammount of RAM memory, allocating more than 60G requires IT team's permission
##SBATCH --nodes=1
##SBATCH --cpus-per-gpu=32
##SBATCH --gpus-per-node=16
##SBATCH --tmp=100G
#SBATCH --mail-user=tzlillev@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL						### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --ntasks-per-node=1
##SBATCH --nodelist=ee-l40s-01
#SBATCH --mem=512G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
##SBATCH --cpus=32
#SBATCH --cpus-per-task=32


module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate BD				### activate a conda environment, replace my_env with your conda environment
cd /home/tzlillev/LLadaSMDM 


python -u datasets_music/musicnet.py 