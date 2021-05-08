#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=4G
#SBATCH --output=res.txt
#SBATCH --time=1:00
#SBATCH --ntasks=1
#SBATCH --job-name=romelhen-test

cd /h/romelhen
python basic_pytorch_script.py