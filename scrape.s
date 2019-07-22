#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=BDAD_Project
#SBATCH --mail-type=END
##SBATCH --mail-user=jl860@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3

cd /scratch/jl860/bdad/fp
source env/bin/activate
python scraper_paris.py




