#!/bin/bash
#
# ================================
# SBATCH CONFIGURATION
# ================================
#SBATCH --job-name=wbx
#SBATCH --partition=fat_genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=192
#SBATCH --time=120:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

# ================================
# Environment setup
# ================================
set -e
set -u

echo "Running on node: $SLURMD_NODENAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

# Activate virtual environment
source env/venv/bin/activate

# Optional but recommended to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ================================
# Run WeatherBenchX evaluation
# ================================

python xaurora_benchmark/run_benchmark_evaluation.py \
    --config=xaurora_configs \
    --prediction=xaurora_small \
    --target=era5 \
    --resolution=1440x721 \
    --time_start=2021-01-01 \
    --time_stop=2021-12-31 \
    --year=2021 \
    --lead_time_start=6 \
    --lead_time_stop=366 \
    --lead_time_frequency=6 \
    --output_dir=/projects/prjs1808/ewalt1/Xaurora/train/16g/2026-01-23_10-18-59/wbx_benchmark/ \
    --lead_time_chunk_size=1 \
    --init_time_chunk_size=8 \
    --runner=DirectRunner \
    -- \
    --job_server_timeout=43200