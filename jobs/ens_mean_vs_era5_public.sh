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

python public_benchmark/run_benchmark_evaluation.py \
    --config=public_configs \
    --prediction=ens_mean \
    --target=era5 \
    --resolution=1440x721 \
    --time_start=2020-01-01 \
    --time_stop=2020-12-31 \
    --year=2020 \
    --lead_time_start=6 \
    --lead_time_stop=366 \
    --lead_time_frequency=6 \
    --output_dir=./results \
    --lead_time_chunk_size=4 \
    --init_time_chunk_size=1 \
    --runner=DirectRunner \
    -- \
    --direct_running_mode=multi_threading \
    --direct_num_workers=1 \
    --sdk_worker_parallelism=1 \
    --direct_runner_control_port_deadline=3600