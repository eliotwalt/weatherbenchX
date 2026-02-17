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
    --prediction=ens_mean \
    --target=era5 \
    --resolution=1440x721 \
    --time_start=2021-01-01 \
    --time_stop=2021-12-31 \
    --year=2021 \
    --lead_time_start=6 \
    --lead_time_stop=3000 \
    --lead_time_frequency=6 \
    --output_dir=./results \
    --lead_time_chunk_size=4 \
    --init_time_chunk_size=1 \
    --runner=FlinkRunner \
    -- \
    --flink_master=local \
    --environment_type=LOOPBACK \
    --flink_submit_uber_jar \
    --parallelism=32 \
    --flink_conf='taskmanager.memory.network.fraction=0.2' \
    --flink_conf='taskmanager.memory.network.min=256mb' \
    --flink_conf='taskmanager.memory.network.max=2gb'