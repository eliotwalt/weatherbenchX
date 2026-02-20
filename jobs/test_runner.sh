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
#SBATCH --cpus-per-task=96
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

# ================================
# parse command-line arguments:
# * --force-direct-runner
# ================================
FORCE_DIRECT_RUNNER=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force-direct-runner) FORCE_DIRECT_RUNNER=true ;;
        *) echo "Unknown parameter passed: $1" ; exit 1 ;;
    esac
    shift
done
echo "Force DirectRunner: $FORCE_DIRECT_RUNNER"

# ================================
# Environment setup
# ================================
set -e
set -u

echo "Running on node: $SLURMD_NODENAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

if [ "$FORCE_DIRECT_RUNNER" = true ]; then
    echo "Forcing DirectRunner for local execution"
    source env/venv_direct/bin/activate
    RUNNER_OPTION="""
    --job_server_timeout=43200
    """
    # RUNNER_OPTION="""
    # --direct_running_mode=multi_threading \
    # --direct_num_workers=0 \
    # --job_server_timeout=43200
    # """
else
    echo "Using default direct runner (Prism)"
    source env/venv/bin/activate
    RUNNER_OPTION="""
    --job_server_timeout=43200
    """
fi

# print apach_beam[gcp] version]
pip show apache-beam

# Optional but recommended to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ================================
# Run WeatherBenchX evaluation
# ================================

# time the execution
start_time=$(date +%s)
python xaurora_benchmark/run_benchmark_evaluation.py \
    --config=xaurora_configs \
    --prediction=aurora_small_pretrained \
    --target=era5 \
    --resolution=1440x721 \
    --time_start=2021-01-01 \
    --time_stop=2021-01-31 \
    --year=2021 \
    --lead_time_start=6 \
    --lead_time_stop=366 \
    --lead_time_frequency=6 \
    --output_dir=/scratch-shared/ewalt1/Xaurora/wbx_benchmark \
    --lead_time_chunk_size=1 \
    --init_time_chunk_size=8 \
    --runner=DirectRunner \
    -- \
    $RUNNER_OPTION
    
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"