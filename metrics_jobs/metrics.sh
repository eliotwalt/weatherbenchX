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
# Parse command-line arguments:
# * --prediction
# * --target
# * --year
# * --output_dir
# ================================
PREDICTION=""
TARGET=""
YEAR=""
OUTPUT_DIR=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prediction=*) PREDICTION="${1#*=}" ;;
        --target=*) TARGET="${1#*=}" ;;
        --year=*) YEAR="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1" ; exit 1 ;;
    esac
    shift
done

# make sure all required arguments are provided
if [ -z "$PREDICTION" ] || [ -z "$TARGET" ] || [ -z "$YEAR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --prediction=<prediction> --target=<target> --year=<year> --output_dir=<output_dir>"
    exit 1
fi

# print parsed arguments
echo "Prediction: $PREDICTION"
echo "Target: $TARGET"
echo "Year: $YEAR"
echo "Output directory: $OUTPUT_DIR"

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
    --prediction=$PREDICTION \
    --target=$TARGET \
    --resolution=1440x721 \
    --time_start=$YEAR-01-01 \
    --time_stop=$YEAR-12-31 \
    --year=$YEAR \
    --lead_time_start=6 \
    --lead_time_stop=366 \
    --lead_time_frequency=6 \
    --output_dir=$OUTPUT_DIR \
    --lead_time_chunk_size=1 \
    --init_time_chunk_size=8 \
    --runner=DirectRunner \
    -- \
    --job_server_timeout=43200