# ================================
# SETUP ENVIRONMENT
# ================================
git clone git@github.com:google-research/weatherbenchX.git .
cd weatherbenchX
python3.11 -m venv env/venv
source env/venv/bin/activate
pip install -e .

# ================================
# RUN BENCHMARK
# full 2020 year from 6 hours to 
# 15 days lead time, every 6 hours
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
    --output_dir=./results/ \
    --runner=DirectRunner