sbatch ./metrics_jobs/metrics.sh \
    --prediction=xaurora_small_fm \
    --target=era5 \
    --year=2021 \
    --output_dir=/projects/prjs1808/ewalt1/Xaurora/train/16g/2026-02-10_15-11-33/forecast_2026-02-16_13-05-34/ \
    --lead_time_stop=126