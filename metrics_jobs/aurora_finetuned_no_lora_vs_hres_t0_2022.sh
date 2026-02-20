# sbatch ./metrics_jobs/metrics.sh \
#     --prediction=aurora_finetuned_no_lora_hres_init \
#     --target=hres_t0 \
#     --year=2022 \
#     --output_dir=/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_hres_init_2022_15d_20-SDE-steps/

sbatch ./metrics_jobs/metrics.sh \
    --prediction=aurora_pretrained \
    --target=era5 \
    --year=2021 \
    --output_dir=/projects/prjs1808/ewalt1/Xaurora/train/large-flow-map_600K_0.1-noise/forecast_2021_15d_20-SDE-steps/