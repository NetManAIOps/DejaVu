#!/usr/bin/env bash
# shellcheck disable=SC2164
cd "$(dirname "$0")"/../..
DATASETS=${1:-A1 A2 B C D}

job_list=()
for dataset in ${DATASETS}; do
  for score_aggregation_method in min max mean sum; do
    job_list+=("python exp/DejaVu/run_random_walk_single_metric.py \
      --data_dir=/SSF/data/${dataset} -f=True \
      --score_aggregation_method=${score_aggregation_method} \
      --window_size 60 10 ")
  done
done

for dataset in ${DATASETS}; do
  for anomaly_score_aggregation_method in min max mean sum; do
    for corr_aggregation_method in min max mean sum; do
      job_list+=("python exp/DejaVu/run_random_walk_failure_instance.py \
        --data_dir=/SSF/data/${dataset} -f=True \
        --window_size 60 10 \
        --anomaly_score_aggregation_method=${anomaly_score_aggregation_method} \
        --corr_aggregation_method=${corr_aggregation_method}")
    done
  done
done


# the selected
#for dataset in A1 A2 B C; do
#  for anomaly_score_aggregation_method in mean; do
#    for corr_aggregation_method in min; do
#      job_list+=("python exp/DejaVu/run_random_walk_failure_instance.py \
#        --data_dir=/SSF/data/${dataset} -f=True \
#        --window_size 60 10 \
#        --anomaly_score_aggregation_method=${anomaly_score_aggregation_method} \
#        --corr_aggregation_method=${corr_aggregation_method}")
#    done
#  done
#done

jobs_file="/tmp/$(date +%s).random_walk.jobs"
rm ${jobs_file} || echo "rm existing jobs failed"
for job in "${job_list[@]}"; do
  echo $job | tee -a ${jobs_file}
done

./batch < ${jobs_file}
