#!/bin/bash

# List of datasets to process
datasets=("trafficFines" "bpic2017" "bpic2012")

# Loop through each dataset
for dataset in "${datasets[@]}"
do
    echo "Starting predictive model for $dataset...";
    # change the size of the ensemble as you need
    ens_size=1;
    python -W ignore  predictive_model/get_catboost_pred_uncer.py $dataset results/predictive/$dataset/ $ens_size > out.txt;
    rm out.txt
    sleep 5;

    echo "Starting causal analysis for $dataset...";
    python -W ignore causal/causallift_adapted.py $dataset ./results/causal/$dataset/ ./results/predictive/$dataset/ > out.txt;
    rm out.txt
    sleep 5;

    echo "Running lower and upper bounds CATE analysis for $dataset (Zahra)...";
    python -W ignore causal/lower_upper_cate.py --data=$dataset --results_dir=./results/causal/$dataset/ > out.txt;
    rm out.txt
    sleep 5;

    echo "Conducting conformal causal analysis for $dataset...";
    Rscript test_cfcuasal.r $dataset > out.txt;
    rm out.txt
    sleep 5

    echo "Starting conformal prediction for $dataset...";
    python -W ignore conformal_prediction/conformal_prediction.py $dataset ./results/predictive/$dataset/ ./results/conformal/$dataset/ ./results/causal/$dataset/ > out.txt;
    rm out.txt
    sleep 5;

    echo "Running conformalized survival model for $dataset...";
    python -W ignore conformalized_survival_model_final_v1.py $dataset > out.txt;
    rm out.txt
    sleep 5

    echo "Preparing data for Reinforcement Learning for $dataset..."
    python -W ignore prepare_data_for_RL_V2.py $dataset > out.txt;
    rm out.txt
    
    echo "Processing completed for $dataset.";
done
