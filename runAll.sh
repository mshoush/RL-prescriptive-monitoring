#!/bin/sh

for VARIABLE in  "bpic2012" "traficFines" "bpic2017"
do
	echo "start predictive...$VARIABLE";
	echo "Start first: $VARIABLE" ;
	echo "start predictive...";
	python predictive_model/get_catboost_pred_uncer.py $VARIABLE results/predicitve/$VARIABLE/ 20;
	sleep 5;

	echo "start causallift...";
	python causal/causallift_adapted.py $VARIABLE ./results/causal/$VARIABLE/  ./results/predicitve/$VARIABLE/;
	sleep 5;

	echo "start lower_upper_cate Zahra...";
	python causal/lower_upper_cate.py --data=$VARIABLE --results_dir=./results/causal/$VARIABLE/;
	sleep 5;

	echo "start Conformal causal ...";
	Rscript test_cfcuasal.r $VARIABLE
	sleep 5;

	echo "start conformal_prediction...";
	python conformal_prediction/conformal_prediction.py $VARIABLE ./results/predicitve/$VARIABLE/ ./results/conformal/$VARIABLE/ ./results/causal/$VARIABLE/;
	sleep 5;

	echo "start conformalized_survival...";
	python conformalized_survival_model_final_v1.py  $VARIABLE;
	sleep 5;

	echo "start prepare_data_for_RL...";
	python  prepare_data_for_RL_V2.py $VARIABLE;
	echo "StDoneart : $VARIABLE" ;
done














