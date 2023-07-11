#!/bin/sh

echo "Start first: $1 $2" ;
echo "Start Mahmoud..." ;
# python rl/ppo_temp_cost_reward_noPred.py <mode> <results_dir> <nr of availble resources>  <tdur>  <cost of negative outcome>  <cost of intervention> <gain_res>  <gain_outcome>  <dataset_name>   
python rl/ppo_temp_cost_reward_noPred.py mahmoud ./results/$1/$1/$2/mahmoud 1 3 5 25 10 60 $1 > out.txt;
rm out.txt&
sleep 250;

echo "Start Zahra..." &
python rl/ppo_temp_cost_reward_noPred.py zahra ./results/$1/$1/$2/zahra 1 3 5 25 10 60 $1 > out1.txt;
sleep 250;
rm out1.txt &

echo "Start Metzger 1..." &
python rl/ppo_temp_cost_reward_noPred.py metzger ./results/$1/$1/$2/metzger 1 3 5 25 10 60 $1 > out2.txt;
sleep 250;
rm out2.txt &

echo "Done All..." ;




