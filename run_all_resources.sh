#!/bin/bash
echo "Start with $1..."
for i in 1 2 3 
do
	echo "iteration... $i" ;
	for k in {1..15}
	do
		echo "Resources: $k"
		python rl/ppo_temp_cost_reward_noPred.py mahmoud ./results/resources_$1/resultsResources_$i/mahmoud $k 1 5 25 10 60 $1 > out.txt
		rm out.txt
		sleep 10
	done
done


