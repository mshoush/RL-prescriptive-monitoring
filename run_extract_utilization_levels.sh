#!/bin/bash

echo "Start with $1..."

case $1 in
    "bpic2012")
        max_range=13
        ;;
    "bpic2017")
        max_range=10
        ;;
    "trafficFines")
        max_range=60
        ;;
    *)
        echo "Invalid value for $1"
        exit 1
        ;;
esac

for i in 1 2 3
do
    echo "iteration... $i"
    for ((k = 1; k <= max_range; k++))
    do
        echo "Resources: $k out of: $max_range"
        python rl/ppo_temp_cost_reward_noPred.py mahmoud ./resultsICPMTest/resourcesv2_$1/resultsResources_$i/mahmoud $k 1 5 25 10 60 $1 all > out.txt
        rm out.txt
        sleep 10
    done
done
