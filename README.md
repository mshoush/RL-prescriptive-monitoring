# Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach

This project contains supplementary material for the article ["Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach"]("add link here") by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en) and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring approach that relies on reinforcement Learning an conformal prediction to an intervention policy under limited resources. 

This paper investigated the hypothesis that the incorporation of significance, urgency, and capacity factors can augment the process of training an RL agent for triggering interventions in the context of a prescriptive process monitoring system. The paper specifically investigated this question in the context where there is a limited number of resources available to perform interventions in the process.



# Dataset: 
Datasets can be downloaded from the following link:
* [BPIC2017, BPIC2012, and traficFines, i.e., a loan application and road fines processes](https://owncloud.ut.ee/owncloud/s/5zpcwR8rtpMC7Ko)



# Reproduce results:
To reproduce the results, please run the following:

* First, you need to install the environment using

                                     conda create -n <environment-name> --file requirements.txt

* Next, download the data folder from the abovementioned link

* Then run the following notebooks to prepare the datasets:
  
                                  prepare_trafficFines.ipynb
                                  prepare_data_bpic2012.ipynb
                                  prepare_data_bpic2017.ipynb

  
*   Next, run the following shell script to start experiments w.r.t the offline and online phases. Replace  datasetName by bpic2012, bpic2017, or traficFines. 


                                     ./runAll.sh
                                     ./test_results.sh <datasetName> <resultsDir>
                                     ./run_all_resources.sh <datasetName>
 
                                     

* Collect results according to RQ1 by running the following notebook: 

                                     plot_all_resources.ipynb
                                     
* Collect results according to RQ2 by running the following notebook: 

                                     plot_RL_results.ipynb



