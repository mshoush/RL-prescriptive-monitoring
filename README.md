# Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach

This project contains supplementary material for the article ["Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach"]("add link here") by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en) and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring approach that relies on reinforcement Learning an conformal prediction to an intervention policy under limited resources. 

This paper investigated the hypothesis that the incorporation of significance, urgency, and capacity factors can augment the process of training an RL agent for triggering interventions in the context of a prescriptive process monitoring system. The paper specifically investigated this question in the context where there is a limited number of resources available to perform interventions in the process.



# Dataset: 
Datasets can be downloaded from the following link:
* [BPIC2017, BPIC2012, and traficFines, i.e., a loan application and road fines processes](https://owncloud.ut.ee/ownclouds/5zpcwR8rtpMC7Ko)



# Reproduce results:
To reproduce the results, please run the following:

* First, you need to install the environment using

                                     conda create -n <environment-name> --file requirements.txt

* Next, download the data folder from the abovementioned link

* Then run the following notebooks to prepare the datasets:
  
                                  prepare_trafficFines.ipynb
                                  prepare_data_bpic2012.ipynb
                                  prepare_data_bpic2017.ipynb

  
*
*   shell script to start experiments w.r.t the training, calibration, and testing phases.


                                     ./run_training_calibration.sh
                                     

* Collect results according to EQ1 by running the following script and replace dataname with "bpic2012" or "bpic2017". 

                                     python  plot_EQ1.py <dataname>
                                     
* Collect results according to EQ2 by running the following "plot_EQ2.ipynb" notebook.  

                                     
                                     

