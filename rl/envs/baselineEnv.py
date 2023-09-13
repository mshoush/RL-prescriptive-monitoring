import gym
import os
from gym import spaces
import matplotlib.pyplot as plt
import tensorflow as tf
import socket
import pandas as pd
import numpy as np

#if self.envm==0:
#from envs.envManager import envManager
from envs.envManager_v2 import envManager
# else:
#     from envs.envManager_v2 import envManager
import time
import ast


import sys





def get_average_last_entries_from_numeric_list(numeric_list: list, max_entries):
    return sum(numeric_list[-max_entries:]) / max_entries


def get_average_last_entries_from_numeric_list_excluding(numeric_list: list, max_entries, excluded_value, end_index):
    temp_list = numeric_list[np.maximum(0, end_index - max_entries):end_index + 1].copy()
    try:
        while True:
            temp_list.remove(excluded_value)
    except ValueError:
        pass
    if len(temp_list) == 0:
        return 0
    else:
        return sum(temp_list[:]) / len(temp_list)

class BaseEnv(gym.Env):
    show_graphs = False
    log_tensorboard = True
    experiment_number = 1
    
    global treatment_counter
    treat_counter_new=0

    

    def __init__(self):
        self.total_steps = 0
        self.episode_count = 1
        self.adapted_count = 0
        self.treat_counter_new = 0

        self.treated_cases = []

        self.treat_counter_new_dict = {}#{case_id: [before, after]}
        # load dataset, get cases, get event by event.
        # done when events related to the same case is done. 
        self.data = envManager() 
        

        self.results_dir = envManager().results_dir
        self.mode = envManager().mode
        self.resources = envManager().resources
        self.tdur = envManager().tdur
        self.cn = envManager().cn
        self.cin = envManager().cin 
        self.gain_res = envManager().gain_res  
        self.gain = envManager().gain  
        self.num_cases = envManager().get_num_of_cases() 
        self.component = envManager().component

        if self.data.finished != True:
            try:
                self.data.get_event_by_event()
            except:
                self.data.get_new_case()

 

        # learning parameter
        self.reward = 0
        self.rewards_per_episode = []
        self.action_value = 0
        self.case_id = -1
        self.actual_outcome = -1
        self.predicted_outcome = 0
        self.actual_treatment = 0
        self.planned_outcome = 0
        self.position = 0
        self.cost = 0
        self.process_length = 0
        self.done = 0
        self.adapted = 0
        self.reliability = 0
        self.recommend_treatment = 0
        self.treat_counter = 0

        self.predicted_proba_0 = 0
        self.predicted_proba_1 = 0

        self.ordinal_case_id =1

        #self.treat_counter_new = 0
        #treat_counter_new = 0
        
        self.true = True

        self.upper_cate=0
        self.lower_cate=0

        folder_name =  time.strftime("%b-%d-%H-%M-%S")
        self.summary_writer = tf.summary.create_file_writer('tensorboard/%s'%folder_name)

        self.actions = []  # Action pro Step
        self.rewards = []  # Reward pro Step (-Kosten)
        self.costs = []  # Kosten pro Step (d.h. 50/length bei nicht-adapt)
        self.adapted_no_violation = []  # 1 if adapted though no violation was predicted

        
        self.case_id_list = []
        self.pred_outcome = []
        self.pred_reliability = []
        self.outcomes= []
        self.treatments = []
        self.avg_adapted_100 = []  # Average end of episode over the last 100, between 1(adapted) and 0(adapted)
        self.tmp_cumul_reward_per_ep = []
        self.tmp_cumul_cost_per_ep = []

        # lists for metrics
        self.cumul_cost_per_ep = []  # Kumulative Kosten pro Episode
        self.cumul_reward_per_ep = []
        self.cumul_gain_per_ep = []# Kumulativer Reward pro Episode
        self.true_per_ep = []  # 1 if Episode ending decision is right, 0 otherwise
        self.adapt_in_ep = []  # Info ob in Episode adaptiert wurde
        self.case_length_per_episode = []  # Länge der Episode
        self.position_of_adaptation_per_episode = []  # Position der Adaption pro Episode; -1 := keine Adaption in der Episode
        self.earliness = []  # Position der Adaption durch Länge der Episode, -1 := keine Adaption in der Episode

        self.percentage_true_last_100 = []  # percentage of true decisions among the last 100


        self.elapsedTime = 0
        self.startTimes2 = 0


        print("\nHERE2\n")
        #startTime = datetime.datetime.now()
        self.startTimes2 = time.time() #datetime.datetime.now().second
        #print("Start time: %s", startTime)
        print(f"\nStart time in secs2: {self.startTimes2}\n")


    def send_action(self, action):        
        print(f"\nsending action...{self.case_id}")        
        #print(f"Resources: {len(self.nr_res)}")
        self.recommend_treatment = action
        print("action sent: " + str(action))

        if action==1:                        
            if self.case_id in self.treated_cases:
                print(f"Case: {self.case_id} is treated Before")
                #self.recommend_treatment = 0
            else:
                self.treated_cases.append(self.case_id)
                print(f"Case: {self.case_id} is treated with action: {action}")
            
        else:
            pass


    def receive_reward_and_state(self):
        adapted = self.recommend_treatment
        adapted = True if adapted == 1 else False
        if self.recommend_treatment:
            self.treat_counter += 1

        cost = self.cin
        gain = 50

        done = self.data.done
        
        event = self.data.get_event_by_event()
        done = True if done == 1 else False

        case_id = event["case_id"].iloc[0]
        


        ordinal_case_id = event["ordinal_case_ids"].iloc[0]

        actual_outcome = event['actual'].iloc[0]
        actual_outcome = float(actual_outcome)

        predicted_proba_1 = event['predicted_proba_1'].iloc[0]
        predicted_proba_1 = float(predicted_proba_1)

        predicted_proba_0 = event['predicted_proba_0'].iloc[0]
        predicted_proba_0 = float(predicted_proba_0)

        ############# FOR MEtzger
        actual_outcome_predictive = event['actual'].iloc[0]
        actual_outcome_predictive = float(actual_outcome_predictive)

        predicted_outcome = event['predicted'].iloc[0]
        predicted_outcome = float(predicted_outcome)

        reliability = event['reliability'].iloc[0]
        reliability = float(reliability)

        planned_outcome = 0
        planned_outcome = float(planned_outcome)


        position = event["prefix_nr"].iloc[0]
        position = float(position)

        process_length = event["case_length"].iloc[0]
        process_length = float(process_length)

        

        upper_ite = event["upper_cate"]
        upper_ite = float(upper_ite)

        lower_ite = event["lower_cate"]
        lower_ite = float(lower_ite)

        upper_cte = event["upper_counterfactual"]
        upper_cte = float(upper_cte)

        lower_cte = event["lower_counterfactual"]
        lower_cte = float(lower_cte)

        deviation = event['deviation']
        deviation = float(deviation)

        # Total uncertainty
        tu = event["total_uncer"].iloc[0]
        tu = float(tu)

        cop = event["alpha=0.9"].iloc[0]
        cop = float(cop)


        time_to_event = event['time_to_event_m'].iloc[0]

        upper_ciw = event["upper_time_to_event_adaptive_QR"]
        upper_ciw = float(upper_ciw)

        lower_ciw = event["lower_time_to_event_adaptive_QR"]
        lower_ciw = float(lower_ciw)

        probaIfTreated = event["Proba_if_Treated"]
        probaIfTreated = float(probaIfTreated)

        probaIfUnTreated = event["Proba_if_Untreated"]
        probaIfUnTreated = float(probaIfUnTreated)

        ite = probaIfTreated - probaIfUnTreated
        ite = float(ite)


        y0 = event["y0"].iloc[0] # predicted outcome if we do not apply the intervention 
        y1 = event["y1"].iloc[0] # predicted outcome if we apply the intervention
        #ite = y1 - y0
       
        # if we apply the intervention
        # this a new reaistic outcome for the agent
        # it's predicted from the causal model but it works as actual outcome for the agent.         
        actual_outcome_causal = y1 if adapted else y0 
        actual_outcome_causal = float(actual_outcome_causal)
        # done = True if done == 1 else False
        actual_outcome = y1 if adapted else y0 
        actual_outcome = float(actual_outcome)



        # compute the reward
        reward = self.compute_reward(adapted, cost, done, predicted_outcome, planned_outcome, reliability, position,
                                     process_length, actual_outcome=actual_outcome)

        self.adapted = adapted
        self.cost = cost
        self.gain = gain
        self.done = done
        self.case_id = case_id
        self.actual_outcome = y1 if self.adapted else y0
        self.predicted_outcome = predicted_outcome
        self.planned_outcome = planned_outcome
        self.reliability = reliability
        self.reward = reward
        self.predicted_proba_0 = predicted_proba_0
        self.predicted_proba_1 = predicted_proba_1
        self.ordinal_case_id = ordinal_case_id


        if position != 0.:
            self.position = position
            self.process_length = process_length

        return reward, done, predicted_outcome, planned_outcome, reliability, position, process_length, cost, adapted

    def log_with_tensorboard(self, tag, simple_value, step):
        if BaseEnv.log_tensorboard:
            # self.summary_writer = tf.summary.create_file_writer('tensorboard/mymetrics')
            with self.summary_writer.as_default():
                tf.summary.scalar(tag, simple_value,step=step)
            # summary = tf.summary(value=[tf.summary.Value(tag=tag, simple_value=simple_value)])
            # self.summary_writer.add_summary(summary, step)

    def do_logging(self, action):
        # Reward, Kosten und Aktion in Gesamtliste speichern:
        self.rewards.append(self.reward)
        self.costs.append(self.cost)
        self.actions.append(int(action))
        if self.predicted_outcome < self.planned_outcome:
            # pred_violation = 1
            adap_no_pred = 0
        else:
            # pred_violation = 0
            if int(action) == 1:
                adap_no_pred = 1
            else:
                adap_no_pred = 0

        # self.violation_predicted.append(pred_violation)
        self.adapted_no_violation.append(adap_no_pred)
        # Reward, Kosten und Aktion in temporaeren Listen fuer Episode speichern:
        self.tmp_cumul_cost_per_ep.append(self.cost)
        self.tmp_cumul_reward_per_ep.append(self.reward)
        # self.tmp_avg_action_per_ep.append(int(action))

        self.log_with_tensorboard(tag='custom/adapted_though_no_violation_predicted', simple_value=adap_no_pred,
                                  step=self.total_steps)

        self.total_steps += 1
        if self.total_steps % 1000 == 0:
            if len(self.percentage_true_last_100) > 0:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count) + " with " + str(
                    self.percentage_true_last_100[-1]) + " true decisions")
            else:
                print("Step " + str(self.total_steps) + " in episode " + str(self.episode_count))

        if self.done:  # Ende einer Episode
            self.treat_counter_new = 0
            before = self.treat_counter_new 
            #print(f"treatment_counter before: {self.treat_counter_new }")
            
            self.treat_counter_new  =  sum([x[4] == True in x for x in self.ongoing_cases.get(self.case_id)])
            #print(self.ongoing_cases)
            print(self.case_id)
            

            # gainsite = []
            # [
            #     gainsite.append((x[6] * self.cn) - self.cin) if x[4] == True else ""
            #     for x in self.ongoing_cases.get(self.case_id)

            #     #gainsite.append(x[6]) if x[4] == True else "" for x in self.ongoing_cases.get(self.case_id)
            # ]
            # gainITE = sum(gainsite)
            # # #print(gainsite)
            # print(gainITE)
            #raise SystemExit("Done")
            
            
            #list({k: self.get_treatment_count(v) for k, v in {i: l for l in self.ongoing_cases.get(self.case_id) for i in range(1, len(self.ongoing_cases.get(self.case_id)) + 1)}.items()}.values())[-1]
            after = self.treat_counter_new 
            #print(f"treatment_counter after: {self.treat_counter_new }")

            self.treat_counter_new_dict[self.case_id]=[before, after]
            self.case_id_list.append(self.case_id)
            # Speichern der kumulativen und durschnittlichen Episodenkosten
            cumul_episode_cost = sum(self.tmp_cumul_cost_per_ep)
            self.cumul_cost_per_ep.append(cumul_episode_cost)
            self.tmp_cumul_cost_per_ep = []

            self.actual_outcome = 1 if self.actual_outcome==0 else 0
           
            # if self.actual_outcome==0:
            #     self.actual_outcome=1
            # else:
            #     self.actual_outcome=0
            # ep_gain = (self.gain * self.actual_outcome) - (self.treat_counter_new*self.cin)
            # print(self.case_id)
            # print(self.treated_cases)
            # if self.case_id in self.treated_cases:
            #     gainsite = []
            #     [
            #         gainsite.append((x[6] * self.cn) - self.cin) if x[4] == True else ""
            #         for x in self.ongoing_cases.get(self.case_id)

            #         #gainsite.append(x[6]) if x[4] == True else "" for x in self.ongoing_cases.get(self.case_id)
            #     ]
            #     gainITE = sum(gainsite)
            #     # #print(gainsite)
            #     print(gainITE)
            #     # print(self.case_id)
            #     # print(self.treated_cases)
            #     #print(self.treated_cases)
            #     ep_gain = (self.gain * self.actual_outcome) - (self.treat_counter_new*self.cin)
            #     #ep_gain = (gainITE * self.gain * self.actual_outcome)
            #     # print(ep_gain)
            #     # print(self.ongoing_cases.get(self.case_id))
            #     #raise SystemExit()
            # else:
            #     ep_gain = 0
            
                
            ep_gain = (self.gain * self.actual_outcome) - (self.treat_counter_new*self.cin) 
            # print(self.case_id)
            # print(self.treated_cases)

            #ep_gain = gainITE#(self.gain * self.actuteal_outcome) - (self.treat_counter_new*self.cin)
            #ep_gain = (gainITE* self.actual_outcome) #- (self.treat_counter_new*self.cin)
            print(f"treatment_counter: {self.treat_counter_new }")

            #ep_gain = 
            # if self.mode=="mahmoud":
            #     ep_gain = (self.gain * self.actual_outcome * self.treat_counter_new ) - (self.treat_counter_new *self.cost)
            # else:
            #     ep_gain = (self.gain * self.actual_outcome * self.treat_counter_new ) - (self.treat_counter_new *self.cost)
            
            print(f"Episode: {self.episode_count},\
                \nfor case: {self.case_id}\
                is finished,\
                \nwith treat_counter_new: {self.treat_counter_new },\
                \nEpisode Gain: {ep_gain}")
            print(self.ongoing_cases.get(self.case_id))
            #raise SystemExit("EPISODE DONE")

            # if treat_counter_new >=50:
            #     raise SystemExit("EXIT TREATMENT")
            #print(f"ITE Done: {self.ite}")
            self.cumul_gain_per_ep.append(ep_gain)
            avg_gain_100_value = get_average_last_entries_from_numeric_list(self.cumul_gain_per_ep,100)
            self.log_with_tensorboard(tag='episode_reward/episode_net_gain*', simple_value=avg_gain_100_value,
                                      step=self.episode_count)

            # if self.episode_count==1:
            #     print(self.treat_counter_new_dict)
            #     print(self.ongoing_cases[self.case_id])
            #     #print(self.ongoing_cases)
            #     print(f"=====================self.case_id_list: {self.case_id_list}==============")
            #     raise SystemExit("Done")
            # print(f"=====================self.case_id_list: {len(self.case_id_list)}==============")
            # print(f"Episode: { self.episode_count}")
            #raise SystemExit("Done")

            # Speichern des kumulativen und durschnittlichen Rewards pro Episode
            cumul_episode_reward = sum(self.tmp_cumul_reward_per_ep)

            # avg_episode_reward = cumul_episode_reward / len(self.tmp_cumul_reward_per_ep)
            self.cumul_reward_per_ep.append(cumul_episode_reward)
            # self.avg_reward_per_ep.append(avg_episode_reward)
            self.tmp_cumul_reward_per_ep = []

            # Speichern der durschnittlichen Aktionen einer Episode und ob adaptiert wurde
            # avg_actions = sum(self.tmp_avg_action_per_ep) / len(self.tmp_avg_action_per_ep)
            # self.avg_action_per_ep.append(avg_actions)
            # elf.tmp_avg_action_per_ep = []

            true_negative_status = 1 if self.true else 0
            self.true_per_ep.append(true_negative_status)
            self.pred_outcome.append(self.predicted_outcome)
            # self.pred_prob.append(self.predicted_proba)
            self.outcomes.append(self.actual_outcome)
            self.pred_reliability.append(self.reliability)
            #print(f"\n self.actual_treatment: {self.actual_treatment}\n")
            self.treatments.append(self.actual_treatment)
            if self.adapted:
                self.adapt_in_ep.append(1)

                self.position_of_adaptation_per_episode.append(self.position)
                self.earliness.append(self.position / self.process_length)
                self.log_with_tensorboard(tag='episode_result/earliness',
                                          simple_value=self.earliness[-1],
                                          step=self.episode_count)
                # self.true_positive_per_positive.append(true_negative_status)
            else:
                self.adapt_in_ep.append(0)
                self.position_of_adaptation_per_episode.append(-1)
                self.earliness.append(-1)

            self.case_length_per_episode.append(self.process_length)
            self.true = True
            self.treat_counter = 0
            #print(f"self.treat_counter: {self.treat_counter}")
            


            # self.true_negative_per_negative.append(true_negative_status)
            avg_adapted_100_value = sum(self.adapt_in_ep[-100:]) / 100
            self.avg_adapted_100.append(avg_adapted_100_value)

            percentage_true_last_100_value = get_average_last_entries_from_numeric_list(self.true_per_ep, 100)
            self.percentage_true_last_100.append(percentage_true_last_100_value)
            # percentage_true_positive_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_positive_per_positive, 100)
            # self.percentage_true_positive_100.append(percentage_true_positive_100_value)
            # percentage_true_negative_100_value = get_average_last_entries_from_numeric_list(
            #    self.true_negative_per_negative, 100)
            # self.percentage_true_negative_100.append(percentage_true_negative_100_value)

            # self.log_with_tensorboard(tag='episode_reward/episode_cost', simple_value=-cumul_episode_cost,
            #                           step=self.episode_count)
            self.log_with_tensorboard(tag='episode_reward/episode_reward*', simple_value=cumul_episode_reward,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapted', simple_value=self.adapted,
            #                           step=self.episode_count)
            self.log_with_tensorboard(tag='episode_result/average_adapted_last_100_episodes',
                                      simple_value=avg_adapted_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/average_action', simple_value=avg_actions,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/ended_correctly', simple_value=true_negative_status,
            #                           step=self.episode_count)
            # if self.adapted:
            #     self.log_with_tensorboard(tag='episode_result/positives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.adapted_count)
            # else:
            #     self.log_with_tensorboard(tag='episode_result/negatives_ended_correctly',
            #                               simple_value=true_negative_status,
            #                               step=self.episode_count - self.adapted_count)
            self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_episodes',
                                      simple_value=percentage_true_last_100_value,
                                      step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_positives',
            #                           simple_value=percentage_true_positive_100_value,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/percentage_of_trues_among_last_100_negatives',
            #                           simple_value=percentage_true_negative_100_value,
            #                           step=self.episode_count)
            #
            # self.log_with_tensorboard(tag='episode_result/percentage_Correct_Adaptation_Decisions',
            #                           simple_value=(self.true_per_ep[-1000:].count(1) / 1000) * 100,
            #                           step=self.episode_count)
            # self.log_with_tensorboard(tag='episode_result/adapt_in_ep',
            #                           simple_value=(self.adapt_in_ep.count(1) / self.adapt_in_ep.__len__()) * 100,
            #                           step=self.episode_count)
            self.episode_count += 1
            #treat_counter_new =0
            #print(f"\n self.episode_count: {self.episode_count } for case: {self.case_id}")
            if self.adapted:
                self.adapted_count += 1
            if self.data.finished != True:
                #self.data.get_new_case()
                try:
                    self.data.get_event_by_event()
                except:
                    self.data.get_new_case()

    def compute_reward(self, adapted, cost, done, predicted_outcome, planned_outcome, reliability, position,
                       process_length, actual_outcome=0.):
        pass

    # def reset(self):
    #     pass

    # def render(self, mode='human'):
    #     pass

    # def step(self, action):
    #     pass

    def close(self):
        print("Closed!")
        self.plot_experiment_data()
        self.write_experiment_data_to_csv(
            os.path.basename(self.__class__.__name__) + "_" + str(BaseEnv.experiment_number))
        BaseEnv.experiment_number += 1

    def plot_experiment_data(self):
        print("plotting_data")

    def write_experiment_data_to_csv(self, csv_name):

        earliness_avg = []
        true_avg_100 = []
        true_avg_1000 = []
        adapt_avg = []
        cost_avg = []
        reward_avg = []
        for ep in range(0, len(self.true_per_ep)):
            earliness_avg.append(get_average_last_entries_from_numeric_list_excluding(self.earliness, 100, -1, ep))
            true_avg_100.append(get_average_last_entries_from_numeric_list_excluding(self.true_per_ep, 100, -1, ep))
            true_avg_1000.append(get_average_last_entries_from_numeric_list_excluding(self.true_per_ep, 100, -1, ep))
            adapt_avg.append(get_average_last_entries_from_numeric_list_excluding(self.adapt_in_ep, 100, -1, ep))
            cost_avg.append(get_average_last_entries_from_numeric_list_excluding(self.cumul_cost_per_ep, 100, -1, ep))
            reward_avg.append(get_average_last_entries_from_numeric_list_excluding(self.cumul_reward_per_ep, 100, -1, ep))

        dataframe_of_metrics = pd.DataFrame(list(zip(self.case_id_list,
                                                     earliness_avg,
                                                     true_avg_100,
                                                     true_avg_1000,
                                                     adapt_avg,
                                                     self.adapt_in_ep,
                                                     cost_avg,
                                                     self.cumul_cost_per_ep,
                                                     reward_avg,
                                                     self.cumul_reward_per_ep,
                                                     self.true_per_ep,
                                                     self.position_of_adaptation_per_episode,
                                                     self.case_length_per_episode,
                                                     self.pred_outcome,
                                                     self.pred_reliability,
                                                     self.outcomes,
                                                     self.treatments,
                                                     self.cumul_gain_per_ep,
                                                     #self.elapsedTime
                                                     )),
                                            columns=['case_id',
                                                     'earliness_avg',
                                                     'true_avg_100',
                                                     'true_avg_1000',
                                                     'adaption_rate_avg',
                                                     'adapt_per_ep',
                                                     'costs_avg',
                                                     'cost_per_ep',
                                                     'rewards_avg',
                                                     'reward_per_ep',
                                                     'true_per_ep',
                                                     'position_adaptation_per_ep',
                                                     'case_length_per_ep',
                                                     'predicted outcome',
                                                     'reliability',
                                                     'actual outcome',
                                                     'actual treatment',
                                                     'gain',
                                                     #"elapsedTime"
                                                     ])

        #results_rl = "results/RL/bpic2012/mahmoud/" 
        results_rl = self.results_dir 
        print(results_rl)

    
        # create results directory
        if not os.path.exists(os.path.join(results_rl)):
            os.makedirs(os.path.join(results_rl))  
        
        endTimes2 = time.time() #datetime.datetime.now().second  #time.time() #datetime.now().total_seconds()
        #print(f"endTime: {endTime}")
        print(f"endTime2 in secs: {endTimes2}")
        elapsedTimes = endTimes2 - self.startTimes2
        print(f"ElapsedTimes: {elapsedTimes}")

        dataframe_of_metrics["elapsedTime"] = elapsedTimes
        dataframe_of_metrics.to_csv(os.path.join(results_rl,  "test_newReward_%s_%s_%s_%s_%s_%s_.csv"%(self.resources,
         self.tdur, self.cn, self.cin, self.gain_res, self.gain)), sep=";", index=False)

        # pd.DataFrame.from_dict(self.ongoing_cases,orient='index').transpose().to_csv(os.path.join(results_rl,  "ongoing_cases_%s_%s_%s_%s_%s_%s_.csv"%(self.resources,
        #  self.tdur, self.cn, self.cin, self.gain_res, self.gain)), sep=";", index=False)
        #dataframe_of_metrics.to_csv('dataframe_of_metrics_mahmoud_to_metzger_tdur30.csv')

        dataframe_of_metrics.plot(subplots=True)

        plt.tight_layout()
        if BaseEnv.show_graphs:
            plt.show()

        

        # dataframe_of_metrics.to_csv(csv_name + '_metrics_of_episodes.csv', header=True, index=False)

        # dataframe_of_metrics.to_csv(os.path.join(results_rl, csv_name + '_metrics_of_episodes.csv'), sep=";", index=False)
