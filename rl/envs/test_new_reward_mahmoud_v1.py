import numpy as np
from gym import spaces
from envs.baselineEnv import BaseEnv
import sys
from datetime import datetime
import time  
from .ResourceAllocator import allocateRes, block_and_release_res


class StatewithTempCostRewardwithoutPred_newReward(BaseEnv):
    metadata = {"render.modes": ["human"]}
    summary_writer = None

    def __init__(self):
        super().__init__()
        #################################
        # Action and state space for the environment
        #################################

        # initilize the state with none.
        self.state = None
        #self.results_dir

        # Action space: 1 or 0. trigger action or not to trigger action.
        self.action_space = spaces.Discrete(
            2
        )  # set action space to adaption true or false

        # resources
        min_resources = 0
        max_resources = np.iinfo(np.int64).max
        # Total Uncertainty
        min_uncer = 0.0
        max_uncer = np.finfo(np.float32).max
        # Idividual Treatment Effect (TE)
        min_ite = np.finfo(np.float32).min
        max_ite = np.finfo(np.float32).max
        #Conformalized TE
        # Lower
        min_lower_cte = np.finfo(np.float32).min
        max_lower_cte = np.finfo(np.float32).max
        # Upper
        min_upper_cte = np.finfo(np.float32).min
        max_upper_cte = np.finfo(np.float32).max
        # Conformalized Outcome Prediction
        min_cop = np.finfo(np.float32).min
        max_cop = np.finfo(np.float32).max
        # Time-to-event: Intervention Window (IW)
        min_iw = np.finfo(np.float32).min
        max_iw = np.finfo(np.float32).max
        # conformalized IW 
        # Lower
        min_lower_ciw = np.finfo(np.float32).min
        max_lower_ciw = np.finfo(np.float32).max
        # Upper
        min_upper_ciw = np.finfo(np.float32).min
        max_upper_ciw = np.finfo(np.float32).max


        if self.mode=="metzger" or self.mode=="metzger_adapted_to_zahra":  
            # State: (Relative Position, Reliability, Deviation)
            low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min])
            high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max])
        elif self.mode=="zahra":
            # state: (Relative Position, Conformal prediction (COP), Lower_Cate, Upper_Cate)
            low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min])
            high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max])
        elif self.mode=="mahmoud":
            # state: (ciw, cte, cop, TU, n)
            # state: (Lower intervention window, Upper intervention window, Lower CTE, Upper CTE, COP, TU, # Resources)
            low_array = np.array([0.0, min_lower_ciw, min_upper_ciw, min_lower_cte, min_upper_cte, min_cop, min_uncer, min_resources])
            high_array = np.array([1.0, max_lower_ciw, max_upper_ciw, max_lower_cte, max_upper_cte, max_cop, max_uncer, max_resources])
        else:
            raise SyntaxError("No Valid mode")

        self.observation_space = spaces.Box(low=low_array, high=high_array)

        # Intervention Window
        self.lower_ciw = 0.0
        self.upper_ciw = 0.0
        # Treatment effect
        self.lower_cte = 0.0
        self.upper_cte = 0.0
        # Outcome prediction
        self.cop = 0.0
        # Total Uncertainty
        self.tu = 0.0
        self.action=None
        
        self.nr_res = list(range(1, self.resources + 1, 1))


    def step(self, action=None):
        #print("STEP+++++++++++++++++++")
        
        if self.data.finished:
            self.close()  
            raise SystemExit("Out of data!")
        if action is None:
            action = 0


        # send action to the environment        
        self.send_action(int(action))
        self.action_value = action 

        # obtain reward after performing the action
        #print("++++++++++++++++++++++++++++++")
        self.receive_reward_and_state()
        print(f"recieved reward: {self.reward}")        

        self.do_logging(action)

        info = {}

        if self.mode=="metzger" or self.mode=="metzger_adapted_to_zahra":            
            self.state = self._create_state_metzger()
        elif self.mode=="zahra":
            self.state = self._create_state_zahra()
        elif self.mode=="mahmoud":
            self.state = self._create_state_mahmoud_v1(action)
        else:
            raise SyntaxError("No Valid mode")
 
        return self.state, self.reward, self.done, info

    

    def compute_reward(
        self,
        adapted,
        cost,
        gain,
        done,
        predicted_outcome,
        planned_outcome,
        reliability,
        position,
        process_length,
        nr_res,
        actual_outcome=0,
        true_effect=0,
        
    ):
 
        if self.mode=="mahmoud":  
            print(f"true_effect: {true_effect}")       
            
            if len(self.nr_res)>0:  
                if true_effect > 0.0:
                    if adapted:
                        reward = self.gain - self.cin + self.gain_res
                        self.true = True
                    else:
                        reward = (-1 * self.gain ) - self.cin - self.gain_res
                        self.true = False
                elif true_effect < 0.0:
                    if adapted:
                        reward = (-1 * self.gain ) - self.cin - self.gain_res
                        self.true = False
                    else:
                        reward =  self.cin + self.gain_res
                        self.true = True
                elif true_effect == 0.0:
                    if adapted:
                        reward = (-1 * self.cin) - self.gain_res
                        self.true = False
                    else:
                        reward = self.cin + self.gain_res
                        self.true = True
            else:
                if true_effect > 0.0:
                    if adapted:
                        reward = -1 * self.gain_res
                        self.true = False
                    else:
                        reward = self.gain_res
                        self.true = True
                elif true_effect < 0.0:
                    if adapted:
                        reward = -1 * self.gain_res
                        self.true = False
                    else:
                        reward = self.gain_res
                        self.true = True
                elif true_effect == 0.0:
                    if adapted:
                        reward = -1 * self.gain_res
                        self.true = False
                    else:
                        reward = self.gain_res
                        self.true = True
            # print(f"true_effect: {true_effect}")       
            
            # if len(self.nr_res)>0:  
            #     if true_effect > 0.0:
            #         if adapted:
            #             reward = -1 * (self.cn - self.cin)
            #             self.true = True
            #         else:
            #             reward = -1 * (self.cn)
            #             self.true = False
            #     elif true_effect < 0.0:
            #         if adapted:
            #             reward = (-1 * self.cn) - self.cin
            #             self.true = False
            #         else:
            #             reward = self.cin
            #             self.true = True
            #     elif true_effect == 0.0:
            #         if adapted:
            #             reward = -1 * self.cin
            #             self.true = False
            #         else:
            #             reward = self.cin
            #             self.true = True
            # else:
            #     if true_effect > 0.0:
            #         if adapted:
            #             reward = -1 * self.cin
            #             self.true = False
            #         else:
            #             reward = self.cin
            #             self.true = True
            #     elif true_effect < 0.0:
            #         if adapted:
            #             reward = -1 * self.cin
            #             self.true = False
            #         else:
            #             reward = self.cin
            #             self.true = True
            #     elif true_effect == 0.0:
            #         if adapted:
            #             reward = -1 * self.cin
            #             self.true = False
            #         else:
            #             reward = self.cin
            #             self.true = True

            
        elif self.mode == "metzger":
            if not done:
                reward = 0
            else:
                alpha = ((1. - 0.5) / process_length) * position
                violation = actual_outcome != planned_outcome
                if adapted:
                    if violation:
                        reward = alpha
                    else:
                        reward = -0.5 - alpha * 0.5
                else:
                    if violation:
                        reward = -1.
                    else:
                        reward = 1.



        elif self.mode == "metzger_adapted_to_zahra":

            alpha = true_effect
            violation = actual_outcome != planned_outcome
            if adapted:
                if violation and (true_effect > 0):
                    reward = 1
                elif violation and (true_effect <= 0):
                    reward = alpha
                elif not violation:
                    reward = -1
            else:
                if violation and (true_effect > 0):
                    reward = -1.
                elif violation and (true_effect <= 0):
                    reward = 1
                elif not violation:
                    reward = 1

 
        elif self.mode == "zahra":
            if adapted:
                if true_effect > 0:
                    reward = gain - cost
                    self.true = True
                elif true_effect == 0:
                    if actual_outcome==0:
                        reward = -1 * cost
                    if actual_outcome==1:
                        reward = -1 * cost
                    self.true = False
                elif true_effect < 0:
                    reward = -1 * cost - gain
                    self.true = False
            else:
                if true_effect > 0:
                    reward = -gain
                    self.true = False
                elif true_effect == 0:
                    if actual_outcome==0:
                        reward = gain
                        self.true = True
                    else:
                        reward = 0
                        self.true = True
                elif true_effect<0:
                    reward = gain
                    self.true = True

        return reward

    def reset(self):
        # self.send_action(-1)
        # self.treat_counter=0
        self.receive_reward_and_state()
        #print(self.state)
        self.treat_counter=0
        

    
        if self.mode=="metzger" or self.mode=="metzger_adapted_to_zahra":            
            self.state = self._create_state_metzger()
        elif self.mode=="zahra":
            self.state = self._create_state_zahra()
        elif self.mode=="mahmoud":
            self.state = self.state = self._create_state_mahmoud_v1(self.action)
        else:
            raise SyntaxError("No Valid mode")
        

        return self.state

    # def _create_state(self):
    #     relative_position = self.position / self.process_length
    #     self.state = np.array([relative_position, self.lower, self.upper])
        
    #     return self.state

    def _create_state_metzger(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.reliability, self.deviation])
        return self.state

    def _create_state_zahra(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.lower_cate, self.upper_cate])
        return self.state

    def _create_state_mahmoud_v1(self, action):

        lower_ciw = self.lower_ciw
        upper_ciw = self.upper_ciw


        lower_cte = self.lower_cte
        upper_cte = self.upper_cte
        cop = self.cop
        tu = self.tu


        if action==1:
            if self.nr_res:
                selceted_res = self.nr_res[0]
                allocateRes(selceted_res, "fixed", self.nr_res, float(self.tdur))
                print(f"Allocate the selected resource: {selceted_res} for case: {self.case_id}")
                self.nr_res.remove(self.nr_res[0])
            else:
                print("No availble resources")
        else:
            pass

        relative_position = self.position / self.process_length
        #self.state = np.array([relative_position, self.reliability, self.deviation, len(self.nr_res), wip, total_uncer, ite, conformal, time_to_event])
        # (Lower intervention window, Upper intervention window, Lower CTE, Upper CTE, COP, TU, # Resources)
        self.state = np.array([relative_position, lower_ciw, upper_ciw, lower_cte, upper_cte, cop, tu, len(self.nr_res)])
        #self.state = np.array([len(self.nr_res), wip, total_uncer, ite, conformal, time_to_event])

        return self.state

    def render(self, mode="human"):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # If the action is zero, then there is no adaptation. Otherwise else.
        adapted = self.recommend_treatment 
        print("HERE")
        print(f"TREATMEN_COUNTER:{self.treat_counter }")
        print(adapted)
        adapted = True if adapted == 1 else False
        print(adapted)
        

        if adapted:
            # This means that there a case recived the treatment
            self.treat_counter += 1
            print(f"TREATMEN_COUNTER:{self.treat_counter }")
            #raise SystemExit("ADAPTED")

        cost = self.cin # cost of the intervention
        gain = self.gain # the gain of triggering the intervention

        done = self.data.done
        event = self.data.get_event()
          
        done = True if done == 1 else False

        case_id = event["case_id"].iloc[0]
        ordinal_case_id = event["ordinal_case_ids"].iloc[0]

        ############# FOR MEtzger
        
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
        upper_ite = event["upper_cate"].iloc[0]
        upper_ite = float(upper_ite)
        lower_ite = event["lower_cate"].iloc[0]
        lower_ite = float(lower_ite)
        upper_cte = event["upper_naive"].iloc[0]
        upper_cte = float(upper_cte)
        lower_cte = event["lower_naive"].iloc[0]
        lower_cte = float(lower_cte)
        deviation = event['deviation'].iloc[0]
        deviation = float(deviation)       
        # Total uncertainty
        tu = event["total_uncer"].iloc[0]
        tu = float(tu)
        cop = event["alpha=0.9"].iloc[0]
        cop = float(cop)
        time_to_event = event['time_to_event_m'].iloc[0]

        upper_ciw = event["upper_time_to_event_naive"].iloc[0]
        upper_ciw = float(upper_ciw)

        lower_ciw = event["lower_time_to_event_naive"].iloc[0]
        lower_ciw = float(lower_ciw)

        probaIfTreated = event["Proba_if_Treated"].iloc[0]
        probaIfTreated = float(probaIfTreated)
        probaIfUnTreated = event["Proba_if_Untreated"].iloc[0]
        probaIfUnTreated = float(probaIfUnTreated)

        ite = probaIfTreated - probaIfUnTreated
        ite = float(ite)
        

        y0 = event["y0"].iloc[0] # predicted outcome if we do not apply the intervention 
        y1 = event["y1"].iloc[0] # predicted outcome if we apply the intervention
        # ite = y1 -y0
        # ite = y1 - y0
        # print(f"ITE: {ite}")
        # if we apply the intervention
        # this a new reaistic outcome for the agent
        # it's predicted from the causal model but it works as actual outcome for the agent.         
        actual_outcome_causal = y1 if adapted else y0 
        actual_outcome_causal = float(actual_outcome_causal)

        actual_outcome_predictive = event['actual'].iloc[0]
        actual_outcome_predictive = float(actual_outcome_predictive)

        actual_outcome = y1 if adapted else y0
        actual_outcome = float(actual_outcome)
        #actual_outcome = actual_outcome_causal

        
        # compute the reward        
        reward = self.compute_reward(
            adapted,
            cost,
            gain,
            done,
            predicted_outcome,
            planned_outcome,
            reliability,
            position,
            process_length,
            #actual_outcome=actual_outcome_predictive,
            actual_outcome=actual_outcome,
            true_effect=ite,
            nr_res=self.nr_res,
        )
        

        self.adapted = adapted
        self.cost = cost
        self.gain = gain
        self.done = done
        self.case_id = case_id
        self.ordinal_case_id=ordinal_case_id
        self.actual_outcome = y1 if self.adapted else y0
        
        self.actual_outcome_causal = y1 if self.adapted else y0
        self.actual_outcome_predictive = actual_outcome_predictive
        #self.actual_outcome = actual_outcome

        self.predicted_outcome = predicted_outcome
        self.planned_outcome = planned_outcome
        self.reliability = reliability
        self.reward = reward        
        # Intervention Window
        self.lower_ciw = lower_ciw
        self.upper_ciw = upper_ciw
        # Treatment effect
        self.lower_cte = lower_cte
        self.upper_cte = upper_cte
        # Outcome prediction
        self.ocp = cop
        # Total Uncertainty
        self.tu = tu

        self.time_to_event = time_to_event
        self.ite = ite

        self.reliability = reliability
        self.deviation = deviation
        
        

        # if the intervention is triggered in a cases then switch done to True.
        if adapted:
            done = True
            self.data.done = 1
            self.done = True

        
        if position != 0.0:
            self.position = position
            self.process_length = process_length
        return 0
