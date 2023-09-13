import numpy as np
from gym import spaces
from envs.baselineEnv import BaseEnv
import sys
from datetime import datetime
import time  
from .ResourceAllocator import allocateRes, block_and_release_res
from datetime import datetime


class StatewithTempCostRewardwithoutPred_v2_newReward(BaseEnv):
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

        self.treated_cases = []

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

        # resources
        min_ongoingCases = 0
        max_ongoingCases = np.iinfo(np.int64).max

        # Upper
        min_arrival = np.finfo(np.float32).min
        max_arrival = np.finfo(np.float32).max

        # Upper
        min_finishing = np.finfo(np.float32).min
        max_finishing  = np.finfo(np.float32).max


        # resources
        min_cases = 1
        max_cases = np.iinfo(np.int64).max

        if self.mode=="metzger" or self.mode=="metzgeradaptedtozahra":  
            # State: (Relative Position, Reliability, Deviation)
            low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min])
            high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max])
        elif self.mode=="zahra":
            # state: (Relative Position, Lower_Cate, Upper_Cate,  Conformal prediction (COP), )
            low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min, min_cop, ])
            high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max, max_cop, ])
        elif self.mode=="mahmoud":         
            # importance_lower =  min_lower_cte, min_upper_cte, min_cop, min_uncer
            # urgency_lower = min_lower_ciw, min_upper_ciw
            # capacity_lower =  0.0, min_arrival

            # importance_upper = max_lower_cte, max_upper_cte, max_cop, max_uncer
            # urgency_upper = max_lower_ciw, max_upper_ciw
            # capacity_upper =  1.0, max_arrival

            # State: (Relative position, percentage of avaible resources,
            #         lower_CTE, upper_CTE, COP, TU,
            #         lower_CIW, upper_CIW,
            #         percentage of ongoing cases, arrival rate)
            if self.component=="all":
                low_array = np.array([min_cases, 0.0, 0.0,
                    min_lower_cte, min_upper_cte,
                    min_cop, min_uncer,
                    min_lower_ciw, min_upper_ciw,
                    0.0, min_arrival])

                high_array = np.array([max_cases, 1.0, 1.0,
                    max_lower_cte, max_upper_cte,
                    max_cop, max_uncer,
                    max_lower_ciw, max_upper_ciw,
                    1.0, max_arrival])

                #pass
            elif self.component=="withoutTU":
                low_array = np.array([min_cases, 0.0, 0.0,
                    min_lower_cte, min_upper_cte,
                    min_cop, #min_uncer,
                    min_lower_ciw, min_upper_ciw,
                    0.0, min_arrival])

                high_array = np.array([max_cases, 1.0, 1.0,
                    max_lower_cte, max_upper_cte,
                    max_cop, #max_uncer,
                    max_lower_ciw, max_upper_ciw,
                    1.0, max_arrival])
                #pass
            elif self.component=="withoutCIW":
                low_array = np.array([min_cases, 0.0, 0.0,
                    min_lower_cte, min_upper_cte,
                    min_cop, min_uncer,
                    #min_lower_ciw, min_upper_ciw,
                    0.0, min_arrival])

                high_array = np.array([max_cases, 1.0, 1.0,
                    max_lower_cte, max_upper_cte,
                    max_cop, max_uncer,
                    #max_lower_ciw, max_upper_ciw,
                    1.0, max_arrival])
                #pass
            elif self.component=="withCATE":
                low_array = np.array([min_cases, 0.0, 0.0,
                    min_lower_cte, min_upper_cte,
                    min_cop, min_uncer,
                    min_lower_ciw, min_upper_ciw,
                    0.0, min_arrival])

                high_array = np.array([max_cases, 1.0, 1.0,
                    max_lower_cte, max_upper_cte,
                    max_cop, max_uncer,
                    max_lower_ciw, max_upper_ciw,
                    1.0, max_arrival])
                #pass
            else:
                print("No Vaild Component")





            #low_array = np.array([min_cases, 0.0, 0.0, 
            #    min_lower_cte, min_upper_cte, 
            #    min_cop, min_uncer, 
            #    min_lower_ciw, min_upper_ciw,
            #    0.0, min_arrival])

           # high_array = np.array([max_cases, 1.0, 1.0,  
             #   max_lower_cte, max_upper_cte, 
             #    max_cop, max_uncer,
             #   max_lower_ciw, max_upper_ciw,
             #   1.0, max_arrival])


        else:
            raise SyntaxError("No Valid mode")

        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.ongoing_cases = {}

        self.visited_cases = {}
        self.visited_cases_arrival_times = {}
        self.visited_cases_finishing_times = {}

        self.nr_ongoing_cases =0
        self.arrival_time = 0
        self.finishing_time = 0
        self.cases_treat_counter = {}



        # Intervention Window
        self.lower_ciw = 0
        self.upper_ciw = 0
        # Treatment effect
        self.lower_cte = 0
        self.upper_cte = 0
        # Outcome prediction
        self.cop = 0
        # Total Uncertainty
        self.tu = 0
        self.action=None
        self.ite = 0

        # self.predicted_proba_0 = 0
        # self.predicted_proba_1 = 1
        
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

        # if self.case_id in self.treated_cases:
        #     print(case_id)
        #     event = self.data.get_event_by_event()
        #     case_id = event["case_id"].iloc[0]
        #     #print(self.treated_cases)
        #     print(f"\n+++++++++++++++++++++++++++{len(self.treated_cases)}++++++++++++++++++++++\n")
        #     if self.data.finished:
        #         self.close()  
        #         raise SystemExit("Out of data!")
        
        
        self.receive_reward_and_state()
        print(f"recieved reward: {self.reward}")        

        self.do_logging(action)

        info = {}

        if self.mode=="metzger" or self.mode=="metzgeradaptedtozahra":            
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
        actual_outcome=0.0,
        actual_outcome_predictive=0.0,
        true_effect=0,
        
    ):

                # alpha = true_effect
                # violation = actual_outcome != planned_outcome
                # if adapted:
                #     if violation and (true_effect > 0):
                #         reward = 1
                #     elif violation and (true_effect <= 0):
                #         reward = alpha
                #     elif not violation:
                #         reward = -1
                # else:
                #     if violation and (true_effect > 0):
                #         reward = -1.
                #     elif violation and (true_effect <= 0):
                #         reward = 1
                #     elif not violation:
                #         reward = 1

        if self.mode=="mahmoud": 
            capacity = True if len(self.nr_res) > 0 else False
            if capacity:
                if adapted: 
                    if true_effect > 0:
                        reward = (gain * true_effect) - cost + self.gain_res
                        self.true = True
                    elif true_effect == 0:
                        if actual_outcome==0: # +ve outcome
                            reward = -1 * cost - self.gain_res
                        if actual_outcome==1: #  # -ve outcome
                            reward = -1 * cost - gain - self.gain_res
                        self.true = False
                    elif true_effect < 0:
                        reward = -1 * cost - gain - self.gain_res
                        self.true = False
                else:
                    if true_effect > 0:
                        reward = -gain - self.gain_res
                        self.true = False
                    elif true_effect == 0:
                        if actual_outcome==0: # +ve outcome
                            reward = gain + self.gain_res
                            self.true = True
                        else:
                            reward = 0
                            self.true = True
                    elif true_effect<0:
                        reward = gain + self.gain_res
                        self.true = True
            else:
                if adapted:
                    reward = -1 * self.gain_res
                    self.true = False
                else:
                    reward = self.gain_res
                    self.true = True

                #reward= -1 * self.gain_res
             
            
        elif self.mode == "metzger":

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


        elif self.mode == "metzgeradaptedtozahra":

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
                    reward = (gain * true_effect) - cost
                    self.true = True
                elif true_effect == 1:
                    if actual_outcome==0: # +ve outcome
                        reward = -1 * cost
                    if actual_outcome==1: # -ve outcome
                        reward = -1 * cost - gain
                    self.true = False
                elif true_effect < 0:
                    reward = -1 * cost - gain
                    self.true = False
            else:
                if true_effect > 0:
                    reward = -gain
                    self.true = False
                elif true_effect == 0:
                    if actual_outcome==0: # +ve outcome
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
        
        self.receive_reward_and_state()

        if self.mode=="metzger" or self.mode=="metzgeradaptedtozahra":            
            self.state = self._create_state_metzger()
        elif self.mode=="zahra":
            self.state = self._create_state_zahra()
        elif self.mode=="mahmoud":
            self.state = self.state = self._create_state_mahmoud_v1(self.action)
        else:
            raise SyntaxError("No Valid mode")

        

        return self.state

    def _create_state(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.lower, self.upper])
        
        return self.state

    def _create_state_metzger(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.reliability, self.deviation])
        return self.state

    def _create_state_zahra(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.lower_cate, self.upper_cate, self.cop])
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
        # importance:  lower_cte, upper_cte, self.cop, self.tu,
        # urgency: self.lower_ciw, self.upper_ciw,
        # capacity: self.nr_ongoing_cases/self.num_cases, self.arrival_time,
        
        if self.component=="all":
            self.state = np.array([self.ordinal_case_id, relative_position, len(self.nr_res)/self.resources, 
                #self.lower_cate, self.upper_cate,
                lower_cte, upper_cte,
                self.cop, self.tu,
                self.lower_ciw, self.upper_ciw,
                self.nr_ongoing_cases/self.num_cases, self.arrival_time])
            #pass
        elif self.component=="withoutTU":
            self.state = np.array([self.ordinal_case_id, relative_position, len(self.nr_res)/self.resources,
                #self.lower_cate, self.upper_cate,
                lower_cte, upper_cte,
                self.cop, #self.tu,
                self.lower_ciw, self.upper_ciw,
                self.nr_ongoing_cases/self.num_cases, self.arrival_time])
            #pass
        elif self.component=="withoutCIW":
            self.state = np.array([self.ordinal_case_id, relative_position, len(self.nr_res)/self.resources,
                #self.lower_cate, self.upper_cate,
                lower_cte, upper_cte,
                self.cop, self.tu,
                #self.lower_ciw, self.upper_ciw,
                self.nr_ongoing_cases/self.num_cases, self.arrival_time])
            pass
        elif self.component=="withCATE":
            self.state = np.array([self.ordinal_case_id, relative_position, len(self.nr_res)/self.resources,
                self.lower_cate, self.upper_cate,
                #lower_cte, upper_cte,
                self.cop, self.tu,
                self.lower_ciw, self.upper_ciw,
                self.nr_ongoing_cases/self.num_cases, self.arrival_time])
            pass
        else:
            print("No Vaild Component")


        #self.state = np.array([self.ordinal_case_id, relative_position, len(self.nr_res)/self.resources, #self.lower_cate, self.upper_cate, 
        #lower_cte, upper_cte,
        #self.cop, self.tu,
        #self.lower_ciw, self.upper_ciw,
        #self.nr_ongoing_cases/self.num_cases, self.arrival_time])        
       
        return self.state

    def render(self, mode="human"):
        # we don't need thisdone
        return

    def receive_reward_and_state(self):
        # If the action is zero, then there is no adaptation. Otherwise else.
        adapted = self.recommend_treatment 
        adapted = True if adapted == 1 else False

        # if adapted:
        #     # This means that there a case recived the treatment
        #     self.treat_counter += 1
        #     self.cases_treat_counter[self.case_id]=self.treat_counter
        #     if self.case_id not in self.treated_cases:
        #         self.treated_cases.append(self.case_id)
        #     else:
        #         print(f"Case: {self.case_id} is treated before")
        #         raise SystemExit()


        cost = self.cin # cost of the intervention
        gain = self.gain # the gain of triggering the intervention

        done = self.data.done    
            
        
        event = self.data.get_event_by_event()
        done = True if done == 1 else False
        case_id = event["case_id"].iloc[0] 

        # while case_id in self.treated_cases:
        #     event = self.data.get_event_by_event()
        #     case_id = event["case_id"].iloc[0]
        #     print("+++++++++++++++++++++++++++++++++++++++++++++++++") 
        
            #raise SystemExit()
        # if case_id in self.treated_cases:
        #     event = self.data.get_event_by_event()
        # else:
        #     pass



        orig_activity = event["orig_activity"].iloc[0]

        ordinal_case_id = event["ordinal_case_ids"].iloc[0]

        predicted_proba_0 = event['predicted_proba_0'].iloc[0]
        predicted_proba_0 = float(predicted_proba_0)

        predicted_proba_1 = event['predicted_proba_1'].iloc[0]
        predicted_proba_1 = float(predicted_proba_1)

        #raise SystemExit("DONE")

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
        upper_ite = event["upper_cate"].iloc[0]
        upper_ite = float(upper_ite)
        lower_ite = event["lower_cate"].iloc[0]
        lower_ite = float(lower_ite)
        upper_cte = event["upper_counterfactual"].iloc[0]
        upper_cte = float(upper_cte)
        lower_cte = event["lower_counterfactual"].iloc[0]
        lower_cte = float(lower_cte)
        deviation = event['deviation'].iloc[0]
        deviation = float(deviation)       
        # Total uncertainty
        tu = event["total_uncer"].iloc[0]
        tu = float(tu)
        cop = event["alpha=0.9_encoded"].iloc[0]
        cop = float(cop)
        time_to_event = event['time_to_event_m'].iloc[0]

        upper_ciw = event["upper_time_to_event_adaptive_QR"].iloc[0]
        upper_ciw = float(upper_ciw)

        lower_ciw = event["lower_time_to_event_adaptive_QR"].iloc[0]
        lower_ciw = float(lower_ciw)

        probaIfTreated = event["Proba_if_Treated"].iloc[0]
        probaIfTreated = float(probaIfTreated)
        probaIfUnTreated = event["Proba_if_Untreated"].iloc[0]
        probaIfUnTreated = float(probaIfUnTreated)

        ite = probaIfTreated - probaIfUnTreated
        ite = float(ite)
        
        # Probability of begative outcome if we intervene or not:
        # y0 = 1 if P(1 | T=0) > 0.5
        # y0 = 1 if P(1 | T=1) > 0.5
        y0 = event["y0"].iloc[0] # predicted outcome if we do not apply the intervention 
        y1 = event["y1"].iloc[0] # predicted outcome if we apply the intervention
        #ite = y1 - y0
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

        

        self.visited_cases[case_id] = [position, process_length, datetime.now()]
        self.nr_ongoing_cases = len(self.visited_cases.keys())
        #print(f"nr_ongoing_cases: {self.nr_ongoing_cases}")

        if position == 1:
            self.visited_cases_arrival_times[case_id] = [position, process_length, datetime.now()]

            if len(self.visited_cases_arrival_times.keys()) >= 2:
                self.arrival_time = (
                    self.visited_cases_arrival_times[list(self.visited_cases_arrival_times.keys())[-1]][-1]
                        - self.visited_cases_arrival_times[list(self.visited_cases_arrival_times.keys())[-2]][-1]
                ).total_seconds()

            else:
                self.arrival_time = 0

        

        if position== process_length:
            self.visited_cases_finishing_times[case_id] = [position, process_length, datetime.now()]
            del self.visited_cases[case_id]           
            if len(self.visited_cases_finishing_times.keys()) >= 2:
                self.finishing_time = (
                    self.visited_cases_finishing_times[list(self.visited_cases_finishing_times.keys())[-1]][-1]
                    - self.visited_cases_finishing_times[list(self.visited_cases_finishing_times.keys())[-2]][-1]
                ).total_seconds()

   
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
            actual_outcome=actual_outcome,
            true_effect=ite,
            nr_res=self.nr_res,
        )
        
        this_event = [
            case_id,
            orig_activity,
            position,
            process_length,
            adapted,
            reward,
            ite, 
            self.treat_counter,
            actual_outcome
            
        ]

        if case_id not in self.ongoing_cases:
            self.ongoing_cases.update({case_id: [this_event]})
        else:
            self.ongoing_cases[case_id].append(this_event)
            self.done = False

        self.predicted_proba_1 = predicted_proba_1
        self.predicted_proba_0 = predicted_proba_0
        self.adapted = adapted
        self.cost = cost
        self.gain = gain
        self.done =  done
        self.case_id = case_id
        self.ordinal_case_id=ordinal_case_id
        
        self.actual_outcome_causal = y1 if self.adapted else y0
        self.actual_outcome_predictive = actual_outcome_predictive
        self.actual_outcome = self.actual_outcome_causal
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

        self.ordinal_case_id = ordinal_case_id

        # if the intervention is triggered in a cases then switch done to True.
        if process_length==position:
            done = True
            self.data.done = 1
            self.done = True
            

        
        if position != 0.0:
            self.position = position
            self.process_length = process_length
        return 0
