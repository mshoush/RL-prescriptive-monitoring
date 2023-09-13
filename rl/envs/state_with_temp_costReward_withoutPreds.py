import numpy as np
from gym import spaces
from envs.baselineEnv import BaseEnv
import sys

from .ResourceAllocator import allocateRes, block_and_release_res


class StatewithTempCostRewardwithoutPred(BaseEnv):
    metadata = {"render.modes": ["human"]}
    summary_writer = None

    def __init__(self):
        super().__init__()
        #################################
        # Action and state space for the environment
        #################################

        # initilize the state with none.
        self.state = None

        # Action space: 1 or 0. trigger action or not to trigger action.
        self.action_space = spaces.Discrete(
            2
        )  # set action space to adaption true or false

        # the state consists of three values:
        #   - the position
        #   - lower bound of cate
        #   - upper bound of cate
        # for lower and upper bounds, it range from min to max float 32 value.
        # np.array([relative_position, self.lower, self.upper])
        min_resources = 0
        max_resources = np.iinfo(np.int64).max

        min_wip = 0
        max_wip = np.iinfo(np.int64).max

        # low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min, min_resources, min_wip])
        # high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max, max_resources, max_wip])

        low_array = np.array([0.0, np.finfo(np.float32).min, np.finfo(np.float32).min, min_resources])
        high_array = np.array([1.0, np.finfo(np.float32).max, np.finfo(np.float32).max, max_resources])
        # nr_resources_space = 

        # For metzger: position, reliability, deviation
        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.upper = None
        self.lower = None
        self.reliability = None
        self.deviation = None
        self.timesincecasestart = None
        self.timesincemidnight = None
        self.month = None
        self.weekday = None
        self.hour = None
        self.open_cases = None
        self.action=None
        self.nr_res = None
        #self.resources = None

        self.nr_res = list(range(1, self.resources + 1, 1))

    # TODO - From where the action comes.
    # NOTE - It is used in DummyVecEnv class, in the step_wait function.
    def step(self, action=None):
        if self.data.finished:
            self.close()
            from datetime import datetime
            import time
    
            endTime = datetime.now()
            print(f"\nEnd Time: {endTime}\n")
            raise SystemExit("Out of data!")
        if action is None:
            action = 0


        # send action to the environment
        self.send_action(int(action))
        self.action_value = action 

        # obtain reqard after performing the action
        self.receive_reward_and_state()
        print(f"recieved reward: {self.reward}")
        

        self.do_logging(action)

        info = {}
        #self.state = self._create_state()
        self.state = self._create_state_mahmoud_v1(action)
        # print(f"recieved state: {self.state}")

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
        print(f"NR RESOURCES REWARD: {nr_res}, and {self.state}, and {len(self.nr_res)}")
        #cost_resources
        # gain of correctly allocating resources when needed and effictive
        gain_res = 50
        if adapted:
            if len(self.nr_res)>0:                
                if true_effect > 0: # intervention is effective
                    reward = (gain * true_effect) - cost + gain_res # NetGain
                    self.true = True                
                elif true_effect == 0: # not effictive
                    if actual_outcome == 0: # no needed
                        reward = -1 * cost -gain_res # losing cost of intervention
                    if actual_outcome == 1: # needed but ineffective
                        reward = -1 * cost -gain_res # 
                    self.true = False
                elif true_effect < 0: # 
                    reward = -1 * cost - gain -gain_res
                    self.true = False 
            else:
                if true_effect > 0: # intervention is effective
                    reward = (gain * true_effect) - cost - gain_res # NetGain
                    self.true = True                
                elif true_effect == 0: # not effictive
                    if actual_outcome == 0: # no needed
                        reward = -1 * cost - gain_res # losing cost of intervention
                    if actual_outcome == 1: # needed but ineffective
                        reward = -1 * cost - gain_res # 
                    self.true = False
                elif true_effect < 0: # 
                    reward = -1 * cost - gain + gain_res
                    self.true = False 
        else:
            if len(self.nr_res)>0: 
                if true_effect > 0: # intervention is effective
                    reward = (-1 * gain) - gain_res # NetGain
                    self.true = True                
                elif true_effect == 0: # not effictive
                    if actual_outcome == 0: # no needed
                        reward = gain_res # losing cost of intervention
                    if actual_outcome == 1: # needed but ineffective
                        reward = gain_res # 
                    self.true = False
                elif true_effect < 0: # 
                    reward = gain + gain_res
                    self.true = False 
            else:
                if true_effect > 0: # intervention is effective
                    reward = (-1 * gain) - gain_res # NetGain
                    self.true = True                
                elif true_effect == 0: # not effictive
                    if actual_outcome == 0: # no needed
                        reward = 0 # losing cost of intervention
                    if actual_outcome == 1: # needed but ineffective
                        reward = 0 # 
                    self.true = False
                elif true_effect < 0: # 
                    reward = gain + gain_res
                    self.true = False 

        return reward

    def reset(self):
        # self.send_action(-1)
        self.receive_reward_and_state()

        #self.state = self._create_state()
        #self.state = self._create_state_metzger()
        # print("here reset")
        self.state = self._create_state_mahmoud_v1(self.action)

        # print(self.state)
        # print("HERE DONE")
        #raise SystemExit("DONE")

        return self.state

    def _create_state(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.lower, self.upper])
        
        return self.state

    def _create_state_metzger(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.reliability, self.deviation])

        #print(f"\nself.state: {self.state}")
        return self.state

    def _create_state_mahmoud_v1(self, action):
        wip = self.open_cases # work in progress, or active cases, cases in the queue.
        
        if action==1:
            if self.nr_res:
                # print(self.nr_res)
                # print(self.adapted)
                # print(self.case_id)
                # print(self.ordinal_case_id)
                selceted_res = self.nr_res[0]
                #if self.adapted == 1:
                allocateRes(selceted_res, "fixed", self.nr_res, int(0.1))
                print(f"Allocate the selected resource: {selceted_res} for case: {self.case_id}")
                self.nr_res.remove(self.nr_res[0])
            else:
                #selceted_res = 0
                print("No availble resources")
        else:
            pass
            #print("No adaptation")

        # relative_position = self.position / self.process_length
        # self.state = np.array([relative_position, self.lower, self.upper, len(self.nr_res)])
        
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.reliability, self.deviation, len(self.nr_res)])
        # self.state = np.array([relative_position, self.lower, self.upper, len(self.nr_res)])

        return self.state

    def render(self, mode="human"):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # If the action is zero, then there is no adaptation. Otherwise else.
        adapted = self.recommend_treatment 
        adapted = True if adapted == 1 else False

        if adapted:
            # This means that there a case recived the treatment
            self.treat_counter += 1

        cost = 25 # cost of the intervention
        gain = 50 # the gain of triggering the intervention

        done = self.data.done
        # print(done)
        # print("\n==============================\n")
        
        # print(f"\nself.data.finished: {self.data.done}")
        # print(f"\nself.data.finished: {self.data.finished}")
        # raise SystemError()
        

        event = self.data.get_event()
  
        done = True if done == 1 else False

        case_id = event["case_id"].iloc[0]
        ordinal_case_id = event["ordinal_case_ids"].iloc[0]

        ############# FOR MEtzger
        actual_outcome_predictive = event['actual'].iloc[0]
        actual_outcome_predictive = float(actual_outcome_predictive)
        predicted_outcome = event['predicted'].iloc[0]
        predicted_outcome = float(predicted_outcome)

        reliability = event['reliability'].iloc[0]
        reliability = float(reliability)

        predicted_outcome = event["predicted"].iloc[0]
        predicted_outcome = float(predicted_outcome)


        planned_outcome = 0
        planned_outcome = float(planned_outcome)

        resources = 3
        nr_res = list(range(1, resources+1, 1))

        # event_nr
        position = event["prefix_nr"].iloc[0]
        position = float(position)

        process_length = event["case_length"].iloc[0]
        process_length = float(process_length)

        upper = event["upper_cate"]
        upper = float(upper)

        lower = event["lower_cate"]
        lower = float(lower)

        reliability = event['reliability']
        reliability = float(reliability)

        deviation = event['deviation']
        deviation = float(deviation)

        # timesincecasestart = event["timesincecasestart"].iloc[0]
        # timesincemidnight = event["timesincemidnight"].iloc[0]
        # month = event["month"].iloc[0]
        # weekday = event["weekday"].iloc[0]
        # hour = event["hour"].iloc[0]
        # open_cases = event["open_cases"].iloc[0]

        y0 = event["y0"].iloc[0] # predicted outcome if we do not apply the intervention 
        y1 = event["y1"].iloc[0] # predicted outcome if we apply the intervention
        # ite = event['ite'].iloc[0]
        ite = y1 - y0
        
        # if we apply the intervention
        # this a new reaistic outcome for the agent
        # it's predicted from the causal model but it works as actual outcome for the agent.         
        actual_outcome = y1 if adapted else y0 
        actual_outcome = float(actual_outcome)

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
            nr_res=nr_res,
        )

        self.adapted = adapted
        self.cost = cost
        self.gain = gain
        self.done = done
        self.case_id = case_id
        self.ordinal_case_id=ordinal_case_id
        self.actual_outcome = y1 if self.adapted else y0
        self.predicted_outcome = predicted_outcome
        self.planned_outcome = planned_outcome
        self.reliability = reliability
        self.reward = reward
        self.upper = upper
        self.lower = lower
        # self.timesincecasestart = timesincecasestart
        # self.timesincemidnight = timesincemidnight
        # self.month = month
        # self.weekday = weekday
        # self.hour = hour
        # self.open_cases = open_cases

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
