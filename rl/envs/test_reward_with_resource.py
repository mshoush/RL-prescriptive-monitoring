import numpy as np
from gym import spaces
from envs.baselineEnv import BaseEnv
from .ResourceAllocator import allocateRes


class StatewithTempCostRewardwithoutPred(BaseEnv):
    metadata = {"render.modes": ["human"]}
    summary_writer = None

    def __init__(self):
        super().__init__()
        #################################
        # Parameter fuer das Environment
        #################################
        self.state = None
        self.action_space = spaces.Discrete(
            2
        )  # set action space to adaption true or false
        # Hier dimensionen des state-arrays anpassen:
        #        low_array = np.array([0, 0, 0])
        #        high_array = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])

        # state = rel. position, TE lower bound, TE upper bound, timesincecasestart, timesincemidnight, month,weekday, hour,open_cases
        low_array = np.array(
            [0.0, np.finfo(np.float32).min, np.finfo(np.float32).min]
        )  # , 0,1,1,0,0,0
        high_array = np.array(
            [1.0, np.finfo(np.float32).max, np.finfo(np.float32).max]
        )  # np.finfo(np.float32).max, np.finfo(np.float32).max,12,6,23, np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=low_array, high=high_array)

        self.upper = None
        self.lower = None
        self.timesincecasestart = None
        self.timesincemidnight = None
        self.month = None
        self.weekday = None
        self.hour = None
        self.open_cases = None
        #self.nr_res=None

        self.resources = 3
        self.nr_res = list(range(1, self.resources + 1, 1))
        # print(f"nr of resources: {self.nr_res}")

    def step(self, action=None):
        if self.data.finished:
            self.close()
            raise SystemExit("Out of data!")
        if action is None:
            action = 0

        # if self.nr_res:
        #     selceted_res = self.nr_res[0]
        #     self.nr_res.remove(self.nr_res[0])
        #     allocateRes(selceted_res, "fixed", self.nr_res, int(1))
        #     self.send_action(int(action))  
        #     self.action_value = action     
        #     self.receive_reward_and_state()
        #     print(f"recieved reward: {self.reward}")     
        # else:
        #     print("There is no resource availble")        

        self.send_action(int(action))
        self.action_value = action     
        self.receive_reward_and_state()
        print(f"recieved reward: {self.reward}") 

        self.do_logging(action)

        info = {}
        self.state = self._create_state()

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
        true_effect=0,        
    ):
        # id apply the intervention and there is an avaible resource
        # reward for resource = 10
        res_reward = 10
        if adapted and self.nr_res: # apply the intervention, action=1
            if true_effect > 0:
                reward = (gain * true_effect) - cost + res_reward
                self.true = True
            elif true_effect == 0:
                if actual_outcome == 1:
                    reward = -1 * cost - res_reward
                if actual_outcome == 0:
                    reward = -1 * cost - res_reward
                self.true = False
            elif true_effect < 0:
                reward = -1 * cost - gain - res_reward
                self.true = False

        elif adapted and not self.nr_res: # apply the intervention, action=1
            reward = -1 * cost - gain - res_reward
            self.true = False
            # if true_effect > 0:
            #     reward = (gain * true_effect) - cost + res_reward
            #     self.true = True
            # elif true_effect == 0:
            #     if actual_outcome == 1:
            #         reward = -1 * cost - res_reward
            #     if actual_outcome == 0:
            #         reward = -1 * cost - res_reward
            #     self.true = False
            # elif true_effect < 0:
            #     reward = -1 * cost - gain - res_reward
            #     self.true = False
        else:
            if self.nr_res:
                if true_effect > 0:
                    reward = -gain - res_reward
                    self.true = False
                elif true_effect == 0:
                    if actual_outcome == 1:
                        reward = gain + res_reward
                        self.true = True
                    else:
                        reward = 0 - res_reward
                        self.true = True
                elif true_effect < 0:
                    reward = gain + res_reward
                    self.true = True
            else:
                if true_effect > 0:
                    reward = -gain + res_reward
                    self.true = False
                elif true_effect == 0:
                    if actual_outcome == 1:
                        reward = gain - res_reward
                        self.true = True
                    else:
                        reward = 0 + res_reward
                        self.true = True
                elif true_effect < 0:
                    reward = gain - res_reward
                    self.true = True

            if self.nr_res:
                print(f"nr_res: {self.nr_res}")
                selceted_res = self.nr_res[0]
                self.nr_res.remove(self.nr_res[0])            
                allocateRes(selceted_res, "fixed", self.nr_res, int(1))
                #reward=reward
            else:
                print("No availble resources")
                #reward = -100
                
        return reward

    def reset(self):
        # self.send_action(-1)
        self.receive_reward_and_state()

        self.state = self._create_state()

        return self.state

    def _create_state(self):
        relative_position = self.position / self.process_length
        self.state = np.array([relative_position, self.lower, self.upper])
        return self.state

    def render(self, mode="human"):
        # we don't need this
        return

    def receive_reward_and_state(self):
        # print("receiving reward parameters...")
        nr_res=self.nr_res

        adapted = self.recommend_treatment
        adapted = True if adapted == 1 else False
        if adapted:
            self.treat_counter += 1

        cost = 25
        gain = 50
        done = self.data.done
        event = self.data.get_event()
        done = True if done == 1 else False

        case_id = event["Case ID"].iloc[0]
        # print(self.data)
        # print(event.columns.values.tolist())
        # print(case_id)
        predicted_outcome = event["preds2"].iloc[0]
        predicted_outcome = float(predicted_outcome)
        planned_outcome = 1
        planned_outcome = float(planned_outcome)
        reliability = event["reliability"].iloc[0]
        reliability = float(reliability)
        position = event["event_nr"].iloc[0]
        position = float(position)
        process_length = event["case_length"].iloc[0]
        process_length = float(process_length)

        upper = event["upper"]
        upper = float(upper)
        lower = event["lower"]
        lower = float(lower)

        timesincecasestart = event["timesincecasestart"].iloc[0]
        timesincemidnight = event["timesincemidnight"].iloc[0]
        month = event["month"].iloc[0]
        weekday = event["weekday"].iloc[0]
        hour = event["hour"].iloc[0]
        open_cases = event["open_cases"].iloc[0]

        y0 = event["probability_0"].iloc[0]
        y1 = event["probability_1"].iloc[0]
        # ite = event['ite'].iloc[0]
        ite = y1 - y0

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
            nr_res = nr_res
        )


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
        self.upper = upper
        self.lower = lower
        self.timesincecasestart = timesincecasestart
        self.timesincemidnight = timesincemidnight
        self.month = month
        self.weekday = weekday
        self.hour = hour
        self.open_cases = open_cases
        if adapted:
            done = True
            self.data.done = 1
            self.done = True

        if position != 0.0:
            self.position = position
            self.process_length = process_length

        return 0
