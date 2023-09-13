import pandas as pd
import sys
import os
from sys import argv

# case_id_col = "Case ID"
# directory = "/home/mshoush/3rdyear/2nd/code/WhenToTreat/"
# filename = "results_adaptive_counterfacs_12.csv"

case_id_col = "case_id"
# directory = "/home/mshoush/3rdyear/2nd/code/myCode/results/compiled_results_for_RL/bpic2012/"
# filename = "ready_to_use_adaptive_bpic2012.csv"

case_id_col = "case_id"
# timestamp_col = "time:timestamp"
timestamp_col = "orig_timestamp"

class envManager():



    def __init__(self):        

        self.current_case_id = 0
        self.finished = False
        self.nr_cases = 0
        self.treated_cases = []

        self.index = 0
        self.done = 0

        self.open_cases = {}
        self.finished_cases = {}
        
        self.mode = argv[1]
        #self.dataset_name = argv[2]  
            
        self.results_dir = argv[2]
        #print(self.mode)

        # Intervention Resource Pool  
        self.resources= int(argv[3])

        # Treatment duration 
        self.tdur= float(argv[4])

        # negative outcome cost (cn) 
        self.cn= int(argv[5])

        # Intervention cost (cin) 
        self.cin= int(argv[6])

        # gain for resources
        self.gain_res = int(argv[7])

        # gain 
        self.gain = int(argv[8])

        self.dataset_name = argv[9]

        self.component = argv[10]

        self.case_id_col = "case_id"
        self.directory = "/home/centos/phd/3rdyear/2nd/myCode/results/compiled_results_for_RL/%s/"%self.dataset_name
        self.filename = "ready_to_use_adaptive_%s.csv"%self.dataset_name



        #self.results_dir=None 
        self.make_results_dir()       
        
        self.load_cases() # read csv file
        self.sort_events()
        self.get_num_of_cases()
        #print(self.get_num_of_cases().num_cases)

        # self.get_num_of_cases() # get the nnumber of unique cases.
        # self.get_cases_list() # get list of unique cases. 

    def make_results_dir(self):

        if not os.path.exists(os.path.join(self.results_dir)):
            os.makedirs(os.path.join(self.results_dir))  
            

    def load_cases(self):
        self.df = pd.read_csv(self.directory+self.filename, sep=';')
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df.sort_values(by=timestamp_col, inplace=True)

    def sort_events(self):
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df.sort_values(by=timestamp_col, inplace=True)

    def get_num_of_cases(self):
        self.num_cases = self.df[case_id_col].nunique()
        return self.num_cases

        


    def get_event_by_event(self):
        if self.index != self.df.shape[0]:            
            self.current_event = self.df.iloc[[self.index]]
            print(f"Observed Event: {self.index}")
            print(f"Len of all events: {self.df.shape[0]}")
            # print(self.treated_cases)
            # if len(self.treated_cases)==1:
            #     raise SystemExit("DONE")
            self.index += 1
            self.case_id = self.current_event[case_id_col].iloc[0]
            self.ordinal_case_id = self.current_event["ordinal_case_ids"].iloc[0]

            print(f"Case: {self.case_id}, ordinal_case_id: {self.ordinal_case_id }")
            #raise SystemError("DONE")
            if self.current_event["prefix_nr"].iloc[0] == self.current_event["case_length"].iloc[0]:
                #print(f"Case: {self.case_id} finished")
                self.finished_cases[self.case_id] = [
                    self.current_event["prefix_nr"].iloc[0],
                    self.current_event["case_length"].iloc[0],
                ]                
                self.done = 1
                #self.finished = False                
            else:
                self.open_cases[self.case_id] = [
                    self.current_event["prefix_nr"].iloc[0],
                    self.current_event["case_length"].iloc[0],
                ]
                #print(f"Case: {self.case_id} ongoing")
                self.done = 0
                #self.finished = False

        else:
            self.done = 1
            self.finished = True
        
        return self.current_event


    


    # def get_num_of_cases(self):
    #     self.num_cases = self.df[case_id_col].nunique()

    # def get_cases_list(self):
    #     self.list_cases = list(self.df[case_id_col].unique())

    # def get_new_event(self):        
    #     pass

    

    # # get a complete trace using the caseid. case by case 
    # def get_new_case(self):
    #     if len(self.list_cases) != 0:
    #         self.current_case_id = self.list_cases.pop(0)            
    #         self.current_df = self.df.loc[self.df[case_id_col]==self.current_case_id]
    #         self.current_df.sort_values(by='prefix_nr', inplace=True)
    #         self.done = 0
    #         self.index = 0
    #     else:       
    #         # finish when there is no more events related to any of the cases.      
    #         self.finished = True
    #         # sys.exit()

    # from the selected case, get event by event. row by row. 
    # def get_event(self):        
    #     if self.index < self.current_df.shape[0]:
    #         # current row 
    #         self.current_event = self.current_df.iloc[[self.index]] 
            
    #         self.index += 1
    #         if self.index == self.current_df.shape[0] - 1:
    #             self.done = 1
    #     else:
    #         self.done = 1

    #     return self.current_event


