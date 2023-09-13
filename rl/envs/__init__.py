# from RL.envs.master_state_master_reward import MasterStateMasterReward
# from RL.envs.state_with_te_intervals import StatewithTEinterval
# from RL.envs.state_baseline_adapted import StateBaselineAdapted
# from RL.envs.state_with_temporal_features import StatewithTemporalFeatures
# from RL.envs.state_with_temp_costReward import StatewithTempCostReward
# from RL.envs.state_costReward_multTreat import StateCostRewardMultTreat
from envs.state_with_temp_costReward_withoutPreds import StatewithTempCostRewardwithoutPred
from envs.test_new_reward_mahmoud_v1 import StatewithTempCostRewardwithoutPred_newReward
from envs.test_new_reward_mahmoud_v2 import StatewithTempCostRewardwithoutPred_v2_newReward
#from envs.state_with_temp_costReward_conformal import StatewithTempCostRewardConformal
# from gym.envs.registration import register

# register(
#     id='state-with-temporal-cost-reward-withoutpred-v0',
#     entry_point='envs:StatewithTempCostRewardwithoutPred',
# )
# register(
#     id='state-with-temporal-cost-reward-conformal-v0',
#     entry_point='envs:StatewithTempCostRewardConformal',
# )

from gym.envs.registration import register

# register(
#     id="threshold-intra_process-action0-v0",
#     entry_point="envs:ActionOEnv",
# )
# register(
#     id="threshold-intra_process-action1-v0",
#     entry_point="envs:Action1Env",
# )
# register(
#     id="master-state-master-reward-v0",
#     entry_point="envs:MasterStateMasterReward",
# )
# register(
#     id="state-with-TE-interval-v0",
#     entry_point="envs:StatewithTEinterval",
# )
# register(
#     id="state-baseline-adapted-v0",
#     entry_point="envs:StateBaselineAdapted",
# )
# register(
#     id="state-with-temporal-features-v0",
#     entry_point="envs:StatewithTemporalFeatures",
# # )
# register(
#     id="state-with-temporal-cost-reward-v0",
#     entry_point="envs:StatewithTempCostReward",
# )
# register(
#     id="state-cost-reward-mult-treat-v0",
#     entry_point="envs:StateCostRewardMultTreat",
# )
register(
    id="state-with-temporal-cost-reward-withoutpred-v0",
    entry_point="envs:StatewithTempCostRewardwithoutPred",
)
register(
    id="state-with-temporal-cost-reward-conformal-v0",
    entry_point="envs:StatewithTempCostRewardConformal",
)

register(
    id="state-with-temporal-cost-reward-conformal-newReward-v0",
    entry_point="envs:StatewithTempCostRewardwithoutPred_newReward",
)

register(
    id="state-with-temporal-cost-reward-conformal-newReward-v1",
    entry_point="envs:StatewithTempCostRewardwithoutPred_v2_newReward",
)