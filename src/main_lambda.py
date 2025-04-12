"""
好みの一致度合いを変化させる
- reg-based
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import seaborn as sns
from scipy.stats import rankdata
from sklearn.neural_network import MLPRegressor

import obp
from obp.utils import sigmoid
from obp.dataset import(
    SyntheticBanditDataset,
    linear_reward_function,
    linear_behavior_policy,
)
#from dataset import SyntheticBanditDataset

from obp.ope import(
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)

from agent.agent import(
    PreviousAgent,
    NewAgent,
) 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output",exist_ok=True)

np.random.seed(0)

num_runs = 50
n_step = 1000

n_users = 200
fixed_user_context = np.random.normal(loc=0, scale=3.0, size=(n_users, 10))


n_action = 100
action_context = np.eye(n_action, dtype=int)


fixed_q_x_a_base = linear_reward_function(
                context=fixed_user_context,
                action_context=action_context,
                random_state=0,
            ) 


fixed_q_x_a_base = np.abs(fixed_q_x_a_base) 
fixed_increace = np.sort(np.random.uniform(low=0.0, high=fixed_q_x_a_base.max(), size=(n_users, n_action)), axis=1)
# fixed_increace += np.sort(np.random.uniform(low=0.0, high=fixed_q_x_a.max(), size=(n_users, n_action)), axis=0)/2
result_list = []
regret_list = []

lambda_array = np.array([i for i in range(0,11,2)])/10
for lambda_ in lambda_array:
    fixed_q_x_a = lambda_*fixed_q_x_a_base + (1-lambda_)*fixed_increace
    
    # print("regret",fixed_q_x_a[:,0].max() - fixed_q_x_a[:,0].min())
    
    
    previous = np.zeros(n_step)
    new = np.zeros(n_step)
    previous_regret = np.zeros(n_step)
    new_regret = np.zeros(n_step)
    arm_reward_previous = np.zeros(n_action)
    arm_reward_new = np.zeros(n_action)
    for _ in tqdm(range(num_runs), desc=f"lambda = {lambda_}"):
        
        supply_previous = np.random.randint(low=1, high=10, size=n_action)
        # print(supply_previous)
        # supply_previous = np.ones(n_action)
        supply_new = supply_previous.copy()
        supply_first = supply_previous.copy()
        x = supply_previous.copy()
        
        previous_agent = PreviousAgent()
        previous_agent.set_regret(fixed_q_x_a)
        previous_agent_revenue = 0
        previous_agent_revenue_list = []
        n_select_arm_previous = np.zeros(n_action)
        # arm_reward_previous = np.zeros(n_action)
        regret_sum_list_previous = [0]
        
        new_agent = NewAgent()
        new_agent.set_regret(fixed_q_x_a)
        new_agent_revenue = 0
        new_agent_revenue_list = []
        n_select_arm_new = np.zeros(n_action)
        # arm_reward_new = np.zeros(n_action)
        regret_sum_list_new = [0]
    
        
        for i in range(n_step):
            user_idx = np.random.randint(low=0, high=n_users,)
            arm_previous, regret_value_previous = previous_agent.select_arm(user_idx=user_idx, fixed_q_x_a=fixed_q_x_a, supply=supply_previous)
            if (supply_previous>=1).sum() ==0:
                click_previous = 0
                r_previous = 0
            else:
                click_previous = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_previous])
                n_select_arm_previous[arm_previous] += 1*click_previous
                r_previous = np.random.normal(loc=fixed_q_x_a[user_idx, arm_previous], scale=3.0)*click_previous
                arm_reward_previous[arm_previous] += r_previous / (fixed_q_x_a.max(axis=0)[arm_previous]*supply_first[arm_previous])
            previous_agent_revenue += r_previous
            previous_agent_revenue_list.append(previous_agent_revenue)
            supply_previous[arm_previous] -= click_previous
            regret_sum_list_previous.append(regret_sum_list_previous[i-1]+regret_value_previous)
        
            arm_new, regret_value_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=fixed_q_x_a, supply=supply_new)
            if (supply_new>=1).sum() ==0:
                click_new = 0
                r_new = 0
            else:
                click_new = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_new])
                n_select_arm_new[arm_new] += 1*click_new
                r_new = np.random.normal(loc=fixed_q_x_a[user_idx, arm_new], scale=3.0)*click_new
                arm_reward_new[arm_new] += r_new / (fixed_q_x_a.max(axis=0)[arm_new]*supply_first[arm_new])
            new_agent_revenue += r_new
            new_agent_revenue_list.append(new_agent_revenue)
            supply_new[arm_new] -= click_new
            regret_sum_list_new.append(regret_sum_list_new[i-1]+regret_value_new)
        
        # arm_reward_previous /= n_select_arm_previous
        # arm_reward_new /= n_select_arm_new
    
        previous += np.array(previous_agent_revenue_list)
        new += np.array(new_agent_revenue_list)
        previous_regret += np.array(regret_sum_list_previous[1:])
        new_regret += np.array(regret_sum_list_new[1:])
        
    result_list.append((new/num_runs)/(previous/num_runs))
    regret_list.append([previous_regret/num_runs,new_regret/num_runs])
    arm_reward_previous /= num_runs
    arm_reward_new /= num_runs
    plt.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}")
    # plt.plot(new/num_runs, label="regret_based")
plt.legend()

plt.xlabel("Time Step",fontsize=12)
plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
plt.title(f"n_users = {n_users}, n_actions = {n_action}")
plt.hlines(1.0, 0, n_step, color="black", linestyles='dashed')
# plt.savefig("output/val_lambda.png")
plt.show()

#lambda vs relative-reward
last_value = []
for i, lambda_ in enumerate(lambda_array):
    last_value.append(result_list[i][-1])
    
plt.plot(lambda_array, last_value, "-o")
plt.xlabel("$\lambda$",fontsize=12)
plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
plt.title(f"n_users = {n_users}, n_actions = {n_action}")
# plt.savefig("output/lambda_vs_lastvalue.png")
plt.show()

# plt.bar(np.arange(n_action), arm_reward_previous, align="edge",width=-0.3)
# plt.bar(np.arange(n_action), arm_reward_new, align="edge",width=0.3)
# plt.xlabel("item index",fontsize=12)
# plt.ylabel("reward",fontsize=12)
# plt.legend(["previous","ours"])
# plt.show()

# plt.plot(regret_list[0][0])
# plt.plot(regret_list[0][1])
# plt.xlabel("timestep",fontsize=12)
# plt.ylabel("Regret",fontsize=12)
# plt.title(f"n_users = {n_users}, n_actions = {n_action}")
# plt.legend(["previous","new"])
# plt.show()