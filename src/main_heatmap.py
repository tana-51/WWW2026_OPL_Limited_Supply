"""
各stepで、どのアイテムを推薦しているかをヒートマップで描画
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

def plot_heat(step_idx_list, item_for_user, name):
    supply_ratio_list = []
    for step_idx in step_idx_list:
        df_list = []
        supply_ratio_list.append(item_for_user[:,:,step_idx].sum(axis=0))
        # step_idx = 20
        # print(item_for_user[:,:,50].sum(axis=0))
        for i in range(n_action):
            df = DataFrame()
            df["user_idx"] = np.arange(n_users)
            df["item_idx"] = i
            df["value"] = item_for_user[np.arange(n_users),i,step_idx]
            df_list.append(df)

        df = pd.concat(df_list, axis=0)
        df = pd.pivot_table(data=df, values='value', columns='item_idx', index='user_idx', aggfunc="sum")
        fontsize = 20
        plt.figure(figsize=(12, 9))
        sns.heatmap(df, annot=True, fmt='g', cmap='Blues')
        plt.title(f"timestep = {step_idx}",fontsize=fontsize)
        plt.xlabel("item index",fontsize=fontsize)
        plt.ylabel("user index",fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f"output2/{name}/heatmap_{step_idx}.png")
        plt.show()
        
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(n_action),item_for_user[:,:,step_idx].sum(axis=0))
        plt.title(f"timestep = {step_idx}",fontsize=fontsize)
        plt.xlabel("item index",fontsize=fontsize)
        plt.ylabel("recommended ratio",fontsize=fontsize)
        plt.ylim(0.0,1.1)
        plt.xticks(np.arange(n_action),fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f"output2/{name}/recommended_ratio_step{step_idx}.png")
        plt.show()



os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output2",exist_ok=True)
os.makedirs("output2/previous",exist_ok=True)
os.makedirs("output2/new",exist_ok=True)

np.random.seed(0)

num_runs = 1000
n_step = 100

n_users = 5
fixed_user_context = np.random.normal(loc=0, scale=3.0, size=(n_users, 10))


n_action = 10
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

# lambda_array = np.array([i for i in range(0,11,2)])/10
lambda_array = np.array([0])
for lambda_ in lambda_array:
    fixed_q_x_a = lambda_*fixed_q_x_a_base + (1-lambda_)*fixed_increace
    
    # print("fixed_q_x_a",fixed_q_x_a)
    # print("max_user",np.argmax(fixed_q_x_a, axis=0))
    df_q_x_a = pd.DataFrame(data=fixed_q_x_a)
    df_q_x_a.to_csv("output2/q_x_a.csv")
    
    
    previous = np.zeros(n_step)
    new = np.zeros(n_step)
    previous_regret = np.zeros(n_step)
    new_regret = np.zeros(n_step)
    
    item_for_user_previous_ = np.zeros((n_users, n_action, n_step))
    item_for_user_new_ = np.zeros((n_users, n_action, n_step))
    
    for _ in tqdm(range(num_runs), desc=f"lambda = {lambda_}"):
        item_for_user_previous = np.zeros((n_users, n_action, n_step))
        item_for_user_new = np.zeros((n_users, n_action, n_step))
        
        # supply_previous = np.array([5]*n_action)
        supply_previous = np.random.randint(low=1, high=10, size=n_action)
        supply_first = supply_previous.copy()
        supply_new = supply_previous.copy()
        x = supply_previous.copy()
        
        previous_agent = PreviousAgent()
        previous_agent.set_regret(fixed_q_x_a)
        previous_agent_revenue = 0
        previous_agent_revenue_list = []
        n_select_arm_previous = np.zeros(n_action)
        arm_reward_previous = np.zeros(n_action)
        regret_sum_list_previous = [0]
        
        new_agent = NewAgent()
        new_agent.set_regret(fixed_q_x_a)
        new_agent_revenue = 0
        new_agent_revenue_list = []
        n_select_arm_new = np.zeros(n_action)
        arm_reward_new = np.zeros(n_action)
        regret_sum_list_new = [0]
    
        
        for i in range(n_step):
            user_idx = np.random.randint(low=0, high=n_users,)
            arm_previous, regret_value_previous = previous_agent.select_arm(user_idx=user_idx, fixed_q_x_a=fixed_q_x_a, supply=supply_previous)
            if (supply_previous>=1).sum() ==0:
                click_previous = 0
                r_previous = 0
                item_for_user_previous[:,:,i] = item_for_user_previous[:,:,i-1].copy()
            else:
                click_previous = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_previous])
                n_select_arm_previous[arm_previous] += 1*click_previous
                r_previous = np.random.normal(loc=fixed_q_x_a[user_idx, arm_previous], scale=3.0)*click_previous
                arm_reward_previous[arm_previous] += r_previous
                if i==0:
                    item_for_user_previous[user_idx,arm_previous,i] += 1/(supply_first[arm_previous]*num_runs)
                else:
                    item_for_user_previous[:,:,i] = item_for_user_previous[:,:,i-1].copy()
                    item_for_user_previous[user_idx,arm_previous,i] += 1/(supply_first[arm_previous]*num_runs)
            previous_agent_revenue += r_previous
            previous_agent_revenue_list.append(previous_agent_revenue)
            supply_previous[arm_previous] -= click_previous
            regret_sum_list_previous.append(regret_sum_list_previous[i-1]+regret_value_previous)
        
            arm_new, regret_value_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=fixed_q_x_a, supply=supply_new)
            if (supply_new>=1).sum() ==0:
                click_new = 0
                r_new = 0
                item_for_user_new[:,:,i] = item_for_user_new[:,:,i-1].copy()
            else:
                click_new = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_new])
                n_select_arm_new[arm_new] += 1*click_new
                r_new = np.random.normal(loc=fixed_q_x_a[user_idx, arm_new], scale=3.0)*click_new
                arm_reward_new[arm_new] += r_new
                if i ==0: 
                    item_for_user_new[user_idx,arm_new,i] += 1/(supply_first[arm_new]*num_runs)
                else:
                    item_for_user_new[:,:,i] = item_for_user_new[:,:,i-1].copy()
                    item_for_user_new[user_idx,arm_new,i] += 1/(supply_first[arm_new]*num_runs)
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
        item_for_user_previous_ += item_for_user_previous
        item_for_user_new_ += item_for_user_new
    result_list.append((new/num_runs)/(previous/num_runs))
    regret_list.append([previous_regret/num_runs,new_regret/num_runs])
    plt.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}")
    # plt.plot(new/num_runs, label="regret_based")
plt.legend()

plt.xlabel("Time Step",fontsize=12)
plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
plt.title(f"n_users = {n_users}, n_actions = {n_action}")
plt.hlines(1.0, 0, n_step, color="black", linestyles='dashed')
plt.savefig("output2/val_lambda.png")
plt.show()

# #lambda-relative reward
# last_value = []
# for i, lambda_ in enumerate(lambda_array):
#     last_value.append(result_list[i][-1])
    
# plt.plot(lambda_array, last_value, "-o")
# plt.xlabel("$\lambda$",fontsize=12)
# plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
# plt.title(f"n_users = {n_users}, n_actions = {n_action}")
# # plt.savefig("output2/lambda_vs_lastvalue.png")
# plt.show()

step_idx_list = [5,10,20,30,40,50,60,70]

plot_heat(step_idx_list, item_for_user_previous_, "previous")
plot_heat(step_idx_list, item_for_user_new_, "new")
