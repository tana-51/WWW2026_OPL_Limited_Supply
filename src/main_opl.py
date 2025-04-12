"""
実際に、OPLをして実験。
- IPS-PG
- reg-based (true, q_hat, noise)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
# import seaborn as sns
from scipy.stats import rankdata
from sklearn.neural_network import MLPRegressor

import obp
from obp.utils import sigmoid
from obp.dataset import(
    # SyntheticBanditDataset,
    linear_reward_function,
    linear_behavior_policy,
)
from dataset import SyntheticBanditDatasetLimittedSupply

from obp.ope import(
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPS,
    DirectMethod as DM,
    DoublyRobust as DR,
)
from obp.utils import softmax

from agent.agent import(
    PreviousAgent,
    NewAgent,
) 
from policylearner import GradientBasedPolicyLearnerSupply

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.makedirs("output_opl",exist_ok=True)

num_runs = 100
n_actions = 100
n_users = 200
reward_std = 0.1
lambda_ = 0.0
n_step = 1000
max_supply = 10
dataset = SyntheticBanditDatasetLimittedSupply(
    n_actions=n_actions,
    dim_context=10,
    reward_std=reward_std,
    beta=-1.0,
    random_state=12345,
    n_users=n_users,
    lambda_=lambda_, #小さいほど好みが揃う
    n_step=n_step,
    max_supply=max_supply,
)

result_list = []
result_list_ips = []
for noise in [1.0]:
    previous = np.zeros(n_step)
    new = np.zeros(n_step)
    ips_ = np.zeros(n_step)
    previous_regret = np.zeros(n_step)
    new_regret = np.zeros(n_step)
    for _ in tqdm(range(num_runs), desc=f"noise = {noise}"):
        
        bandit_data = dataset.obtain_batch_bandit_feedback()

        if noise=="q_hat":    
            reg_model = RegressionModel(
                n_actions=dataset.n_actions, 
                base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),
            )
            estimated_rewards = reg_model.fit(
                context=bandit_data["context"], # context; x
                action=bandit_data["action"], # action; a
                reward=bandit_data["reward"], # reward; r
            )
            estimated_rewards = reg_model.predict(
                context=bandit_data["fixed_user_context"], # context; x
            )
            q_hat = estimated_rewards[:,:,0]
        elif noise == "true":
            q_hat = bandit_data["fixed_q_x_a"]
        else:
            q_hat = bandit_data["fixed_q_x_a"] + np.random.normal(loc=0.0,scale=noise,size=bandit_data["fixed_q_x_a"].shape)
        fixed_q_x_a = bandit_data["fixed_q_x_a"]

        
        supply_previous = np.random.randint(low=1, high=max_supply, size=n_actions)
        # supply_previous = np.ones(n_action)
        supply_new = supply_previous.copy()
        supply_ips = supply_previous.copy()
        
        x = supply_previous.copy()
        
        #ips
        ips = GradientBasedPolicyLearnerSupply(dim_context = dataset.dim_context+dataset.n_actions, n_action=dataset.n_actions, epoch = 30)
        ips.fit(dataset=bandit_data, dataset_test=bandit_data, q_hat=q_hat[bandit_data["user_idx"]])
        ips_agent_revenue = 0
        ips_agent_revenue_list = []
        
        previous_agent = PreviousAgent()
        previous_agent.set_regret(q_hat)
        previous_agent_revenue = 0
        previous_agent_revenue_list = []
        n_select_arm_previous = np.zeros(n_actions)
        arm_reward_previous = np.zeros(n_actions)
        regret_sum_list_previous = [0]
        
        new_agent = NewAgent()
        new_agent.set_regret(q_hat)
        new_agent_revenue = 0
        new_agent_revenue_list = []
        n_select_arm_new = np.zeros(n_actions)
        arm_reward_new = np.zeros(n_actions)
        regret_sum_list_new = [0]

        
        for i in range(n_step):
            user_idx = np.random.randint(low=0, high=n_users,)
            #previous
            arm_previous, regret_value_previous = previous_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_previous)
            if (supply_previous>=1).sum() ==0:
                click_previous = 0
                r_previous = 0
            else:
                click_previous = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_previous])
                n_select_arm_previous[arm_previous] += 1*click_previous
                r_previous = np.random.normal(loc=fixed_q_x_a[user_idx, arm_previous], scale=3.0)*click_previous
                arm_reward_previous[arm_previous] += r_previous
            previous_agent_revenue += r_previous
            previous_agent_revenue_list.append(previous_agent_revenue)
            supply_previous[arm_previous] -= click_previous
            regret_sum_list_previous.append(regret_sum_list_previous[i-1]+regret_value_previous)

            #new
            arm_new, regret_value_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_new)
            if (supply_new>=1).sum() ==0:
                click_new = 0
                r_new = 0
            else:
                click_new = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_new])
                n_select_arm_new[arm_new] += 1*click_new
                r_new = np.random.normal(loc=fixed_q_x_a[user_idx, arm_new], scale=3.0)*click_new
                arm_reward_new[arm_new] += r_new
            new_agent_revenue += r_new
            new_agent_revenue_list.append(new_agent_revenue)
            supply_new[arm_new] -= click_new
            regret_sum_list_new.append(regret_sum_list_new[i-1]+regret_value_new)
            
            #ips
            pi_ips = ips.predict(np.concatenate([bandit_data["fixed_user_context"][user_idx],supply_ips],axis=0).reshape(1,-1))
            if (supply_ips>=1).sum() ==0:
                click_ips = 0
                r_ips = 0
            else:
                pi_ips[0,supply_ips<=0] = -1000000
                pi_ips = softmax(pi_ips)
                arm_ips = np.random.choice(np.arange(dataset.n_actions),p=pi_ips[0])
                click_ips = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_new])
                r_ips = np.random.normal(loc=fixed_q_x_a[user_idx, arm_ips], scale=3.0)*click_ips
            ips_agent_revenue += r_ips
            ips_agent_revenue_list.append(ips_agent_revenue)
            supply_ips[arm_ips] -= click_ips
        # arm_reward_previous /= n_select_arm_previous
        # arm_reward_new /= n_select_arm_new
        previous += np.array(previous_agent_revenue_list)
        new += np.array(new_agent_revenue_list)
        ips_ += np.array(ips_agent_revenue_list)
        previous_regret += np.array(regret_sum_list_previous[1:])
        new_regret += np.array(regret_sum_list_new[1:])
    result_list.append((new/num_runs)/(previous/num_runs))
    result_list_ips.append((ips_/num_runs)/(previous/num_runs))
    
    plt.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}, $\sigma$={noise}, model")
    plt.plot((ips_/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}, $\sigma$={noise}, pg")
    # plt.plot(new/num_runs, label="regret_based")
plt.legend()

plt.xlabel("Time Step",fontsize=12)
plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
plt.title(f"n_users = {n_users}, n_actions = {n_actions}")
plt.hlines(1.0, 0, n_step, color="black", linestyles='dashed')
# plt.savefig("output_opl/model_vs_ips.png")
plt.show()
