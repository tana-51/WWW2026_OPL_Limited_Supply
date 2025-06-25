"""
OPLをして実験。
- reg-based (true, q_hat, noise)
q_hatのノイズを変化
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_random_state

from dataset import RealBanditDatasetLimittedSupply

from obp.ope import(
    RegressionModel,
)

from agent.agent import(
    PreviousAgent,
    NewAgent,
)

from dataset import obtain_supply


num_runs = 100
n_users = 1411
n_actions = 1000
reward_std = 1.0
beta = -0.5
max_supply = 10
n_step = (n_actions*max_supply)+1
random_state = 12345
np.random.seed(random_state)
random_ = check_random_state(random_state)
supply_type_list = ["random", "supply_demand_law", "inverse"]
noise_list = ["true",0.5, 1.0, 3.0, 5.0, "estimate"]

all_result_list = []
for supply_type in supply_type_list:

    result_list = []
    for noise in noise_list:
    
        dataset = RealBanditDatasetLimittedSupply( 
        n_actions=n_actions,
        dim_context=10,
        reward_std=reward_std,
        beta=beta,
        random_state=ramdom_state,
        n_users=n_users,
        n_step=n_step,
        max_supply=max_supply,
        )
        
        previous = np.zeros(n_step)
        new = np.zeros(n_step)
        previous_regret = np.zeros(n_step)
        new_regret = np.zeros(n_step)
        for _ in tqdm(range(num_runs), desc=f"supply_type = {supply_type}"):
            
            bandit_data = dataset.obtain_batch_bandit_feedback(supply_type)
    
            if noise=="estimate":
                reg_model = RegressionModel(
                    n_actions=dataset.n_actions, 
                    base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),#max_iter変えてもいいかも
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
        
            supply_previous = obtain_supply(n_action=n_actions, fixed_q_x_a=fixed_q_x_a, max_supply=max_supply, supply_type=supply_type)
            supply_new = supply_previous.copy()

            #previouse
            previous_agent = PreviousAgent()
            previous_agent.set_regret(q_hat)
            previous_agent_revenue = 0
            previous_agent_revenue_list = []

            #new
            new_agent = NewAgent()
            new_agent.set_regret(q_hat)
            new_agent_revenue = 0
            new_agent_revenue_list = []
        
            
            for i in range(n_step):
                user_idx = np.random.randint(low=0, high=n_users,)
                
                #previous
                arm_previous, regret_value_previous = previous_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_previous)
                if (supply_previous>=1).sum() ==0:
                    click_previous = 0
                    r_previous = 0
                else:
                    click_previous = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_previous])
                    r_previous = dataset.sample_reward_given_expected_reward(fixed_q_x_a[user_idx, arm_previous])*click_previous
                    #r_previous = np.random.normal(loc=fixed_q_x_a[user_idx, arm_previous], scale=3.0)*click_previous
                previous_agent_revenue += r_previous
                previous_agent_revenue_list.append(previous_agent_revenue)
                supply_previous[arm_previous] -= click_previous
        
                #new
                arm_new, regret_value_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_new)
                if (supply_new>=1).sum() ==0:
                    click_new = 0
                    r_new = 0
                else:
                    click_new = 1 #np.random.binomial(n=1, p=click_prob[user_idx, arm_new])
                    r_new = dataset.sample_reward_given_expected_reward(fixed_q_x_a[user_idx, arm_new])*click_new
                    #r_new = np.random.normal(loc=fixed_q_x_a[user_idx, arm_new], scale=3.0)*click_new
                new_agent_revenue += r_new
                new_agent_revenue_list.append(new_agent_revenue)
                supply_new[arm_new] -= click_new
    
                if (supply_previous>=1).sum() == 0 and (supply_new>=1).sum() == 0:
                   break
                    
            paded_previous_agent_revenue_list = np.pad(previous_agent_revenue_list, (0, max(0, n_step - len(previous_agent_revenue_list))), mode='edge')
            paded_new_agent_revenue_list = np.pad(new_agent_revenue_list, (0, max(0, n_step - len(new_agent_revenue_list))), mode='edge')
            previous += np.array(paded_previous_agent_revenue_list)
            new += np.array(paded_new_agent_revenue_list)
        result_list.append((new/num_runs)/(previous/num_runs))

    all_result_list.append(result_list)

records = []

for i, supply_type in enumerate(supply_type_list):
    for j, noise in enumerate(noise_list):
        for step, value in enumerate(all_result_list[i][j]):
            records.append({
                'supply_type': supply_type,
                'noise': str(noise),
                'step': step,
                'value': value
            })

df = pd.DataFrame(records)

df.to_csv('estimation_noise.csv', index=False)
