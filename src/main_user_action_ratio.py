"""
user-action ratioを変化
"""

from omegaconf import DictConfig, OmegaConf
import hydra
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
    # SyntheticBanditDataset,
    linear_reward_function,
    linear_behavior_policy,
)
from dataset import obtain_supply
from dataset import SyntheticBanditDatasetLimittedSupply

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



@hydra.main(config_path="../conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.setting.random_state)
    num_runs = cfg.setting.num_runs
    n_step = cfg.setting.n_step
    n_users_list = cfg.setting.user_action_ratio.n_users_list

    n_action = cfg.setting.n_action
    lambda_ = cfg.setting.user_action_ratio.lambda_
    supply_type_list = cfg.setting.user_action_ratio.supply_type_list

    last_value_list = []
    for supply_type in supply_type_list:
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(7,7),tight_layout=True)
        ax = fig.add_subplot(1,1,1)

        result_list = []
        for n_users in n_users_list:
        
            dataset = SyntheticBanditDatasetLimittedSupply(
                n_actions=n_action,
                dim_context=cfg.setting.dim_context,
                reward_std=cfg.setting.reward_std,
                beta=cfg.setting.beta,
                random_state=cfg.setting.random_state,
                n_users=n_users,
                lambda_=lambda_, #小さいほど好みが揃う
                n_step=n_step,
                max_supply=cfg.setting.max_supply,
                supply_type=supply_type,
            )

            # bandit_data = dataset.obtain_batch_bandit_feedback()

            # if noise=="q_hat":    
            #     reg_model = RegressionModel(
            #         n_actions=dataset.n_actions, 
            #         base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),
            #     )
            #     estimated_rewards = reg_model.fit(
            #         context=bandit_data["context"], # context; x
            #         action=bandit_data["action"], # action; a
            #         reward=bandit_data["reward"], # reward; r
            #     )
            #     estimated_rewards = reg_model.predict(
            #         context=bandit_data["fixed_user_context"], # context; x
            #     )
            #     q_hat = estimated_rewards[:,:,0]
            # elif noise == "true":
            #     q_hat = bandit_data["fixed_q_x_a"]
            # else:
            #     q_hat = bandit_data["fixed_q_x_a"] + np.random.normal(loc=0.0,scale=noise,size=bandit_data["fixed_q_x_a"].shape)
            # fixed_q_x_a = bandit_data["fixed_q_x_a"]

            previous = np.zeros(n_step)
            new = np.zeros(n_step)
            previous_regret = np.zeros(n_step)
            new_regret = np.zeros(n_step)
            for _ in tqdm(range(num_runs), desc=f"supply_type = {supply_type}, n_users = {n_users}"):

                fixed_q_x_a, fixed_click, fixed_conversion = dataset.obtain_q_x_a()
                
                supply_previous = obtain_supply(
                    n_action=n_action,
                    fixed_q_x_a=fixed_q_x_a,
                    max_supply=cfg.setting.max_supply,
                    supply_type=supply_type,
                )

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
                    else:
                        click_previous = np.random.binomial(n=1, p=fixed_click[user_idx, arm_previous])
                        n_select_arm_previous[arm_previous] += 1*click_previous
                        r_previous = np.random.normal(loc=fixed_conversion[user_idx, arm_previous], scale=cfg.setting.reward_std)*click_previous
                        arm_reward_previous[arm_previous] += r_previous
                    previous_agent_revenue += r_previous
                    previous_agent_revenue_list.append(previous_agent_revenue)
                    supply_previous[arm_previous] -= click_previous
                    regret_sum_list_previous.append(regret_sum_list_previous[i-1]+regret_value_previous)
                
                    arm_new, regret_value_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=fixed_q_x_a, supply=supply_new)
                    if (supply_new>=1).sum() ==0:
                        click_new = 0
                        r_new = 0
                    else:
                        click_new = np.random.binomial(n=1, p=fixed_click[user_idx, arm_new])
                        n_select_arm_new[arm_new] += 1*click_new
                        r_new = np.random.normal(loc=fixed_conversion[user_idx, arm_new], scale=cfg.setting.reward_std)*click_new
                        arm_reward_new[arm_new] += r_new
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
            ax.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}, ratio={n_users/n_action}")
            # plt.plot(new/num_runs, label="regret_based")
        ax.legend()

        ax.set_xlabel("Time Step",fontsize=12)
        ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
        # plt.title(f"n_users = {n_users}, n_actions = {n_action}")
        plt.title(f"Supply Type: {supply_type}")
        ax.axhline(1.0, 0, n_step, color="black", linestyle='dashed')
        plt.savefig(f"val_lambda_{supply_type}.png")
        plt.close()

        #user-action-ratio vs -relative reward
        last_value = []
        for i, j in enumerate(n_users_list):
            last_value.append(result_list[i][-1])
        
        last_value_list.append(last_value)
            
    # plt.plot(n_users_array/ n_actions, last_value, "-o")
    # plt.xlabel("n_users / n_actions",fontsize=12)
    # plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
    # plt.title(f"$\lambda$ = {lambda_}, n_actions = {n_actions}")
    # # plt.savefig("output3/user-action-ratio_vs_lastvalue.png")
    # plt.show()

    df = DataFrame()
    df["n_user"] = n_users_list
    for i, supply_type in enumerate(supply_type_list):
        df[supply_type] = last_value_list[i]
    df.to_csv(f"user_action_rartio_vs_lastvalue.csv", index=False)

    fig = plt.figure(figsize=(7,7),tight_layout=True)
    ax = fig.add_subplot(1,1,1)

    for i, supply_type in enumerate(supply_type_list):
        last_value = df[supply_type]
        ax.plot(np.array(n_users_list)/n_action, last_value, "-o", label=supply_type)
    ax.set_xlabel("$user / action ratio$",fontsize=12)
    ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
    ax.legend(fontsize=15)
    # plt.title(f"n_users = {n_users}, n_actions = {n_action}")
    plt.savefig("user_action_rartio_vs_lastvalue.png")
    plt.show()

if __name__ == "__main__":
    main()
