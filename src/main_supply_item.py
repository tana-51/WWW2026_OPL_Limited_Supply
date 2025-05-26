"""
良いアイテムの在庫を変化させる
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
    SyntheticBanditDataset,
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

    n_users = cfg.setting.n_users
    n_action = cfg.setting.n_action
    lambda_ = cfg.setting.supply_item.lambda_

    result_list = []
    regret_list = []

    gamma_list = cfg.setting.supply_item.gamma_list
    supply_type = cfg.setting.supply_item.supply_type

    last_value_list = []
    # for supply_type in supply_type_list:

    result_list = []
    regret_list = []

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(7,7),tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    for gamma in gamma_list:
        
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

        previous = np.zeros(n_step)
        new = np.zeros(n_step)
        previous_regret = np.zeros(n_step)
        new_regret = np.zeros(n_step)
        arm_reward_previous = np.zeros(n_action)
        arm_reward_new = np.zeros(n_action)
        for _ in tqdm(range(num_runs), desc=f"supply_type = {supply_type}, gamma = {gamma}"):

            fixed_q_x_a, fixed_click, fixed_conversion = dataset.obtain_q_x_a()
            
            supply_previous = gamma*np.sort(np.random.randint(low=1, high=cfg.setting.max_supply, size=n_action)) + (1-gamma)*np.sort(np.random.randint(low=1, high=cfg.setting.max_supply, size=n_action))[::-1]
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
                    click_previous = np.random.binomial(n=1, p=fixed_click[user_idx, arm_previous])
                    n_select_arm_previous[arm_previous] += 1*click_previous
                    r_previous = np.random.normal(loc=fixed_conversion[user_idx, arm_previous], scale=cfg.setting.reward_std)*click_previous
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
                    click_new = np.random.binomial(n=1, p=fixed_click[user_idx, arm_new])
                    n_select_arm_new[arm_new] += 1*click_new
                    r_new = np.random.normal(loc=fixed_conversion[user_idx, arm_new], scale=cfg.setting.reward_std)*click_new
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
        ax.plot((new/num_runs)/(previous/num_runs), label=f"$\gamma$={gamma}")
        # plt.plot(new/num_runs, label="regret_based")

    ax.legend()

    ax.set_xlabel("Time Step",fontsize=12)
    ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
    # plt.title(f"n_users = {n_users}, n_actions = {n_action}")
    plt.title(f"Supply Type: {supply_type}")
    ax.axhline(1.0, 0, n_step, color="black", linestyle='dashed')
    plt.savefig(f"val_gamma_{supply_type}.png")
    plt.close()


    #lambda vs relative-reward
    last_value = []
    for i, lambda_ in enumerate(gamma_list):
        last_value.append(result_list[i][-1])
    
    last_value_list.append(last_value)
        
    # plt.plot(gamma_list, last_value, "-o")
    # plt.xlabel("$\gamma$",fontsize=12)
    # plt.ylabel("Relative Reward (Ours/previous)",fontsize=12)
    # plt.title(f"n_users = {n_users}, n_actions = {n_action}")
    # # plt.savefig("output/lambda_vs_lastvalue.png")
    # plt.show()

    df = DataFrame()
    df["gamma"] = gamma_list
    df[supply_type] = last_value_list[0]
    df.to_csv(f"gamma_vs_lastvalue.csv", index=False)

    fig = plt.figure(figsize=(7,7),tight_layout=True)
    ax = fig.add_subplot(1,1,1)

    last_value = df[supply_type]
    ax.plot(gamma_list, last_value, "-o", label=supply_type)
    ax.set_xlabel("$\gamma$",fontsize=12)
    ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
    ax.legend(fontsize=15)
    plt.title(f"n_users = {n_users}, n_actions = {n_action}")
    plt.savefig("gamma_vs_lastvalue.png")
    plt.show()

if __name__ == "__main__":
    main()