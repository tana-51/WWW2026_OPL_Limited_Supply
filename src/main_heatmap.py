"""
各stepで、どのアイテムを推薦しているかをヒートマップで描画
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

def plot_heat(step_idx_list, item_for_user, name, n_users, n_action):

    supply_ratio_list = []
    for step_idx in step_idx_list:
        df_list = []
        supply_ratio_list.append(item_for_user[:,:,step_idx].sum(axis=0))
        
        for i in range(n_action):
            df = DataFrame()
            df["user_idx"] = np.arange(n_users)
            df["item_idx"] = i
            df["value"] = item_for_user[np.arange(n_users),i,step_idx]
            df_list.append(df)

        df = pd.concat(df_list, axis=0)
        df = pd.pivot_table(data=df, values='value', columns='item_idx', index='user_idx', aggfunc="sum")
        fontsize = 20
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 9))
        sns.heatmap(df, annot=True, fmt='g', cmap='Blues')
        plt.title(f"timestep = {step_idx}",fontsize=fontsize)
        plt.xlabel("item index",fontsize=fontsize)
        plt.ylabel("user index",fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f"{name}/heatmap_{step_idx}.png")
        plt.close()
        
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(n_action),item_for_user[:,:,step_idx].sum(axis=0))
        plt.title(f"timestep = {step_idx}",fontsize=fontsize)
        plt.xlabel("item index",fontsize=fontsize)
        plt.ylabel("recommended ratio",fontsize=fontsize)
        plt.ylim(0.0,1.1)
        plt.xticks(np.arange(n_action),fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f"{name}/recommended_ratio_step{step_idx}.png")
        plt.close()



@hydra.main(config_path="../conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    os.makedirs("previous", exist_ok=True)
    os.makedirs("new", exist_ok=True)   
    np.random.seed(cfg.setting.random_state)

    num_runs = cfg.setting.num_runs
    n_step = cfg.setting.heatmap.n_step

    n_users = cfg.setting.heatmap.n_users
    n_action = cfg.setting.heatmap.n_action
    lambda_ = cfg.setting.heatmap.lambda_
    supply_type = cfg.setting.heatmap.supply_type

    result_list = []
    regret_list = []

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
    fixed_q_x_a, fixed_click, fixed_conversion = dataset.obtain_q_x_a()
    
    previous = np.zeros(n_step)
    new = np.zeros(n_step)
    previous_regret = np.zeros(n_step)
    new_regret = np.zeros(n_step)
    
    item_for_user_previous_ = np.zeros((n_users, n_action, n_step))
    item_for_user_new_ = np.zeros((n_users, n_action, n_step))
    
    for _ in tqdm(range(num_runs), desc=f"lambda = {lambda_}"):
        item_for_user_previous = np.zeros((n_users, n_action, n_step))
        item_for_user_new = np.zeros((n_users, n_action, n_step))
        

        supply_previous = obtain_supply(
                    n_action=n_action,
                    fixed_q_x_a=fixed_q_x_a,
                    max_supply=cfg.setting.max_supply,
                    supply_type=supply_type,
                )
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
                click_previous = np.random.binomial(n=1, p=fixed_click[user_idx, arm_previous])
                n_select_arm_previous[arm_previous] += 1*click_previous
                r_previous = np.random.normal(loc=fixed_conversion[user_idx, arm_previous], scale=cfg.setting.reward_std)*click_previous
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
                click_new = np.random.binomial(n=1, p=fixed_click[user_idx, arm_new])
                n_select_arm_new[arm_new] += 1*click_new
                r_new = np.random.normal(loc=fixed_conversion[user_idx, arm_new], scale=cfg.setting.reward_std)*click_new
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

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(7,7),tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}")
    # plt.plot(new/num_runs, label="regret_based")
    ax.legend()

    ax.set_xlabel("Time Step",fontsize=12)
    ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
    # plt.title(f"n_users = {n_users}, n_actions = {n_action}")
    plt.title(f"Supply Type: {supply_type}")
    ax.axhline(1.0, 0, n_step, color="black", linestyle='dashed')
    plt.savefig(f"val_lambda_{supply_type}.png")
    plt.close()

    step_idx_list = np.array([5,10,20,30,40,50,60,70,80,90,100,110])

    plot_heat(step_idx_list, item_for_user_previous_, "previous", n_users, n_action)
    plot_heat(step_idx_list, item_for_user_new_, "new", n_users, n_action)

if __name__ == "__main__":
    main()