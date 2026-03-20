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
from sklearn.linear_model import LogisticRegression

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
    NewAgentStep,
) 



@hydra.main(config_path="../conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.setting.random_state)
    num_runs = cfg.setting.num_runs
    n_users = cfg.setting.n_users

    n_action = cfg.setting.n_action
    lambda_ = cfg.setting.step.lambda_
    supply_type_list = cfg.setting.user_action_ratio.supply_type_list

    s_max_list = cfg.setting.s_max.s_max_list

    n_step = cfg.setting.s_max.n_step

    last_value_list = []
    r_df_list = []
    for supply_type in supply_type_list:
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10,7),tight_layout=True)
        ax = fig.add_subplot(1,1,1)

        result_list = []
        for s_max in s_max_list:
        
            dataset = SyntheticBanditDatasetLimittedSupply(
                n_actions=n_action,
                dim_context=cfg.setting.dim_context,
                reward_std=cfg.setting.reward_std,
                beta=cfg.setting.beta,
                random_state=cfg.setting.random_state,
                n_users=n_users,
                lambda_=lambda_, 
                n_step=n_step,
                max_supply=s_max,
                supply_type=supply_type,
            )


            previous = np.zeros(n_step)
            new = np.zeros(n_step)
            previous_regret = np.zeros(n_step)
            new_regret = np.zeros(n_step)
            for _ in tqdm(range(num_runs), desc=f"supply_type = {supply_type}, s_max = {s_max}"):
                
                bandit_data = dataset.obtain_batch_bandit_feedback()
                
                fixed_q_x_a, fixed_click, fixed_conversion = dataset.obtain_q_x_a()

                # estimate click probability
                reg_model = RegressionModel(
                    n_actions=dataset.n_actions, 
                    base_model=LogisticRegression(max_iter=1000, random_state=12345),
                )
                estimated_rewards = reg_model.fit(
                    context=bandit_data["context"], # context; x
                    action=bandit_data["action"], # action; a
                    reward=bandit_data["click"], # reward; r
                )
                estimated_rewards = reg_model.predict(
                    context=bandit_data["fixed_user_context"], # context; x
                )
                estimated_click_probability = estimated_rewards[:,:,0]
                ######################################
                
                supply_previous = obtain_supply(
                    n_action=n_action,
                    fixed_q_x_a=fixed_q_x_a,
                    max_supply=s_max,
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
                
                new_agent = NewAgentStep()
                if (supply_new>=1).sum() == 0:
                    coef_ = np.ones(n_action) 
                else:
                    coef_ = supply_new - (n_step / (supply_new>=1).sum())*np.average(estimated_click_probability, axis=0)
                    coef_ = (coef_ <= 0).astype(int) 
                new_agent.obtain_opls_value(fixed_q_x_a, coef_=coef_, user_idx=bandit_data["user_idx"])
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
                
                previous += np.array(previous_agent_revenue_list)
                new += np.array(new_agent_revenue_list)
                previous_regret += np.array(regret_sum_list_previous[1:])
                new_regret += np.array(regret_sum_list_new[1:])

                r_df = DataFrame()
                r_df["value"] = [new_agent_revenue_list[-1] / previous_agent_revenue_list[-1]]
                r_df["s_max"] = s_max
                r_df["supply_type"] = supply_type
                r_df_list.append(r_df)

                result_df = pd.concat(r_df_list).reset_index(level=0)
                result_df.to_csv("s_max.csv")
                
            result_list.append((new/num_runs)/(previous/num_runs))
            ax.plot((new/num_runs)/(previous/num_runs), label=f"$\lambda$={lambda_}, s_max={s_max}")
            print(f"supply_new={supply_new.sum()/x.sum()}, supply_previous={supply_previous.sum()/x.sum()}")
            print("-------"*10)
        ax.legend()

        ax.set_xlabel("Time Step",fontsize=12)
        ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
        plt.title(f"Supply Type: {supply_type}")
        ax.axhline(1.0, 0, n_step, color="black", linestyle='dashed')
        plt.savefig(f"val_step_{supply_type}.png")
        plt.close()

        #user-action-ratio vs -relative reward
        last_value = []
        for i, j in enumerate(s_max_list):
            last_value.append(result_list[i][-1])
        
        last_value_list.append(last_value)

    df = DataFrame()
    df["s_max"] = s_max_list
    for i, supply_type in enumerate(supply_type_list):
        df[supply_type] = last_value_list[i]
    df.to_csv(f"s_max_vs_lastvalue.csv", index=False)

    fig = plt.figure(figsize=(10,7),tight_layout=True)
    ax = fig.add_subplot(1,1,1)

    for i, supply_type in enumerate(supply_type_list):
        last_value = df[supply_type]
        ax.plot(s_max_list, last_value, "-o", label=supply_type)
    ax.set_xlabel("$s_{max}$",fontsize=12)
    ax.set_ylabel("Relative Reward (Ours/previous)",fontsize=12)
    ax.legend(fontsize=15)
    plt.savefig("s_max_vs_lastvalue.png")
    plt.show()

    result_df = pd.concat(r_df_list).reset_index(level=0)
    result_df.to_csv("s_max.csv")

if __name__ == "__main__":
    main()
