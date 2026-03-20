"""
Vary s_max
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from scipy import stats

from dataset import obtain_supply
from dataset import RealBanditDatasetLimittedSupply

from obp.ope import(
    RegressionModel,
)

from agent.agent import(
    PreviousAgent,
    NewAgentStep,
)

@hydra.main(config_path="./conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

    n_users = cfg.setting.n_users
    n_actions = cfg.setting.n_actions
    n_step = cfg.setting.s_max.n_step
    random_state = cfg.setting.random_state
    np.random.seed(random_state)

    s_max_list = cfg.setting.s_max.s_max_list
    supply_type_list = cfg.setting.supply_type_list
    noise_list = cfg.setting.s_max.noise_list

    all_result_list = []
    for supply_type in supply_type_list:

        result_list_noise = []
        for noise in noise_list:

            result_list_s_max = []
            for s_max in s_max_list:
                ratios_over_runs = []
            
                dataset = RealBanditDatasetLimittedSupply( 
                n_actions=n_actions,
                reward_std=cfg.setting.reward_std,
                beta=cfg.setting.beta,
                random_state=random_state,
                n_users=n_users,
                n_step=n_step,
                supply_type=supply_type,
                max_supply=s_max,
                )
        
                for _ in tqdm(range(cfg.setting.num_runs), desc=f"supply_type = {supply_type}, noise = {noise}, s_max = {s_max}"):
    
                    bandit_data = dataset.obtain_batch_bandit_feedback()
            
                    if noise=="estimate":
                        reg_model = RegressionModel(
                            n_actions=dataset.n_actions, 
                            base_model=MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=3000,early_stopping=True,random_state=12345),
                        )
                        estimated_rewards = reg_model.fit(
                            context=bandit_data["context"], # context: x
                            action=bandit_data["action"], # action: a
                            reward=bandit_data["reward"], # reward: r
                        )
                        estimated_rewards = reg_model.predict(
                            context=bandit_data["fixed_user_context"], # context: x
                        )
                        q_hat = estimated_rewards[:,:,0]
                    elif noise == "true":
                        q_hat = bandit_data["fixed_q_x_a"]
                    else:
                        q_hat = bandit_data["fixed_q_x_a"] + np.random.normal(loc=0.0,scale=noise,size=bandit_data["fixed_q_x_a"].shape)
    
                    fixed_q_x_a = bandit_data["fixed_q_x_a"]

                    # obtain supply
                    supply_previous = obtain_supply(n_action=n_actions, fixed_q_x_a=fixed_q_x_a, max_supply=s_max, supply_type=supply_type)
                    supply_new = supply_previous.copy()

                    # conventional greedy method
                    previous_agent = PreviousAgent()
                    previous_agent_revenue = 0
                    previous_agent_revenue_list = []

                    # OPLS
                    new_agent = NewAgentStep()
                    if (supply_new>=1).sum() == 0:
                        coef_ = np.ones(n_actions)
                    else:
                        coef_ = supply_new - (n_step / (supply_new>=1).sum())*1  # q_c(x,a) = 1
                        coef_ = (coef_ <= 0).astype(int) # 
                    new_agent.obtain_opls_value(q_hat,bandit_data["user_idx"],coef_)
                    new_agent_revenue = 0
                    new_agent_revenue_list = []
    
                    for i in range(n_step):
                        user_idx = np.random.randint(low=0, high=n_users,)

                        # conventional greedy method
                        arm_previous = previous_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_previous)
                        if (supply_previous>=1).sum() ==0:
                            click_previous = 0
                            r_previous = 0
                        else:
                            click_previous = 1  # q_c(x,a) = 1
                            r_previous = dataset.sample_reward_given_expected_reward(fixed_q_x_a[user_idx, arm_previous])*click_previous
                        previous_agent_revenue += r_previous
                        previous_agent_revenue_list.append(previous_agent_revenue)
                        supply_previous[arm_previous] -= click_previous
    
                        # OPLS
                        arm_new = new_agent.select_arm(user_idx=user_idx, fixed_q_x_a=q_hat, supply=supply_new)
                        
                        if (supply_new>=1).sum() == 0:
                            click_new = 0
                            r_new = 0
                        else:
                            click_new = 1  # q_c(x,a) = 1
                            r_new = dataset.sample_reward_given_expected_reward(fixed_q_x_a[user_idx, arm_new])*click_new
                        new_agent_revenue += r_new
                        new_agent_revenue_list.append(new_agent_revenue)
                        supply_new[arm_new] -= click_new
    
                        if (supply_previous>=1).sum() == 0 and (supply_new>=1).sum() == 0:
                            break
                            
                    paded_previous_agent_revenue_list = np.pad(
                        previous_agent_revenue_list, (0, max(0, n_step - len(previous_agent_revenue_list))), mode='edge'
                    )
                    paded_new_agent_revenue_list = np.pad(
                        new_agent_revenue_list, (0, max(0, n_step - len(new_agent_revenue_list))), mode='edge'
                    )
                    
                    ratio = np.array(paded_new_agent_revenue_list) / np.array(paded_previous_agent_revenue_list)
                    ratios_over_runs.append(ratio)
                    
                ratios_over_runs = np.array(ratios_over_runs)
                result_list_s_max.append(ratios_over_runs)
    
            result_list_noise.append(result_list_s_max)
            
        all_result_list.append(result_list_noise)

    ## summarize results
    records = []
    for i, supply_type in enumerate(supply_type_list):
        for j, noise in enumerate(noise_list):
            for k, s_max in enumerate(s_max_list):
                ratio_matrix = all_result_list[i][j][k]
                mean_ratio = np.mean(ratio_matrix, axis=0)
                sem = stats.sem(ratio_matrix, axis=0)
                ci95 = 1.96 * sem
                
                for step in range(n_step):
                    records.append({
                        'supply_type': supply_type,
                        'noise': str(noise),
                        'n_step': n_step,
                        's_max': s_max,
                        'step': step,
                        'mean': mean_ratio[step],
                        'ci95_lower': mean_ratio[step] - ci95[step],
                        'ci95_upper': mean_ratio[step] + ci95[step],
                    })
    
    df = pd.DataFrame(records)
    df.to_csv('main_s_max.csv', index=False)

if __name__ == "__main__":
    main()