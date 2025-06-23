from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from scipy.stats import truncnorm

from obp.dataset.base import BaseBanditDataset
from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import softmax

@dataclass
class RealBanditDatasetLimittedSupply(BaseBanditDataset):

    n_actions: int = 100
    #n_components: int = 10
    dim_context: int = 10
    reward_std: float = 1.0
    beta: float = 1.0
    random_state: int = 12345
    n_users: int = 200
    n_step: int = 1001
    max_supply: int = 10
    #supply_type: str = "random"  # "random", "supply_demand_law", "inverse"
    dataset_name: str = "real_bandit_dataset_with_limmited_supply"
    
    def __post_init__(self):
        self.data_path = Path(r"/home/kawagi/My/my_OPL_with_limited_supply/real/data/kuairec")
        self.pca = PCA(n_components=self.dim_context, random_state=self.random_state)
        self.sc = StandardScaler()

        self.random_ = check_random_state(self.random_state)
        self.interactions, self.contexts = self.pre_process(self.data_path)

    def pre_process(
        self, 
        file_path: Path
    ) -> Tuple[np.array, np.array]:
        """Preprocess raw dataset."""
        df_small_matrix = pd.read_csv(file_path / "small_matrix.csv")
        df_user_feature = pd.read_csv(file_path / "user_features.csv")

        # small_matrix
        df_watch_ratio = df_small_matrix.pivot_table(index="user_id",columns="video_id",values="watch_ratio")
        df_watch_ratio = df_watch_ratio.dropna(axis=1)

        watch = df_watch_ratio.to_numpy()

        # context
        delete_column = df_user_feature.columns.values[df_user_feature.dtypes == object]
        df_user_feature = df_user_feature.drop(delete_column, axis=1)
        df_user_feature = df_user_feature.dropna(axis=1)
        user_ids = df_watch_ratio.index
        df_user_feature = df_user_feature[df_user_feature["user_id"].isin(user_ids)]
        df_user_feature = df_user_feature.iloc[:, 1:]

        context = df_user_feature.to_numpy()

        return watch, context
        
    def obtain_batch_bandit_feedback(self, supply_type: str) -> BanditFeedback:
        """Obtain batch logged bandit data."""
        # user and contexts
        user_idx = self.random_.choice(self.contexts.shape[0], size=self.n_users, replace=False)
        fixed_user_context = self.contexts[user_idx]
        fixed_user_context = self.sc.fit_transform(
            self.pca.fit_transform(fixed_user_context)
        )

        # action
        action_idx = self.random_.choice(self.interactions.shape[1], size=self.n_actions, replace=False)
        
        # expected reward function
        fixed_q_x_a = self.interactions[np.ix_(user_idx,action_idx)]
        pi_b_logits = fixed_q_x_a.copy()

        # calculate the action choice probabilities of the behavior policy
        fixed_pi_b = softmax(self.beta * pi_b_logits)

        # supply
        supply = obtain_supply(n_action=self.n_actions, fixed_q_x_a=fixed_q_x_a, max_supply=self.max_supply, supply_type=supply_type)
        
        # sample actions for each round based on the behavior policy
        unique_action_set = np.arange(self.n_actions)
        user_idx_list = []
        action_list = []
        reward_list = []
        pscore_list = []
        supply_list = []
        for i in range(self.n_step):
            user_idx = self.random_.randint(low=0, high=self.n_users)
            pi_b_logits_ = self.beta * pi_b_logits[user_idx, unique_action_set]
            pi_b = softmax(pi_b_logits_.reshape(1,-1))[0,:]
            
            action = self.random_.choice(unique_action_set, p=pi_b)
            sampled_action_index = np.where(unique_action_set == action)[0][0]
            
            #reward = self.random_.normal(loc=fixed_q_x_a[user_idx,action], scale=self.reward_std)
            reward = self.sample_reward_given_expected_reward(fixed_q_x_a[user_idx,action])
            
            user_idx_list.append(user_idx)
            action_list.append(action)
            reward_list.append(reward)
            pscore_list.append(pi_b[sampled_action_index])
            supply_list.append(supply.reshape(1,-1))
            supply[action] -= 1

             # delete action
            unique_action_set = np.delete(
                np.arange(self.n_actions), supply <=0,
            )
            if (supply>=1).sum()==0:
                break

        return dict(
            n_users=self.n_users,
            user_idx=np.array(user_idx_list),
            n_actions=self.n_actions,
            fixed_user_context=fixed_user_context,
            context=fixed_user_context[np.array(user_idx_list)],
            action=np.array(action_list),
            reward=np.array(reward_list),
            fixed_q_x_a=fixed_q_x_a,
            pscore=np.array(pscore_list),
            n_step= self.n_step,
            max_supply= self.max_supply,
            supply_each_step=np.concatenate(supply_list,axis=0),
            context_supply=np.concatenate([fixed_user_context[np.array(user_idx_list)],np.concatenate(supply_list,axis=0)],axis=1)
        )

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        #action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        #expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        reward_min = 0
        reward_max = 1e3
        mean = expected_reward
        a = (reward_min - mean) / self.reward_std
        b = (reward_max - mean) / self.reward_std
        reward = truncnorm.rvs(
            a=a,
            b=b,
            loc=mean,
            scale=self.reward_std,
            random_state=self.random_state,
        )

        return reward

def obtain_supply(n_action, fixed_q_x_a, supply_type, max_supply=10):
    if supply_type == "random":
        supply = np.random.randint(low=1, high=max_supply, size=n_action)
        
    elif supply_type == "supply_demand_law":
        demand = fixed_q_x_a.mean(axis=0)
        demand = np.clip(demand, 0, None)

        normalized_demand = demand / demand.max()
        supply = (normalized_demand * max_supply).astype(int)
        
    elif supply_type == "inverse":
        demand = fixed_q_x_a.mean(axis=0)
        demand = np.clip(demand, 1e-6, None)

        inverse_demand = 1 / np.sqrt(demand)
        normalized_demand = inverse_demand / inverse_demand.max()
        supply = (normalized_demand * max_supply).astype(int)

    else:
        raise TypeError("Unsupported supply type or format")

    return supply