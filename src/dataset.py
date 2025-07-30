from dataclasses import dataclass
from typing import Callable, Tuple
from typing import Optional

import numpy as np
from scipy.stats import truncnorm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid
from obp.utils import softmax
from obp.utils import sample_action_fast
from obp.dataset.base import BaseBanditDataset
from reward_type import RewardType
from obp.dataset import(
    logistic_reward_function,
    linear_reward_function,
)


@dataclass
class SyntheticBanditDatasetLimittedSupply(BaseBanditDataset):

    n_actions: int
    dim_context: int = 5
    reward_type: str = RewardType.BINARY.value
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    reward_std: float = 1.0
    action_context: Optional[np.ndarray] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    beta: float = 1.0
    random_state: int = 12345
    n_users: int = 100
    lambda_: float = 0.5 #小さいほど好みが揃う
    n_step: int = 1000
    max_supply: int = 10
    supply_type: str = "random"  # "random", "supply_demand_law", "inverse"
    dataset_name: str = "synthetic_bandit_dataset_with_limmited_supply"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(self.beta, "beta", (int, float))

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)

        # one-hot encoding characterizing actions.
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)
        else:
            check_array(
                array=self.action_context, name="action_context", expected_dim=2
            )
            if self.action_context.shape[0] != self.n_actions:
                raise ValueError(
                    "Expected `action_context.shape[0] == n_actions`, but found it False."
                )


    def obtain_batch_bandit_feedback(self, ) -> BanditFeedback:
        
        fixed_user_context = self.random_.normal(size=(self.n_users, self.dim_context))
        # user_idx = self.random_.randint(low=0, high=self.n_users,n=n_rounds)
        # contexts = fixed_user_context[user_idx]

        # obtain expected reward
        fixed_click_base = logistic_reward_function(
            context=fixed_user_context,
            action_context=self.action_context,
            random_state=self.random_state,
        )
        fixed_click_increase = np.sort(
            self.random_.uniform(
                low=0.0, high=fixed_click_base.max(), size=(self.n_users, self.n_actions)
            ),
            axis=1,
        )
        fixed_click = self.lambda_ * fixed_click_base + (1 - self.lambda_) * fixed_click_increase
        fixed_click = sigmoid(fixed_click)

        fixed_conversion_base = linear_reward_function(
                context=fixed_user_context,
                action_context=self.action_context,
                random_state=0,
            ) 
        fixed_conversion_base = np.abs(fixed_conversion_base) 
        fixed_conversion_increace = np.sort(self.random_.uniform(low=0.0, high=fixed_conversion_base.max(), size=(self.n_users, self.n_actions)), axis=1)
        fixed_conversion = self.lambda_*fixed_conversion_base + (1-self.lambda_)*fixed_conversion_increace

        fixed_q_x_a = fixed_click * fixed_conversion
        
        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = fixed_q_x_a.copy()
        else:
            pi_b_logits = self.behavior_policy_function(
                context=fixed_user_context,
                action_context=self.action_context,
                random_state=self.random_state,
            )

        fixed_pi_b = softmax(self.beta * pi_b_logits)
        
        #supply
        supply = obtain_supply(n_action=self.n_actions, fixed_q_x_a=fixed_q_x_a, max_supply=self.max_supply, supply_type=self.supply_type)
        
        # sample actions for each round based on the behavior policy
        unique_action_set = np.arange(self.n_actions)
        user_idx_list = []
        action_list = []
        reward_list = []
        pscore_list = []
        supply_list = []
        click_list = []

        for i in range(self.n_step):
            user_idx = self.random_.randint(low=0, high=self.n_users)
            pi_b_logits_ = self.beta * pi_b_logits[user_idx, unique_action_set]
            pi_b = softmax(pi_b_logits_.reshape(1,-1))[0,:]
            
            action = self.random_.choice(unique_action_set, p=pi_b)
            sampled_action_index = np.where(unique_action_set == action)[0][0]
            click = self.random_.binomial(n=1, p=fixed_click[user_idx, action])

            reward = self.random_.normal(loc=fixed_q_x_a[user_idx,action], scale=self.reward_std)
            reward *= click
            
            user_idx_list.append(user_idx)
            action_list.append(action)
            reward_list.append(reward)
            pscore_list.append(pi_b[sampled_action_index])
            supply_list.append(supply.reshape(1,-1))
            supply[action] -= click
            click_list.append(click)

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
            action_context=self.action_context,
            action=np.array(action_list),
            position=None,  # position effect is not considered in synthetic data
            reward=np.array(reward_list),
            fixed_click=fixed_click,
            fixed_conversion=fixed_conversion,
            fixed_q_x_a=fixed_q_x_a,
            pscore=np.array(pscore_list),
            lambda_= self.lambda_, #小さいほど好みが揃う
            n_step= self.n_step,
            max_supply= self.max_supply,
            supply_each_step=np.concatenate(supply_list,axis=0),
            context_supply=np.concatenate([fixed_user_context[np.array(user_idx_list)],np.concatenate(supply_list,axis=0)],axis=1),
            click=np.array(click_list),
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()
    
    def obtain_q_x_a(self,):
        fixed_user_context = self.random_.normal(size=(self.n_users, self.dim_context))

        # obtain expected reward
        fixed_click_base = logistic_reward_function(
            context=fixed_user_context,
            action_context=self.action_context,
            random_state=self.random_state,
        )
        fixed_click_increase = np.sort(
            self.random_.uniform(
                low=0.0, high=fixed_click_base.max(), size=(self.n_users, self.n_actions)
            ),
            axis=1,
        )
        fixed_click = self.lambda_ * fixed_click_base + (1 - self.lambda_) * fixed_click_increase
        fixed_click = sigmoid(fixed_click)

        fixed_conversion_base = linear_reward_function(
                context=fixed_user_context,
                action_context=self.action_context,
                random_state=0,
            ) 
        fixed_conversion_base = np.abs(fixed_conversion_base) 
        fixed_conversion_increace = np.sort(self.random_.uniform(low=0.0, high=fixed_conversion_base.max(), size=(self.n_users, self.n_actions)), axis=1)
        fixed_conversion = self.lambda_*fixed_conversion_base + (1-self.lambda_)*fixed_conversion_increace

        fixed_q_x_a = fixed_click * fixed_conversion

        return fixed_q_x_a, fixed_click, fixed_conversion


def obtain_supply(n_action, fixed_q_x_a, supply_type, max_supply=10):
    if supply_type == "random":
        supply = np.random.randint(low=1, high=max_supply, size=n_action)
    elif supply_type == "supply_demand_law":
        demand = fixed_q_x_a.mean(axis=0)
        demand = np.clip(demand, 0, None)

        normalized_demand = demand / demand.max()
        supply = (normalized_demand * max_supply).astype(int)

    else:
        demand = fixed_q_x_a.mean(axis=0)
        demand = np.clip(demand, 1e-6, None)

        inverse_demand = 1 / np.sqrt(demand)
        normalized_demand = inverse_demand / inverse_demand.max()
        supply = (normalized_demand * max_supply).astype(int)

    return supply
