# Off-Policy Learning with Limited Supply
This repository contains the code used for the experiments in ["Off-Policy Learning with Limited Supply"](https://arxiv.org/abs/2603.18702) by Koichi Tanaka, Ren Kishimoto, Bushun Kawagishi, Yusuke Narita, Yasuo Yamamoto, Nobuyuki Shimizu, Yuta Saito. This paper was accepted at [TheWebConf 2026](https://www2026.thewebconf.org).

## Abstract
We study off-policy learning (OPL) in contextual bandits, which plays a key role in a wide range of real-world applications such as recommendation systems and online advertising. Typical OPL in contextual bandits assumes an unconstrained environment where a policy can select the same item infinitely. However, in many practical applications, including coupon allocation and e-commerce, limited supply constrains items through budget limits on distributed coupons or inventory restrictions on products. In these settings, greedily selecting the item with the highest expected reward for the current user may lead to early depletion of that item, making it unavailable for future users who could potentially generate higher expected rewards. As a result, OPL methods that are optimal in unconstrained settings may become suboptimal in limited supply settings. To address the issue, we provide a theoretical analysis showing that conventional greedy OPL approaches may fail to maximize the policy performance, and demonstrate that policies with superior performance must exist in limited supply settings. Based on this insight, we introduce a novel method called Off-Policy learning with Limited Supply (OPLS). Rather than simply selecting the item with the highest expected reward, OPLS focuses on items with relatively higher expected rewards compared to the other users, enabling more efficient allocation of items with limited supply. Our empirical results on both synthetic and real-world datasets show that OPLS outperforms existing OPL methods in contextual bandit problems with limited supply.

## Citation
```
coming soon...
```

## Setup
The Python environment is built using uv. You can build the same environment as in our experiments by cloning the repository and running uv sync directly under the folder.
```
# clone the repository
git clone https://github.com/tana-51/WWW2026_OPL_Limited_Supply.git

# build the environment with uv
uv sync

# activate the environment
source .venv/bin/activate
```

## Runing the code
### Section 5: Synthetic Experiments
The commands needed to reproduce the experiments are summarized below. Please move under the `src` directly first and then run the commands.

```
cd src

# How does OPLS choose actions compared to the conventional greedy method?
uv run python main_heatmap.py setting.run_file="main_heatmap"

# How does OPLS perform varying the action popularity? 
uv run python main_lambda.py setting.run_file="main_lambda"

# How does OPLS perform varying the the number of users?
uv run python main_user_action_ratio.py setting.run_file="main_user_action_ratio"

# How does OPLS perform varying estimation noises?
uv run python main_estimation_noise.py setting.run_file="main_estimation_noise"


# How does OPLS perform varying the max supply s_{max}?
uv run python main_s_max.py setting.run_file="main_s_max_naive" setting.n_step=1000
uv run python main_s_max_ours.py setting.run_file="main_s_max_ours" setting.n_step=1000
```
