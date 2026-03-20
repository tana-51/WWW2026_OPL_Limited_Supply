# lambda
# uv run python ./src/main_lambda.py \
#   setting.run_file="main_lambda" \
#   setting.max_supply=20 \
#   setting.n_step=2500


# heatmap
# uv run python ./src/main_heatmap.py \
#   setting.run_file="main_heatmap"


# user_action_ratio
# uv run python ./src/main_user_action_ratio.py \
#   setting.run_file="main_user_action_ratio" \
#   setting.max_supply=20 \
#   setting.n_step=2500 \
#   setting.user_action_ratio.lambda_=0.5


# noise
# uv run python ./src/main_estimation_noise.py \
#   setting.run_file="main_estimation_noise" \
#   setting.max_supply=20 \
#   setting.n_step=2500 \
#   setting.noise.lambda_=0.5 \
#   setting.noise.noise_list="[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]"


# s_max
# uv run python ./src/main_s_max.py \
#   setting.run_file="main_s_max_naive" \
#   setting.n_step=1000

# uv run python ./src/main_s_max_ours.py \
#   setting.run_file="main_s_max_ours" \
#   setting.n_step=1000