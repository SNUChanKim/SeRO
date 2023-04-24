for seed in 1 2 3 4 5
do
python retraining_with_env_reward.py --env-name HopperOOD-v2 --observation-type vector --cuda \
	--cuda-device 0 --num-steps 1000001 --policy sac --buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol_coef 0
done
