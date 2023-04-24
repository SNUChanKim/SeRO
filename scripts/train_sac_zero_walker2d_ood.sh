for seed in 6 7 8 9 10
do
python retraining.py --env-name Walker2dOOD-v2 --observation-type vector --cuda \
	--cuda-device 0 --num-steps 1000001 --policy sac --buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol_coef 0
done
