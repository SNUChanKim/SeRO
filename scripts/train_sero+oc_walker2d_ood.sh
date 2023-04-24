for seed in 6 7 8 9 10
do
	python retraining_with_own_criteria.py --env-name Walker2dOOD-v2 --observation-type vector --cuda \
		--cuda-device 0 --num-steps 1000001 --policy sero --use-aux-reward --aux-coef 1 --env-coef 1 --uncertainty-type scalar \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol_coef 0.3
done
