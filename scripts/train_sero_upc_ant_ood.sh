for seed in 1 2 3 4 5
do
	python retraining.py --env-name AntOOD-v2 --observation-type vector --cuda \
		--cuda-device 0 --num-steps 1000001 --policy sero --use-aux-reward --aux-coef 0 --env-coef 1 --uncertainty-type scalar \
		--buffer-size 1000000 --start-steps 10000 --seed $seed --load-model --consol_coef 0.1
done
