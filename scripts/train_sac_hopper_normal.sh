for seed in 1 2 3 4 5
do
python training.py --env-name HopperNormal-v2 --observation-type vector --cuda \
	--cuda-device 0 --num-steps 1000001 --policy sac --buffer-size 1000000 --start-steps 10000 --seed $seed
done
