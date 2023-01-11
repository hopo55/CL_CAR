results="./results/cars196"
CARS196="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --batch_size 10 --log_every 10 --samples_per_task 10000 --data_file cars196.pt --tasks_to_preserve 5 --cuda yes "

mkdir $results
MY_PYTHON="python"
cd data/

$MY_PYTHON cars196.py \
	--o cars196.pt \
	--seed 0 \
	--n_tasks 5

cd ..

seed=0

# echo "***********************iCaRL***********************"
# $MY_PYTHON main.py $CARS196 --model icarl --lr 0.1 --n_memories 1000  --n_iter 3 --memory_strength 1 --seed $seed

# echo "***********************GEM***********************"
# $MY_PYTHON main.py $CARS196 --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed $seed

echo "***********************GSS_Greedy***********************"
$MY_PYTHON main.py $CARS196 --model GSS_Greedy --lr 0.01 --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10 --n_iter 10 --change_th 0. --seed $seed --subselect 1