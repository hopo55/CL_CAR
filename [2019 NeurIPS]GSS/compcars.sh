results="./results/compcars"
COMPCARS="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --batch_size 10 --log_every 10 --samples_per_task 25000 --data_file compcars.pt --tasks_to_preserve 10 --cuda yes "
# 전체 데이터 수에 따라 samples_per_task 조정 필요 할듯

mkdir $results
MY_PYTHON="python"
cd data/

$MY_PYTHON compcars.py \
	--o compcars.pt \
	--seed 0 \
	--n_tasks 10

cd ..

seed=0

# echo "***********************iCaRL***********************"
# $MY_PYTHON main.py $COMPCARS --model icarl --lr 0.1 --n_memories 1000  --n_iter 3 --memory_strength 1 --seed $seed

# echo "***********************GEM***********************"
# $MY_PYTHON main.py $COMPCARS --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed $seed

echo "***********************GSS_Greedy***********************"
CUDA_VISIBLE_DEVICES=3 $MY_PYTHON main.py $COMPCARS --model GSS_Greedy --lr 0.01 --n_memories 10 --n_sampled_memories 5000 --n_constraints 10 --memory_strength 10 --n_iter 10 --change_th 0. --seed $seed --subselect 1