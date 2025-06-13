#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=1

# Problem size
n_L=5 # default 10
n_N=4 # default 8
n_O=3 # default 4

# Parameters for the training
k_epochs=4
update_timestep=2048
learning_rate=0.0001
entropy_value=0.001
eps_clip=0.1
batch_size=128
latent_dim=128
hidden_layer=2

# Others
plot_training=0
mode=gpu

# Folder to save the trained model
network_arch=hide-$hidden_layer-late-$latent_dim/
result_root=trained-models/ppo/cards/L$n_L/N$n_N/O$n_O/seed-$seed/$network_arch
# save_dir=$result_root/k_epochs-$k_epochs-update_timestep-$update_timestep-batch_size-$batch_size-learning_rate-$learning_rate-entropy_value-$entropy_value-eps_clip-$eps_clip # too long
save_dir=$result_root/k_epoc-$k_epochs-tstep-$update_timestep-bs-$batch_size-lr-$learning_rate-ent_v-$entropy_value-eps-$eps_clip


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python src/problem/cards/main_training_ppo_cards.py \
    --seed $seed \
    --n_L $n_L \
    --n_N $n_N \
    --n_O $n_O \
    --k_epochs $k_epochs \
    --update_timestep $update_timestep \
    --learning_rate $learning_rate \
    --eps_clip $eps_clip \
    --entropy_value $entropy_value \
    --batch_size $batch_size \
    --latent_dim $latent_dim \
    --hidden_layer $hidden_layer \
    --save_dir $save_dir \
    --plot_training $plot_training \
    --mode $mode \
    2>&1 | tee $save_dir/log-training.txt