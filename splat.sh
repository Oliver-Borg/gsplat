export CUDA_VISIBLE_DEVICES=2

choices=("vggt") # "vggt" 
# nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
nums=(50)
# nums=(100)
dataset="bonsai"
seeds=(42)
factor=2
conf_thres_values=(0.0)
num_points_values=(10000 20000 30000 50000 75000 100000 200000 500000 1000 5000)
sampling_modes=("random" "voxels")  # "confidence"
# conf_thres_values=(1.0 1.5 2.0)

for seed in "${seeds[@]}"; do
    for choice in "${choices[@]}"; do
        for num in "${nums[@]}"; do
            for conf_thres_value in "${conf_thres_values[@]}"; do
                for num_points_value in "${num_points_values[@]}"; do
                    for sampling_mode in "${sampling_modes[@]}"; do
                        NAME=${dataset}_${factor}_n100_s${seed}_c${conf_thres_value}_p${num_points_value}_${sampling_mode}
                        RESULT_DIR=./results/${choice}_outputs/${NAME}_i${num}
                        STATS_DIR=${RESULT_DIR}/stats/val_step29999.json
                        DATADIR=../vggt/${choice}_outputs/$NAME
                        # if [ ! -d "$STATS_DIR" ]; then
                        # echo "$STATS_DIR not found. Running splatting"
                        echo "Using data from: $DATADIR"
                        echo "Result dir: $RESULT_DIR"
                        python examples/simple_trainer.py mcmc \
                            --data_dir $DATADIR\
                            --data_factor 1 \
                            --result-dir $RESULT_DIR --disable_viewer \
                            --max_train_cameras $num \
                            --max_steps 7000
                        # fi
                    done
                done
            done
        done
    done
done
