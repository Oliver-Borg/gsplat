export CUDA_VISIBLE_DEVICES=2

choices=("vggt") # "colmap" 
# nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
nums=(10 20 50 100)
# nums=(20)
dataset="bonsai"
seeds=(42 43 44)
factor=2
conf_thres_values=(2.0 3.0 4.0 5.0)
# conf_thres_values=(1.0 1.5 2.0)

for seed in "${seeds[@]}"; do
    for choice in "${choices[@]}"; do
        for num in "${nums[@]}"; do
            for conf_thres_value in "${conf_thres_values[@]}"; do
                NAME=${dataset}_${factor}_n${num}_s${seed}_c${conf_thres_value}
                RESULT_DIR=./results/${choice}_outputs/$NAME
                STATS_DIR=${RESULT_DIR}/stats/val_step29999.json
                DATADIR=../vggt/${choice}_outputs/$NAME
                # if [ ! -d "$STATS_DIR" ]; then
                # echo "$STATS_DIR not found. Running splatting"
                echo "Using data from $DATADIR"
                python examples/simple_trainer.py mcmc \
                    --data_dir  $DATADIR\
                    --data_factor 1 \
                    --result-dir $RESULT_DIR --disable_viewer
                # fi
            done
        done
    done
done
