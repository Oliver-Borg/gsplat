export CUDA_VISIBLE_DEVICES=2

choices=("vggt") # "vggt" 
# nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
nums=(30)
# nums=(100)
dataset="lego"
seeds=(42 43 44)  #  
factor=1
conf_thres_values=(0.0)
num_points_values=(1000 5000 10000 20000 30000 50000 75000 100000 200000 500000)
sampling_modes=("voxels" "random" "confidence" "ba")
# conf_thres_values=(1.0 1.5 2.0)

for seed in "${seeds[@]}"; do
    for choice in "${choices[@]}"; do
        for num in "${nums[@]}"; do
            if [ $choice == "colmap" ]; then
                NAME=${dataset}_${factor}_n${num}_s${seed}
                RESULT_DIR=./results/${choice}_outputs/${NAME}_i${num}
                STATS_DIR=${RESULT_DIR}/stats/val_step6999.json
                if [ -f "$STATS_DIR" ]; then
                    continue
                fi
                DATADIR=../vggt/${choice}_outputs/$NAME
                echo "Using data from: $DATADIR"
                echo "Result dir: $RESULT_DIR"
                python examples/simple_trainer.py mcmc \
                    --data_dir $DATADIR\
                    --data_factor 1 \
                    --result-dir $RESULT_DIR --disable_viewer \
                    --max_train_cameras $num \
                    --max_steps 7000
                continue
            fi

            for conf_thres_value in "${conf_thres_values[@]}"; do
                for num_points_value in "${num_points_values[@]}"; do
                    for sampling_mode in "${sampling_modes[@]}"; do
                        if [ $sampling_mode == "ba" ]; then
                            NAME=${dataset}_${factor}_n${num}_s${seed}_${sampling_mode}
                        else
                            NAME=${dataset}_${factor}_n${num}_s${seed}_c${conf_thres_value}_p${num_points_value}_${sampling_mode}
                        fi
                        RESULT_DIR=./results/${choice}_outputs/${NAME}_i${num}
                        STATS_DIR=${RESULT_DIR}/stats/val_step6999.json
                        DATADIR=../vggt/${choice}_outputs/$NAME
                        if [ -f "$STATS_DIR" ]; then
                            echo "$STATS_DIR found. Skipping splatting"
                        else
                            echo "$STATS_DIR not found. Running splatting"
                            echo "Using data from: $DATADIR"
                            echo "Result dir: $RESULT_DIR"
                            python examples/simple_trainer.py mcmc \
                                --data_dir $DATADIR\
                                --data_factor 1 \
                                --result-dir $RESULT_DIR --disable_viewer \
                                --max_train_cameras $num \
                                --max_steps 7000
                        fi
                    done
                done
            done
        done
    done
done
