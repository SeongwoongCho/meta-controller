cmd="python main.py --stage 0 --config_path configs/train_config.yaml ${@:1:$#}"
echo $cmd
eval $cmd
