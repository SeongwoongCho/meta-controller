cmd="python main.py --stage 1 --config_path configs/finetune_config.yaml --exp_name MetaController --env_name $1 ${@:2:$#}" 

echo $cmd
eval $cmd
