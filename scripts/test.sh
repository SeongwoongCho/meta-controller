cmd="python main.py --stage 2 --config_path configs/test_config.yaml --exp_name MetaController --exp_subname $1 ${@:2:$#}"

echo $cmd
eval $cmd
