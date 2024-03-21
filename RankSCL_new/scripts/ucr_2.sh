

python Rank_loss_norm_model.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset Symbols --batchsize 4
python Rank_loss_norm_model.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset Symbols --batchsize 8
python Rank_loss_norm_model.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset Symbols --batchsize 16
python Rank_loss_norm_model.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset Symbols --batchsize 32

