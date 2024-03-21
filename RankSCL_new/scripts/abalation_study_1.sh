python Rank_loss_raw_space.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 0 --seed 1 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset PigCVP --batchsize 8
python Rank_loss_raw_space.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 10 --seed 2 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset PigCVP --batchsize 8
python Rank_loss_raw_space.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 10 --seed 3 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset PigCVP --batchsize 8
python Rank_loss_raw_space.py --path UCR --cuda --device 1 --model FCN  --distance EU --aug_positives 10 --seed 42 --lr 0.0001 --weight_decay 0.0008 --epochs_up 300 --epochs_down 100 --loss rank --dataset PigCVP --batchsize 8

