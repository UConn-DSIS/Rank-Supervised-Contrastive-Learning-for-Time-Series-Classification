
python Rank_loss_norm_model.py --path UEA --cuda  --device 1 --model FCN --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0005 --epochs_up 300 --epochs_down 100 --loss rank --dataset SpokenArabicDigits --batchsiz 4
python Rank_loss_norm_model.py --path UEA --cuda  --device 1 --model FCN --distance EU --aug_positives 5 --seed 1 --lr 0.0001 --weight_decay 0.0005 --epochs_up 300 --epochs_down 100 --loss rank --dataset SpokenArabicDigits --batchsiz 4
python Rank_loss_norm_model.py --path UEA --cuda  --device 1 --model FCN --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0005 --epochs_up 300 --epochs_down 100 --loss rank --dataset UWaveGestureLibrary --batchsiz 4
python Rank_loss_norm_model.py --path UEA --cuda  --device 1 --model FCN --distance EU --aug_positives 5 --seed 0 --lr 0.0001 --weight_decay 0.0005 --epochs_up 300 --epochs_down 100 --loss rank --dataset InsectWingbeat --batchsiz 4
