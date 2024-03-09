
python  -m tnc.tnc --dataset FaceDetection --path UEA --seed 1 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset FaceDetection --path UEA --seed 1 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset FaceDetection --path UEA --seed 2 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset FaceDetection --path UEA --seed 2 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset FaceDetection --path UEA --seed 3 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset FaceDetection --path UEA --seed 3 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset FaceDetection --path UEA --seed 42 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset FaceDetection --path UEA --seed 42 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset JapaneseVowels --path UEA --seed 2 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset JapaneseVowels --path UEA --seed 2 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset JapaneseVowels --path UEA --seed 3 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset JapaneseVowels --path UEA --seed 3 --batchsize 4 --cv 1
python  -m tnc.tnc --dataset JapaneseVowels --path UEA --seed 42 --batchsize 4 --cv 1
python  -m evaluations.classification_test --dataset JapaneseVowels --path UEA --seed 42 --batchsize 4 --cv 1

