import os
def tsne():
    with open('./tsne.sh','w') as fout:
        for file in os.listdir('/home/qir21001/AAAI2022/UCR/'):
            cmd = f'python  -m evaluations.classification_test --dataset {file} --path UCR    --seed 42 --batchsize 16 --cv 1'
            fout.write(cmd+'\n')
        for file in os.listdir('/home/qir21001/AAAI2022/UEA/'):
            cmd = f'python  -m evaluations.classification_test --dataset {file} --path UEA    --seed 42 --batchsize 4 --cv 1'
            fout.write(cmd+'\n')
tsne()
