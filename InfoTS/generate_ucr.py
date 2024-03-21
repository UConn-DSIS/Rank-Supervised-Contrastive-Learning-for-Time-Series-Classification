import os
def ucr():
    with open('./ucr.sh','w') as fout:
        for file in os.listdir('/home/qir21001/AAAI2022/UCR'):
            cmd = f'python -u train.py {file} --archive UCR --repr-dims 320 --max-threads 8 --seed 42 --eval --aug_p1 0.5 --max-train-length 3000 --iters 200 --supervised_meta --log_file ucr.result'
            fout.write(cmd+'\n')
        #for file in os.listdir('/home/luods/Desktop/InfoTSC/datasets/UEA/'):
         #   cmd = f'python -u train.py {file} --archive UEA --repr-dims 320 --max-threads 8 --seed 42 --eval --max-train-length 3000 --iters 200 --supervised_meta --log_file uea.result'
          #  fout.write(cmd+'\n')

'''
def generate_sota():
    files = []
    with open('./SOTA/uea.txt','w') as fout:
        for file in os.listdir('/home/luods/Desktop/InfoTSC/datasets/UEA/'):
            files.append(file)
            files = sorted(files)
        print(files)
        for f in files:
            fout.write(f+'\n')
    
'''
ucr()
        