import os

if __name__ == '__main__':
   root ='/media/legal/DATA1/REFUGE-challenge/REFUGE-Train/image'


   with open('train.txt', mode='w') as f:

    for name in os.listdir(root):
        line = os.path.join(root, name)
        f.writelines(line)
        f.write('\n')
