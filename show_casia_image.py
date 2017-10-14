import numpy as np
import tensorflow as tf
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='standard')
args = parser.parse_args()


if args.dataset == 'standard':
    data = np.load('/home/danyang/mfs/data/hccr/image_1000x200x64x64_stand.npy')
elif args.dataset == 'casia-online':
    data = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-online.npy')
elif args.dataset == 'casia-offline':
    data = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-offline.npy')
    data = 1.0 - data
    print np.max(data) , np.min(data)
    #np.save('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-offline-reverse.npy' , data)
elif args.dataset == 'sogou':
    data = np.load('/home/danyang/mfs/data/hccr/image_500x1000x64x64_sogou.npy')
else:
    print 'dataset error!'

n_y = np.shape(data)[0]
handwritten_index = [3 , 9 , 10 , 20 , 194 , 195 , 196 , 197 , 198 , 193]
standard_index = [0 , 1 , 5 , 13 , 18 , 31 , 39 , 44 , 182 , 168]
sample_num = np.shape(data)[1]


char_num = 100
font_num = 500
display = np.reshape(data[:char_num , :font_num  , : , :] , (-1 , 64 , 64 , 1))
#np.random.shuffle(display)
utils.save_image_collections(display, './result/%s.png' % args.dataset, shape=(char_num, 300),scale_each=True)