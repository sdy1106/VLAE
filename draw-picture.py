from matplotlib import pyplot as plt
import re
import numpy as np

methods = ['content' , 'binary' , 'onehot']
characters = ['10' , '100' , '200' , '500']
data = {}
data['content-10'] = [0.0] * 100
data['binary-10'] = [0.0] * 100
data['onehot-10'] = [0.0] * 100
data['content-100'] = [0.0] * 100
data['binary-100'] = [0.0] * 100
data['onehot-100'] = [0.0] * 100
data['content-200'] = [0.0] * 100
data['binary-200'] = [0.0] * 100
data['onehot-200'] = [0.0] * 100
data['content-500'] = [0.0] * 100
data['binary-500'] = [0.0] * 100

#Epoch=87 Iter=80 (2.200s/iter): Lower Bound=-342.894042969 , Tv loss=0.19

pattern = re.compile(r'Bound=-(\d+\.\d*) ,')

for method in methods:
    for char in characters:
        print method , char
        if method == 'onehot' and char == '500':
            continue
        fr = open(method + '-' + char)
        for index , line in enumerate(fr):
            line = line.strip()
            res = pattern.search(line).group(1)
            data[method + '-' + char][index] = float(res)
        fr.close()

x_array = np.arange(1,101)

plt.figure()
for index , char in enumerate(characters):
    p = plt.subplot(221+index)
    for method in methods:
        if method == 'onehot' and char == '500':
            continue
        # mark = ''
        # if method == 'content':
        #     mark = '*'
        # elif method == 'onehot':
        #     mark = '.'
        # else:
        #     mark = '-'
        legend = ''
        if method == 'content':
            legend = 'content'
        elif method == 'onehot':
            legend = 'onehot'
        else:
            legend = 'binary'

        p.plot(x_array, data[method + '-' + char], label=legend)
        p.set_xlabel('epoch')
        p.set_ylabel('lower bound')
        p.set_title('character num = %s' % char)
        p.legend(fontsize=10)

plt.show()



