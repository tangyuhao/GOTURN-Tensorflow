import pickle
import numpy as np
f = open('pkl/goturn_weights.pkl', 'rb')
pretrained_weights = pickle.load(f,encoding='latin1')
f.close()
right_order_weights = pretrained_weights.copy()
for key in pretrained_weights.keys():
    if 'conv' in key:
        right_order_weights[key]['weights'] = np.transpose(pretrained_weights[key]['weights'],(2,3,1,0))# right order~
    else:
        right_order_weights[key]['weights'] = np.transpose(pretrained_weights[key]['weights'],(1,0))# right order~

f = open('pkl/right_order_goturn_weights.pkl', 'wb')
pickle.dump(right_order_weights, f)
f.close()
