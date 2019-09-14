import numpy as np
from utils import beamformer_LIS,dtypeconvert
import time
import h5py

#%% system parameters
M = 2
N = 32
lr = 0.001

#%% testing 
# test data loading for matlab 2018
model = beamformer_LIS(M,N,lr)
best_model_path = './models/best_%d_and_%d.h5'%(M,N)

test_dataset = h5py.File('./data/test_dataset_%d_and_%d.mat'%(M,N),'r')
test_num = len(np.transpose(test_dataset['hd_list']))
Q_test = dtypeconvert(np.transpose(test_dataset['Q_list']))
G_0_test = dtypeconvert(np.transpose(test_dataset['G_list']))
hd_test = dtypeconvert(np.transpose(test_dataset['hd_list']))
hr_0_test = dtypeconvert(np.transpose(test_dataset['hr_list']))
hr_test = np.expand_dims(hr_0_test,axis=1)
hr_test = np.tile(hr_test,(1,M,1))

# 融合G和hr
Ghr_0_test = G_0_test*hr_test
Ghr_test = np.reshape(Ghr_0_test,(test_num,-1))

test_dataset = np.concatenate((np.real(Ghr_test),np.imag(Ghr_test),np.real(hd_test),np.imag(hd_test)),axis=-1)

# load the final best model
model.load_weights(best_model_path)

start = time.time()
predicted_phase = model.predict(test_dataset,verbose=1)
end = time.time()
mean_time = (end-start)/len(test_dataset)
print('Mean predict time of neural network:%.4f ms'%(mean_time*1000))