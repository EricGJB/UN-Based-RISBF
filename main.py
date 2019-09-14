import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from utils import Q2theta,beamformer_LIS,dtypeconvert
import time

#%% system parameters
M = 8
N = 16
lr = 0.001
epochs = 1000
batch_size = 5000

#%% training
total_num = 1000000
G_0 = np.sqrt(1/2)*(np.random.randn(total_num,M,N)+1j*np.random.randn(total_num,M,N))
G = np.reshape(G_0,(total_num,-1))
hd = np.sqrt(1/2)*(np.random.randn(total_num,M)+1j*np.random.randn(total_num,M))
# 这样操作更快
hr_0 = np.sqrt(1/2)*(np.random.randn(total_num,N)+1j*np.random.randn(total_num,N))
hr = np.expand_dims(hr_0,axis=1)
hr = np.tile(hr,(1,M,1))

origin_dataset = np.concatenate((np.real(G),np.imag(G),np.real(hr_0),np.imag(hr_0),np.real(hd),np.imag(hd)),axis=-1)
# 融合G和hr
Ghr_0 = G_0*hr
Ghr = np.reshape(Ghr_0,(total_num,-1))

train_dataset = np.concatenate((np.real(Ghr),np.imag(Ghr),np.real(hd),np.imag(hd)),axis=-1)

train_labelset = origin_dataset

best_model_path = './models/best_%d_and_%d.h5'%(M,N)
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)#True for Lambda layer usage
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=20)

model = beamformer_LIS(M,N,lr)
model.fit(train_dataset,train_labelset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                       validation_split=0.2,callbacks=[checkpointer,early_stopping,reduce_lr])

#%% testing 
# test data loading for matlab 2018
import h5py
test_dataset = h5py.File('./data/test_dataset_%d_and_%d.mat'%(M,N),'r')
test_num = len(np.transpose(test_dataset['hd_list']))
Q_test = dtypeconvert(np.transpose(test_dataset['Q_list']))
G_0_test = dtypeconvert(np.transpose(test_dataset['G_list']))
hd_test = dtypeconvert(np.transpose(test_dataset['hd_list']))
hr_0_test = dtypeconvert(np.transpose(test_dataset['hr_list']))
hr_test = np.expand_dims(hr_0_test,axis=1)
hr_test = np.tile(hr_test,(1,M,1))
cvx_theta_test = Q2theta(Q_test,N)

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

predicted_theta = np.cos(predicted_phase)+1j*np.sin(predicted_phase)
random_phase = np.random.uniform(-np.pi,np.pi,predicted_phase.shape)
random_theta = np.cos(random_phase)+1j*np.sin(random_phase)

test_reward_list_cvx = []
test_reward_list_un = []
test_reward_list_random = []
for j in range(test_num):
    value_cvx = (G_0_test[j].dot(np.diag(cvx_theta_test[j]))).dot(hr_0_test[j]) + hd_test[j]
    test_reward_list_cvx.append(np.linalg.norm(value_cvx)**2)  
    value_un =  (G_0_test[j].dot(np.diag(predicted_theta[j]))).dot(hr_0_test[j]) + hd_test[j]
    test_reward_list_un.append(np.linalg.norm(value_un)**2)  
    value_random = (G_0_test[j].dot(np.diag(random_theta[j]))).dot(hr_0_test[j]) + hd_test[j]
    test_reward_list_random.append(np.linalg.norm(value_random)**2)  
print("CVX method:%.2f"%np.mean(test_reward_list_cvx))
print("Pure unsupervised learning:%.2f"%np.mean(test_reward_list_un))
print("Random phase:%.2f"%np.mean(test_reward_list_random))

#%%
# test data loading for matlab 2019
#from scipy import io
#test_dataset = io.loadmat('./data/test_dataset_%d_and_%d.mat'%(M,N))
#hd_test = test_dataset['hd_list']
#hr_test = test_dataset['hr_list']
#test_num = len(hd_test)
#Q_test = test_dataset['Q_list']
#G_0_test = test_dataset['G_list']
#cvx_theta_test = Q2theta(Q_test,N)