import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from utils import beamformer_LIS,big_scale,optimal
from sklearn import preprocessing
import time
from scipy import io

#%% system parameters
M = 1
N = 8
lr = 0.001
epochs = 1000
batch_size = 5000

#%% training
total_num = 500000

# path_loss for channel between BS and LIS is fixed
d0 = np.random.uniform(0,8,total_num)
d1 = np.random.uniform(1,6,total_num)        
dBU = np.sqrt(d0**2+d1**2)
dLU = np.sqrt((8-d0)**2+d1**2)
big_scale_BU = big_scale(dBU,total_num)
big_scale_LU = big_scale(dLU,total_num)
# fixed distance between BS and LIS
big_scale_BL = big_scale([8],total_num)

G_0 = np.sqrt(1/2)*(np.random.randn(total_num,M,N)+1j*np.random.randn(total_num,M,N))
G_0 = G_0 * np.expand_dims(big_scale_BL,axis=-1)
G = np.reshape(G_0,(total_num,-1))

hd = np.sqrt(1/2)*(np.random.randn(total_num,M)+1j*np.random.randn(total_num,M))
hd = hd * big_scale_BU

hr_0 = np.sqrt(1/2)*(np.random.randn(total_num,N)+1j*np.random.randn(total_num,N))
hr_0 = hr_0 * big_scale_LU
hr = np.expand_dims(hr_0,axis=1)
hr = np.tile(hr,(1,M,1))

origin_dataset = np.concatenate((np.real(G),np.imag(G),np.real(hr_0),np.imag(hr_0),np.real(hd),np.imag(hd)),axis=-1)

# 融合G和hr
Ghr_0 = G_0*hr
Ghr = np.reshape(Ghr_0,(total_num,-1))

train_dataset = np.concatenate((np.real(Ghr),np.imag(Ghr),np.real(hd),np.imag(hd)),axis=-1)
scaler = preprocessing.StandardScaler().fit(train_dataset)
train_dataset = scaler.transform(train_dataset)

best_model_path = './models/bigscale/best_%d_and_%d.h5'%(M,N)
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)#True for Lambda layer usage
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=20)

model = beamformer_LIS(M,N,lr)
model.fit(train_dataset,origin_dataset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                       validation_split=0.2,callbacks=[checkpointer,early_stopping,reduce_lr])

#%% testing 
test_num = 10000

d0_test = np.random.uniform(0,8,test_num)
d1_test = np.random.uniform(1,6,test_num)        
dBU_test = np.sqrt(d0_test**2+d1_test**2)
dLU_test = np.sqrt((8-d0_test)**2+d1_test**2)
big_scale_BU_test = big_scale(dBU_test,test_num)
big_scale_LU_test = big_scale(dLU_test,test_num)
# fixed distance between BS and LIS
big_scale_BL_test = big_scale([8],test_num)

G_0_test = np.sqrt(1/2)*(np.random.randn(test_num,M,N)+1j*np.random.randn(test_num,M,N))
G_0_test = G_0_test * np.expand_dims(big_scale_BL_test,axis=-1)

hd_test = np.sqrt(1/2)*(np.random.randn(test_num,M)+1j*np.random.randn(test_num,M))
hd_test = hd_test * big_scale_BU_test

hr_0_test = np.sqrt(1/2)*(np.random.randn(test_num,N)+1j*np.random.randn(test_num,N))
hr_0_test = hr_0_test * big_scale_LU_test
hr_test = np.expand_dims(hr_0_test,axis=1)
hr_test = np.tile(hr_test,(1,M,1))

# 融合G和hr
Ghr_0_test = G_0_test*hr_test
Ghr_test = np.reshape(Ghr_0_test,(test_num,-1))

# optimal solution can be solved for single antenna case
cvx_theta_test = optimal(Ghr_0_test,hd_test,N)

test_dataset = np.concatenate((np.real(Ghr_test),np.imag(Ghr_test),np.real(hd_test),np.imag(hd_test)),axis=-1)
test_dataset = scaler.transform(test_dataset)
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
cvx_performance = np.mean(test_reward_list_cvx)
un_performance = np.mean(test_reward_list_un)
random_performance = np.mean(test_reward_list_random)
print("Optimal method:%.4f"%cvx_performance)
print("Unsupervised learning:%.4f"%un_performance)
print("Random phase:%.4f"%random_performance)
io.savemat('./mat/bigscale/M=%dN=%d.mat'%(M,N),{'cvx_performance':cvx_performance,\
                                          'un_performance':un_performance,\
                                          'random_performance':random_performance})