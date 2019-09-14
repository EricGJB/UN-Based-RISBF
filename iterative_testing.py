import numpy as np
from utils import Q2theta,beamformer_LIS,dtypeconvert
import h5py
from scipy import io

#%% system parameters
M_list = [2,4,8]
N_list = [8,16,32,64,128]
lr = 0.001
epochs = 1000
batch_size = 5000

#%% testing 
# test data loading for matlab 2018
for M in M_list:
    CVX_results = []
    Random_results = []
    UN_results = []
    for N in N_list:
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
        cvx_theta_test = Q2theta(Q_test,N)
        
        # 融合G和hr
        Ghr_0_test = G_0_test*hr_test
        Ghr_test = np.reshape(Ghr_0_test,(test_num,-1))
        
        test_dataset = np.concatenate((np.real(Ghr_test),np.imag(Ghr_test),np.real(hd_test),np.imag(hd_test)),axis=-1)
        
        # load the final best model
        model.load_weights(best_model_path)

        predicted_phase = model.predict(test_dataset,verbose=1)
        
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
        print("Unsupervised learning:%.2f"%np.mean(test_reward_list_un))
        print("Random phase:%.2f"%np.mean(test_reward_list_random))
        CVX_results.append(np.mean(test_reward_list_cvx))
        UN_results.append(np.mean(test_reward_list_un))
        Random_results.append(np.mean(test_reward_list_random))
    
    io.savemat('./mat/M=%d.mat'%M,{'CVX_results':CVX_results,'UN_results':UN_results,'Random_results':Random_results})
        
        