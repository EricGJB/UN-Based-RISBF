import numpy as np
from utils import beamformer_LIS,optimal
import time
from scipy import io

#%% system parameters
M = 1
N_list = [8,16,32,64,128]
lr = 0.001
epochs = 1000
batch_size = 5000

#%% testing 
# for matlab 2018
test_num = 30000
Optimal_results = []
Random_results = []
UN_results = []
for N in N_list:
    model = beamformer_LIS(M,N,lr)
    best_model_path = './models/best_%d_and_%d.h5'%(M,N)
    G_0_test = np.sqrt(1/2)*(np.random.randn(test_num,M,N)+1j*np.random.randn(test_num,M,N))
    hd_test = np.sqrt(1/2)*(np.random.randn(test_num,M)+1j*np.random.randn(test_num,M))
    hr_0_test = np.sqrt(1/2)*(np.random.randn(test_num,N)+1j*np.random.randn(test_num,N))
    hr_test = np.expand_dims(hr_0_test,axis=1)
    hr_test = np.tile(hr_test,(1,M,1))
    
    # 融合G和hr
    # 融合G和hr
    Ghr_0_test = G_0_test*hr_test
    Ghr_test = np.reshape(Ghr_0_test,(test_num,-1))
                
    cvx_theta_test = optimal(Ghr_0_test,hd_test,N)
    
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
    print("Optimal method:%.2f"%np.mean(test_reward_list_cvx))
    print("Unsupervised learning:%.2f"%np.mean(test_reward_list_un))
    print("Random phase:%.2f"%np.mean(test_reward_list_random))
    Optimal_results.append(np.mean(test_reward_list_cvx))
    UN_results.append(np.mean(test_reward_list_un))
    Random_results.append(np.mean(test_reward_list_random))

io.savemat('./mat/M=%d.mat'%M,{'Optimal_results':Optimal_results,'UN_results':UN_results,'Random_results':Random_results})
    
    