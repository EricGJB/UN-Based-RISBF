import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Lambda,Dropout,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def dtypeconvert(test_list):
    result_list = np.zeros(test_list.shape)+1j*np.zeros(test_list.shape)
    if len(test_list.shape)==2:
        for i in range(test_list.shape[0]):
            for j in range(test_list.shape[1]):
                result_list[i,j] = test_list[i,j][0]+1j*test_list[i,j][1]
    elif len(test_list.shape)==3:
        for i in range(test_list.shape[0]):
            for j in range(test_list.shape[1]):
                for k in range(test_list.shape[2]):
                    result_list[i,j,k] = test_list[i,j,k][0]+1j*test_list[i,j,k][1]
    return result_list

def Q2theta(Q_test,N):
    num = int(Q_test.shape[-1]/(N+1))
    theta_test = np.zeros((num,N))+1j*np.zeros((num,N))
    for i in range(len(theta_test)):
        Q = Q_test[:,(N+1)*i:(N+1)*(i+1)]
        b = np.linalg.eig(Q)
        SIGMA_root = np.diag(np.sqrt(b[0]))
        U = b[1]
        r = 1/np.sqrt(2)*(np.random.randn(N+1,1)+1j*np.random.randn(N+1,1))
        theta = (U.dot(SIGMA_root)).dot(r)
        theta = theta[:N]/theta[-1]
        # element wise normalization
        theta = np.squeeze(theta) / np.linalg.norm(theta,axis=-1)
        theta_test[i] = theta
    return theta_test

def optimal(Ghr_0_test,hd_test,N):
    Ghr_0_test = np.squeeze(Ghr_0_test)
    hd_test = np.squeeze(hd_test)
    num = len(Ghr_0_test)
    theta_test = np.zeros((num,N))+1j*np.zeros((num,N))
    for i in range(num):
        for j in range(N):
            # 将各项的方向都旋转至与hd相同
            item = hd_test[i]/Ghr_0_test[i,j]
            item = item/np.linalg.norm(item)
            theta_test[i,j] = item
    return theta_test

def mse_fitter(M,N,lr):
    merged_inputs = Input(shape=(2*(N*M+N+M),)) 
    temp = Dense(32*N,activation='relu')(merged_inputs)
#    temp = Dropout(rate=0.1)(temp)
    temp = Dense(16*N,activation='relu')(temp) 
#    temp = Dropout(rate=0.1)(temp)
    temp = Dense(8*N,activation='relu')(temp) 
#    temp = Dropout(rate=0.1)(temp)
    temp = Dense(4*N,activation='relu')(temp) 
    out_phase = Dense(N, activation='linear')(temp)
    model = Model(inputs=merged_inputs, outputs=out_phase)
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()
    return model

def beamformer_LIS(M,N,lr):
    def minus_mean_reward(y_true,y_pred):
        phase = y_pred  
        # convert to complex form
        phase = tf.cast(phase,tf.complex128)
        theta_real = tf.cos(phase)
        theta_imag = tf.sin(phase)
        theta = theta_real+1j*theta_imag
        # 根据神经网络预测得到的θ（y_pred），以及实际的参数（y_true），估算期望值
        G = y_true[:,:2*N*M]
        G = tf.cast(G,tf.complex128)
        G = G[:,:N*M]+1j*G[:,N*M:]
        G = tf.reshape(G,[-1,M,N])
        hr = y_true[:,2*N*M:2*(N*M+N)]
        hr = tf.cast(hr,tf.complex128)
        hr = hr[:,:N]+1j*hr[:,N:]
        # 维度扩展
        hr = tf.expand_dims(hr,-1)
        hd = y_true[:,2*(N*M+N):2*(N*M+N+M)]
        hd = tf.cast(hd,tf.complex128)
        hd = hd[:,:M]+1j*hd[:,M:]
        # 维度扩展
        hd = tf.expand_dims(hd,-1)
  
        reward_vector = tf.matmul(tf.matmul(G,tf.linalg.diag(theta)),hr) + hd
        # 去除多余的维度1
        reward_vector = reward_vector[:,:,0] 
        reward = tf.norm(reward_vector,axis=-1)**2
        reward = tf.cast(reward,tf.float32)
        
        # loss为负的reward
        loss = -reward
        return loss
    
    #输入包括G矩阵、hr向量和hd向量的实部与虚部
    merged_inputs = Input(shape=(2*(N*M+M),)) 
#    merged_inputs = BatchNormalization()(merged_inputs)
    temp = Dense(32*N,activation='relu')(merged_inputs)
    temp = BatchNormalization()(temp)
    temp = Dense(16*N,activation='relu')(temp) 
    temp = BatchNormalization()(temp)
    temp = Dense(8*N,activation='relu')(temp) 
    temp = BatchNormalization()(temp)
    temp = Dense(4*N,activation='relu')(temp) 
    temp = BatchNormalization()(temp)
    # output phase of θ
    out_phase = Dense(N, activation='linear')(temp)
    # 截断输出范围,发现性能略略提升了一点（0.01）
#    out_phase = Lambda(lambda x:tf.maximum(tf.minimum(x,np.pi),-np.pi))(out_phase)
    model = Model(inputs=merged_inputs, outputs=out_phase)
    model.compile(loss=minus_mean_reward, optimizer=Adam(lr=lr))
    model.summary()
    return model
