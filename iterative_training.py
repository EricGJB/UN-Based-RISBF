import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from utils import beamformer_LIS

#%% system parameters
M_list = [1,2,4,8]
N_list = [8,16,32,64,128]
lr = 0.001
epochs = 1000
batch_size = 5000

#%% training
total_num = 1000000
for M in M_list:
    for N in N_list:
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
        early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.001,patience=20)
        
        model = beamformer_LIS(M,N,lr)
        model.fit(train_dataset,train_labelset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                               validation_split=0.2,callbacks=[checkpointer,early_stopping,reduce_lr])