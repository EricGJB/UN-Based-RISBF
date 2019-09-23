import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import beamformer_LIS, beamformer_LIS_NOBF,big_scale
from scipy import io
from sklearn import preprocessing

#%% system parameters
lr = 0.001
epochs = 30
total_num = 1000000
M = 8
N = 64

#%% training
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

# combination 1
batch_size = 500
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
model = beamformer_LIS(M,N,lr)
history1 = model.fit(train_dataset,origin_dataset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                       validation_split=0.2,callbacks=[reduce_lr])

# combination 2
batch_size = 5000
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
model = beamformer_LIS(M,N,lr)
history2 = model.fit(train_dataset,origin_dataset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                       validation_split=0.2,callbacks=[reduce_lr])

# combination 3
batch_size = 5000
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
model = beamformer_LIS_NOBF(M,N,lr)
history3 = model.fit(train_dataset,origin_dataset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
                       validation_split=0.2,callbacks=[reduce_lr])

train_loss_1 = history1.history['loss']
train_loss_2 = history2.history['loss']
train_loss_3 = history3.history['loss']
val_loss_1 = history1.history['val_loss']
val_loss_2 = history2.history['val_loss']
val_loss_3 = history3.history['val_loss']
io.savemat('./mat/history.mat',{'train_loss_1':train_loss_1,'train_loss_2':train_loss_2,'train_loss_3':train_loss_3,\
                                'val_loss_1':val_loss_1,'val_loss_2':val_loss_2,'val_loss_3':val_loss_3})

#%% history plot
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
history = io.loadmat('./mat/bigscale/history.mat')
train_loss_1 = np.squeeze(history['train_loss_1'])
train_loss_2 = np.squeeze(history['train_loss_2'])
train_loss_3 = np.squeeze(history['train_loss_3'])
val_loss_1 = np.squeeze(history['val_loss_1'])
val_loss_2 = np.squeeze(history['val_loss_2'])
val_loss_3 = np.squeeze(history['val_loss_3'])

epochs = 30
epochs = np.arange(1,1+epochs,1)
ax = plt.subplot(111)
plt.plot(epochs,train_loss_1,'ro-',epochs,train_loss_2,'bo-',epochs,train_loss_3,'go-',\
         epochs,val_loss_1,'r^-',epochs,val_loss_2,'b^-',epochs,val_loss_3,'g^-')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('M=8, N=64')
plt.legend(['val loss, BN, batch size 500','val loss, no BN, batch size 5000',\
             'train loss, BN, batch size 5000','train loss, BN, batch size 500',\
            'train loss, no BN, batch size 5000','val loss, BN, batch size 5000'])

xmajorLocator   = MultipleLocator(4)
xmajorFormatter = FormatStrFormatter('%d') 
xminorLocator   = MultipleLocator(1) 
ymajorLocator   = MultipleLocator(0.5) 
ymajorFormatter = FormatStrFormatter('%.1f') 
yminorLocator   = MultipleLocator(0.1)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.savefig(r'F:\research\AI+WC\beamforming\Passive BF LIS\paper\arxiv_edit\loss.png')