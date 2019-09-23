from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import io
import numpy as np

#%% single antenna
M = 1
N_list = [8,16,32,64]
random_list = []
optimal_list = []
un_list = []
for N in N_list:
    results = io.loadmat('./mat/bigscale/M=%dN=%d.mat'%(M,N))
    random_list.append(np.squeeze(results['random_performance']))
    optimal_list.append(np.squeeze(results['cvx_performance']))
    un_list.append(np.squeeze(results['un_performance']))

ax=plt.subplot(111)
#plt.xlim(-20.5, 20)
#plt.ylim(0, 105)
plt.plot(N_list,random_list,'yo-',N_list,optimal_list,'ro-',N_list,un_list,'bo-')
plt.xlabel('N')
plt.ylabel('Objective')
plt.title('M = %d'%M)
plt.legend(['Random','Optimal','RISBFNN'])
xmajorLocator   = MultipleLocator(8)
xmajorFormatter = FormatStrFormatter('%d') 
xminorLocator   = MultipleLocator(8) 
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
#plt.savefig('./SingleAntenna.eps',format='eps',doi=1000)
plt.savefig(r'F:\research\AI+WC\beamforming\Passive BF LIS\paper\arxiv_edit\SingleAntenna.png')

#%% impact of N in multi-antenna scenario
N_list = [8,16,32,64]
ax=plt.subplot(111)

M = 4
random_list = []
cvx_list = []
un_list = []
for N in N_list:
    results = io.loadmat('./mat/bigscale/M=%dN=%d.mat'%(M,N))
    random_list.append(np.squeeze(results['random_performance']))
    cvx_list.append(np.squeeze(results['cvx_performance']))
    un_list.append(np.squeeze(results['un_performance']))

plt.plot(N_list,random_list,'yo-',N_list,cvx_list,'ro-',N_list,un_list,'bo-')
#plt.plot(N_list,random_list,'o-',N_list,cvx_list,'o-',N_list,un_list,'o-',\
#         N_list,random_list2,'^-',N_list,cvx_list2,'^-',N_list,un_list2,'^-')

plt.xlabel('N')
plt.ylabel('Objective')
plt.title('M = %d'%M)
plt.legend(['Random','SDR','RISBFNN'])
#plt.legend(['Random,M=2','SDR,M=2','RISBFNN,M=2','Random,M=4','SDR,M=4','RISBFNN,M=4'])
xmajorLocator   = MultipleLocator(8)
xmajorFormatter = FormatStrFormatter('%d') 
xminorLocator   = MultipleLocator(8) 
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
plt.savefig(r'F:\research\AI+WC\beamforming\Passive BF LIS\paper\arxiv_edit\VaryN.png')

#%% impact of M
M_list = [1,2,4,8]
ax=plt.subplot(111)

N = 16
cvx_list = []
un_list = []
for M in M_list:
    results = io.loadmat('./mat/bigscale/M=%dN=%d.mat'%(M,N))   
    cvx_list.append(np.squeeze(results['cvx_performance']))  
    un_list.append(np.squeeze(results['un_performance']))
    
N = 8
cvx_list2 = []
un_list2 = []
for M in M_list:
    results = io.loadmat('./mat/bigscale/M=%dN=%d.mat'%(M,N))   
    cvx_list2.append(np.squeeze(results['cvx_performance']))  
    un_list2.append(np.squeeze(results['un_performance']))
    
N = 32
cvx_list3 = []
un_list3 = []
for M in M_list:
    results = io.loadmat('./mat/bigscale/M=%dN=%d.mat'%(M,N))   
    cvx_list3.append(np.squeeze(results['cvx_performance']))  
    un_list3.append(np.squeeze(results['un_performance']))
    
#plt.xlim(0.5, 8.5)
#plt.ylim(0, max(np.max(cvx_list),np.max(un_list))+0.5)
#plt.plot(M_list,un_list,'o-',M_list,cvx_list,'o-')
plt.plot(M_list,un_list,'o-',M_list,cvx_list,'o-',M_list,un_list2,'^-',M_list,cvx_list2,'^-',\
         M_list,un_list3,'*-',M_list,cvx_list3,'*-')
plt.xlabel('M')
plt.ylabel('Objective')
#plt.title('N = %d'%N)
#plt.legend(['RISBFNN','Baseline'])
plt.legend(['RISBFNN,N=16','Baseline,N=16','RISBFNN,N=8','Baseline,N=8','RISBFNN,N=32','Baseline,N=32'])
xmajorLocator   = MultipleLocator(1)
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
plt.savefig(r'F:\research\AI+WC\beamforming\Passive BF LIS\paper\arxiv_edit\VaryM.png')