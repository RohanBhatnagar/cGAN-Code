import numpy as np
import matplotlib.pyplot as plt

x_data = np.load('X_data.npy') 
y_clean_data = np.load('Y_clean_data.npy')
y_noisy_data = np.load('Y_noisy_data.npy')

samples = 10

plt.figure()
fig,axs = plt.subplots(samples,2,figsize=(20,5*samples))

axs = axs.flatten()

ind = 0
for i in range(samples):
    axs[ind].plot(x_data[i,:])
    axs[ind].set_ylabel('x')
    ind+=1

    axs[ind].plot(y_clean_data[i,:])
    axs[ind].plot(y_noisy_data[i,:])
    axs[ind].set_ylabel('y')
    ind+=1

plt.savefig('samples.pdf',bbox_inches='tight')



    
