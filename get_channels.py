'''
Script to extract channels from raytrace data
Authors:
Wesin Ribeiro
Marcus Yuichi
2019
'''
from scipy.io import loadmat, savemat
import numpy as np

mat = loadmat("rosslyn_5s.mat")
Ht = mat['Ht']



# permute dimensions before reshape: scenes before episodes
# found out np.moveaxis as alternative to permute in matlab
Ht = Ht[~np.isnan(Ht).any(axis=4)]
Ht = Ht.reshape((-1,1,1,8,8))

n_episodes, n_scenes, n_receivers, n_r, n_t = Ht.shape

Ht = np.moveaxis(Ht, 1, 0)

reshape_dim = n_episodes * n_scenes * n_receivers

Harray = np.reshape(Ht, (reshape_dim, n_r, n_t))
Hvirtual = np.zeros(Harray.shape, dtype='complex128')
scaling_factor = 1 / np.sqrt(n_r * n_t)

for i in range(reshape_dim):
    m = np.squeeze(Harray[i,:,:])
    Hvirtual[i,:,:] = scaling_factor * np.fft.fft2(m)

savemat('channel_rosslyn60Ghz.mat', {'Harray': Harray, 'Hvirtual':Hvirtual})
