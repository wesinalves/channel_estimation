'''
Script to extract channels from raytrace data
Authors:
Wesin Ribeiro
Marcus Yuichi
2019
'''
from scipy.io import loadmat, savemat
import numpy as np

##################################
### Script configuration
##################################

if True: #### Load pre-processed RT data
    inputFile = 'rosslyn_5s.mat'
    outputFile = 'channel_rosslyn60Ghz.mat'
    inputType = 'mat' # mat or h5
else: ### Provide RT data to pre-process
    
    # LASSE's mimo-python package
    # https://gitlab.lasse.ufpa.br/software/mimo-python.git
    mimoPythonPath = '/home/ilan/git/lasse/mimo-python/mimopython/channel_characterization'
    
    # Provide path the Raymobtime h5 data that can be downloaded at
    # https://www.lasse.ufpa.br/raymobtime/
    inputFile = \
        '/home/ilan/Documents/Raymobtime/dataset/s004/rosslyn_mobile_60GHz_ts1s_V_e*.hdf5'
    outputFile = 'channels_rosslyn_60Ghz_Nr8Nt32_mobile_s004.mat'
    inputType = 'h5' # mat or h5
    Nr = 8
    Nt = 32

#################################
#### Start processing
#################################

if inputType == 'mat':
    mat = loadmat(inputFile)
    Ht = mat['Ht']
    Nr = 8
    Nt = 8
elif inputType == 'h5':
    
    import h5py
    import os
    import sys
    
    sys.path.append(mimoPythonPath)
    import mimo_channels
    
    episodeRays = h5py.File(inputFile.replace('*', str(0)), 'r').get('allEpisodeData')
    numUsers = episodeRays.shape[1]
    numScenes = episodeRays.shape[0]
    numRays = episodeRays.shape[2]
    numEpisodes = 0
    while True:
        if not os.path.isfile(inputFile.replace('*',str(numEpisodes))):
            break
        numEpisodes += 1
        
    Ht = np.zeros((numEpisodes,numScenes,numUsers,Nr,Nt))
    
    currentEpisode = 0
    numChannels = 0
    numValidChannels = 0
    print('Processing ...')
    for iEpisode in range(numEpisodes):
        
        print('\r\t Episode: %d'%iEpisode, end='')
        currentFile = inputFile.replace('*', str(iEpisode))
        
        episodeRays = h5py.File(currentFile, 'r').get('allEpisodeData')
        
        for iScene in range(numScenes):
            Huser = np.zeros((numUsers,Nr,Nt))
            for iUser in range(numUsers):
                numChannels += 1
                # Check valid rays for user
                numUserRays = 0
                for iRay in range(numRays):
                    if np.isnan(episodeRays[iScene,iUser,iRay,:]).any():
                        break
                    numUserRays += 1
                
                if numUserRays == 0:
                    Huser[iUser,:,:] = np.nan*np.zeros((1,Nr,Nt))
                else:
                    numValidChannels += 1
                    gain_in_dB = episodeRays[iScene, iUser, 0:numUserRays, 0]
                    timeOfArrival = episodeRays[iScene, iUser, 0:numUserRays, 1]
                    AoD_el = episodeRays[iScene, iUser, 0:numUserRays, 2]
                    AoD_az = episodeRays[iScene, iUser, 0:numUserRays, 3]
                    AoA_el = episodeRays[iScene, iUser, 0:numUserRays, 4]
                    AoA_az = episodeRays[iScene, iUser, 0:numUserRays, 5]
                    isLOSperRay = episodeRays[iScene, iUser, 0:numUserRays, 6]
                    pathPhases = episodeRays[iScene, iUser, 0:numUserRays, 7]
                        
                    Huser[iUser,:,:] = mimo_channels.getNarrowBandULAMIMOChannel(\
                        AoD_az, AoA_az, gain_in_dB, Nt, Nr)
            
            Ht[iEpisode, :, :, :, :] = Huser
    
    print('### Finished processing channels')
    print('\t %d Total of channels'%numChannels)
    print('\t %d Total of valid channels'%numValidChannels)

# permute dimensions before reshape: scenes before episodes
# found out np.moveaxis as alternative to permute in matlab
Ht = Ht[~np.isnan(Ht).any(axis=4)]
Ht = Ht.reshape((-1,1,1,Nr,Nt))

n_episodes, n_scenes, n_receivers, n_r, n_t = Ht.shape

Ht = np.moveaxis(Ht, 1, 0)

reshape_dim = n_episodes * n_scenes * n_receivers

Harray = np.reshape(Ht, (reshape_dim, n_r, n_t))
Hvirtual = np.zeros(Harray.shape, dtype='complex128')
scaling_factor = 1 / np.sqrt(n_r * n_t)

for i in range(reshape_dim):
    m = np.squeeze(Harray[i,:,:])
    Hvirtual[i,:,:] = scaling_factor * np.fft.fft2(m)

# savemat(outputFile, {'Harray': Harray, 'Hvirtual':Hvirtual})
