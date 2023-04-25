import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import os

#### for beijing #####

matched1 = pd.read_csv('1-bit\\all_nmse_matched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched1 = pd.read_csv('1-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed1 = pd.read_csv('1-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
gamp1 = pd.read_csv('1-bit\\all_nmse_gamp_Nr8_Nt64_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched3 = pd.read_csv('3-bit\\all_nmse_matched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched3 = pd.read_csv('3-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed3 = pd.read_csv('3-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
gamp3 = pd.read_csv('3-bit\\all_nmse_gamp_Nr8_Nt64_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched5 = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched5 = pd.read_csv('5-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed5 = pd.read_csv('5-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
gamp5 = pd.read_csv('5-bit\\all_nmse_gamp_Nr8_Nt64_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
#### for rosslyn ####
'''
matched1 = pd.read_csv('1-bit\\all_nmse_matched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched1 = pd.read_csv('1-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed1 = pd.read_csv('1-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched3 = pd.read_csv('3-bit\\all_nmse_matched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched3 = pd.read_csv('3-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed3 = pd.read_csv('3-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched5 = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
matched5t = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt32_numEx256_rosslyntest.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
matched5p = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt32_numEx128_rosslynpilot128.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
matched5f16 = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt32_numEx256_rosslynpilot16f.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched5 = pd.read_csv('5-bit\\all_nmse_mismatched_Nr8_Nt32_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed5 = pd.read_csv('5-bit\\all_nmse_pretrained_Nr8_Nt32_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
'''


## computing performance
print(matched1.loc[0,10:])
sum_mat = sum(matched1.loc[0,10:])
sum_prop = sum(proposed1.loc[0,10:])

print(sum_mat)
print(sum_prop)

lower = 100 - (sum_mat * 100) / sum_prop
print(lower)

plt.figure(figsize=(7,2.5))
mismatched1.loc[0,:].plot(style='b-s', label='1 bit mismatched')
mismatched3.loc[0,:].plot(style='b-P', label='3 bits mismatched')
mismatched5.loc[0,:].plot(style='b-o', label='5 bits mismatched')
matched1.loc[0,:].plot(style='r--s', label='1 bit matched')
proposed1.loc[0,:].plot(style='k-s', label='1 bit DTL')
proposed3.loc[0,:].plot(style='k-P', label='3 bits DTL')
matched3.loc[0,:].plot(style='r--P', label='3 bits matched')
proposed5.loc[0,:].plot(style='k-o', label='5 bits DTL')
matched5.loc[0,:].plot(style='r--o', label='5 bits matched')

gamp1.loc[0,:].plot(style='g-o', label='1 bit gamp')
gamp3.loc[0,:].plot(style='g-s', label='3 bits gamp')
gamp5.loc[0,:].plot(style='g-P', label='5 bits gamp')
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.xlim(-21.5, 21.5)
plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.grid(True)

'''
plt.figure(figsize=(10,7))
plt.subplot(1,3,1)
mismatched1.loc[0,:].plot(style='b--s', label='mismatched')
gamp1.loc[0,:].plot(style='g--s', label='gamp Nt=64')
proposed1.loc[0,:].plot(style='k--s', label='DTL')
matched1.loc[0,:].plot(style='r--s', label='matched')
plt.title('Channel Estimation (256 pilots 1 bit) \n (Nt=32, Nr=8)')
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.xlim(-21.5, 21.5)
#plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.legend()
plt.grid(True)


plt.subplot(1,3,2)
mismatched3.loc[0,:].plot(style='b--s', label='mismatched')
gamp3.loc[0,:].plot(style='g--s', label='gamp Nt=64')
proposed3.loc[0,:].plot(style='k--s', label='DTL')
matched3.loc[0,:].plot(style='r--s', label='matched')
plt.title('Channel Estimation (256 pilots 3 bits) \n (Nt=32, Nr=8)')
plt.xlabel("SNR (dB)")
plt.xlim(-21.5, 21.5)
#plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
mismatched5.loc[0,:].plot(style='b--s', label='mismatched')
gamp5.loc[0,:].plot(style='g--s', label='gamp Nt=64')
proposed5.loc[0,:].plot(style='k--s', label='DTL')
matched5.loc[0,:].plot(style='r--s', label='matched')
plt.title('Channel Estimation (256 pilots 5 bits) \n (Nt=32, Nr=8)')
plt.xlabel("SNR (dB)")
plt.xlim(-21.5, 21.5)
#plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.legend()
plt.grid(True)

# matched5t.loc[0,:].plot(style='g--o', label='5 bits matched test')
# matched5p.loc[0,:].plot(style='g-s', label='5 bits matched p128')
# matched5f16.loc[0,:].plot(style='g-P', label='5 bits matched f16')
'''
plt.subplots_adjust(top=0.97,
                    bottom=0.185,
                    left=0.11,
                    right=0.68,
                    hspace=0.2,
                    wspace=0.2)

'''
bits = [1,3,5]
colors = ['r','b','k']
styles = ['-','--','-d']
for index,b in enumerate(bits):
    #if b != 3:
    #    continue
    eval(f"matched{b}.loc[0,:].plot(style='{colors[index]}{styles[1]}', label='matched {b} bits')")
    eval(f"mismatched{b}.loc[0,:].plot(style='{colors[index]}{styles[0]}', label='mismatched {b} bits')")
    eval(f"proposed{b}.loc[0,:].plot(style='{colors[index]}{styles[2]}', label='proposed {b} bits')")
'''

# plt.xlabel("SNR (dB)")
# plt.ylabel("NMSE (dB)")
# plt.xlim(-21.5, 21.5)
# #plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
# plt.grid(True)
plt.show()