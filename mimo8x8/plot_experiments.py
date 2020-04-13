import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import os

#### for beijing #####
matched1 = pd.read_csv('1-bit\\all_nmse_matched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched1 = pd.read_csv('1-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed1 = pd.read_csv('1-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched3 = pd.read_csv('3-bit\\all_nmse_matched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched3 = pd.read_csv('3-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed3 = pd.read_csv('3-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched5 = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched5 = pd.read_csv('5-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed5 = pd.read_csv('5-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_rosslyn_beijing.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
#### for rosslyn ####
'''
matched1 = pd.read_csv('1-bit\\all_nmse_matched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched1 = pd.read_csv('1-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed1 = pd.read_csv('1-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched3 = pd.read_csv('3-bit\\all_nmse_matched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched3 = pd.read_csv('3-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed3 = pd.read_csv('3-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))

matched5 = pd.read_csv('5-bit\\all_nmse_matched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
mismatched5 = pd.read_csv('5-bit\\all_nmse_mismatched_Nr8_Nt8_numEx256_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
proposed5 = pd.read_csv('5-bit\\all_nmse_pretrained_Nr8_Nt8_numEx256_beijing_rosslyn.txt', sep=" ", header=None, names = np.arange(-21, 22, 3))
'''

## computing performance
print(matched1.loc[0,10:])
sum_mat = sum(matched1.loc[0,10:])
sum_prop = sum(proposed1.loc[0,10:])

print(sum_mat)
print(sum_prop)

lower = 100 - (sum_mat * 100) / sum_prop
print(lower)

mismatched1.loc[0,:].plot(style='r-', label='1 bit mismatched')
mismatched3.loc[0,:].plot(style='b-', label='3 bits mismatched')
mismatched5.loc[0,:].plot(style='k-', label='5 bits mismatched')
matched1.loc[0,:].plot(style='r--', label='1 bit matched')
proposed1.loc[0,:].plot(style='r-d', label='1 bit proposed')
proposed3.loc[0,:].plot(style='b-d', label='3 bits proposed')
matched3.loc[0,:].plot(style='b--', label='3 bits matched')
proposed5.loc[0,:].plot(style='k-d', label='5 bits proposed')
matched5.loc[0,:].plot(style='k--', label='5 bits matched')

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

plt.xlabel("SNRdB")
plt.ylabel("NMSE AVG")
#plt.title("MIMO low resolution - Deep Learning Models for Rosslyn")
plt.legend()
plt.show()