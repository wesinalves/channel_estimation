import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import os

bits = [1,3,5]
#y = np.linspace(10, 100, 20, dtype="int")
y = list(range(20))
#directory = '5-bit/sample_complexity'
def read_file(path, number, city, pretrained=0):
    if pretrained == 0:
        filename = os.path.join(path, f'mean_nmse_matched_Nr8_Nt8_numEx256_{city}_{number}.txt')
    else:
        filename = os.path.join(path, f'mean_nmse_pretrained_Nr8_Nt8_numEx256_{city}_{number}.txt')
    return filename

# read files regard to matched models    
for n in y:
    for b in bits:
        exec(f"directory = f'{b}-bit/sample_complexity'")
        exec(f"size{n}_{b} = pd.read_csv(read_file(directory,{n},'beijing',0), sep=' ', header=None, names = ['1db'])")


for b in bits:
    matched_string = "["
    for number in y:
        matched_string += f"size{number}_{b}.iloc[0],"
    matched_string += "]"
    exec(f"matched{b} = eval(matched_string)")

# read files regard to proposed models
for n in y:
    for b in bits:
        exec(f"directory = f'{b}-bit/sample_complexity'")
        exec(f"size{n}_{b} = pd.read_csv(read_file(directory,{n},'rosslyn_china',1), sep=' ', header=None, names = ['1db'])")

for b in bits:
    proposed_string = "["
    for n in y:
        proposed_string += f"size{n}_{b}.iloc[0],"
    proposed_string += "]"
    exec(f"proposed{b} = eval(proposed_string)")


## computing performance
print(matched1[:8])
sum_mat = sum(matched1[:8])
sum_prop = sum(proposed1[:8])

print(sum_mat)
print(sum_prop)

lower = 100 - (sum_mat * 100) / sum_prop
print(lower)

plt.figure(figsize=(7,2.5))

x = np.linspace(10*5,100*5,20,dtype="int")
plt.plot(x, matched1, "r--s", label="1 bit matched")
plt.plot(x, proposed1, "k-s", label="1 bit DTL")
plt.plot(x, matched3, "r--P", label="3 bits matched")
plt.plot(x, proposed3, "k-P", label="3 bits DTL")
plt.plot(x, matched5, "r--o", label="5 bits matched")
plt.plot(x, proposed5, "k-o", label="5 bits DTL")

plt.subplots_adjust(top=0.97,
                    bottom=0.185,
                    left=0.11,
                    right=0.990,
                    hspace=0.2,
                    wspace=0.2)


plt.xlabel("Number of training examples")
plt.ylabel("NMSE")
#plt.title("MIMO low resolution - Sample Complexity")
#plt.legend(["Matched 1bit", "Proposed 1bit", "Matched 3bit", "Proposed 3bit", "Matched 5bit", "Proposed 5bit"])
plt.xlim(45,505)
#plt.ylim((0.090, 0.990))
plt.grid(True)
plt.legend(ncol=2)
plt.show()
