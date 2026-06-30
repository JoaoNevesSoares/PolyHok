import csv
import scipy
import sys
from scipy import stats
import numpy as np

i = 0
ufs = np.empty(29)
fs = np.empty(29)

file_path = sys.argv[1]
with open(file_path, newline='') as f: 
    reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        ufs[i] = float(row[0])
        fs[i] = float(row[1])
        i = i + 1

print("--------------------------------------------")
print(f"FILE = {file_path}")
res = stats.wilcoxon(fs,ufs)
print(f"W = {res.statistic}")
print(f"p-value = {res.pvalue}")

min_fs = np.min(fs)
min_ufs = np.min(ufs)

q1_fs = np.percentile(fs, 25)
q1_ufs = np.percentile(ufs,25)

median_fs = np.median(fs)
median_ufs = np.median(ufs)

q3_fs = np.percentile(fs, 75)
q3_ufs = np.percentile(ufs, 75)

max_fs = np.max(fs)
max_ufs = np.max(ufs)

std_fs = np.std(fs)
std_ufs = np.std(ufs)

mean_ufs = np.mean(ufs)
mean_fs = np.mean(fs)

print("Fused: ")
print(f"Mean: {mean_fs}")
print(f"std dev: {std_fs}")
print(f"Min: {min_fs}")
print(f"Q1: {q1_fs}")
print(f"Median: {median_fs}")
print(f"Q3: {q3_fs}")
print(f"Max: {max_fs}")


print("Without Fusion: ")
print(f"Mean: {mean_ufs}")
print(f"std dev: {std_ufs}")
print(f"Min: {min_ufs}")
print(f"Q1: {q1_ufs}")
print(f"Median: {median_ufs}")
print(f"Q3: {q3_ufs}")
print(f"Max: {max_ufs}")
