#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA

# Paired comparison (within group)
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

# Unpaired comparison (between group)
from scipy.stats import ttest_ind
from scipy.stats import ranksums

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.anova import anova_lm

%matplotlib qt
#%% Generate random Gaussian data

n = 1000
np.random.seed(0)
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n) + 0.5

fig, ax = plt.subplots()
ax.scatter(x, y)    # plot the data
ax.set_aspect('equal')    # make the aspect ratio equal
ax.set_title('Gaussian data')    # set the title

fig, ax = plt.subplots()
ax.hist(x, bins=100, density=True)    # plot the histogram
ax.hist(y, bins=100, density=True)    # plot the histogram

fig, ax = plt.subplots()
ax.boxplot([x, y])    # plot the mean and standard deviation

res_ttest_rel = ttest_rel(x, y)
print(res_ttest_rel.pvalue)
print(res_ttest_rel.statistic)


#%% Generate non-Gaussian data

x_non_gaussian = np.exp(x)
y_non_gaussian = np.exp(y)


fig, ax = plt.subplots()
ax.scatter(x_non_gaussian, y_non_gaussian)    # plot the data
ax.set_aspect('equal')    # make the aspect ratio equal
ax.set_title('Non-Gaussian data')    # set the title

fig, ax = plt.subplots()
ax.hist(x_non_gaussian, bins=30, density=True)    # plot the histogram
ax.hist(y_non_gaussian, bins=30, density=True)    # plot the histogram

fig, ax = plt.subplots()
ax.boxplot([x_non_gaussian, y_non_gaussian])    # plot the mean and standard deviation