#%%
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler # Copy qt figure to clipboard using Ctrl+C
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

from sklearn.manifold import TSNE

# Paired comparison (within group)
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

# Unpaired comparison (between group)
from scipy.stats import ttest_ind
from scipy.stats import ranksums

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.anova import anova_lm

# Display plots in a separate window (pyqt)
# %matplotlib qt

#%% Load toy datasets from sickit-learn
plt.close("all")
from sklearn.datasets import load_iris, load_diabetes
from scipy.stats import shapiro # Test if data follows Normal distribution


print('Loading iris dataset..\n')
data_iris = load_iris()

print('Target names: ', data_iris.target_names)
print('Feature names: ', data_iris.feature_names)
print()

X, y = data_iris.data, data_iris.target

X_class0 = X[y == 0]
X_class1 = X[y == 1]
X_class2 = X[y == 2]

# Display reduced feture maps using t-sne
mdl = TSNE(n_components=2, random_state=0)
X_embedded = mdl.fit_transform(X)
fig, ax = plt.subplots(1, 3)
palette = sns.color_palette("bright", 3)
df_embedded = pd.DataFrame({'x1': X_embedded[:, 0], 'x2': X_embedded[:, 1], 'label': y})
sns.scatterplot(data=df_embedded, x='x1', y='x2',
                hue=df_embedded["label"], palette=palette, ax=ax[0], legend='full')
ax[0].set_title('t-SNE')

mdl2 = PCA(n_components=2)
X_embedded_PCA = mdl2.fit_transform(X)
palette = sns.color_palette("bright", 3)
df_embedded = pd.DataFrame({'x1': X_embedded_PCA[:, 0], 'x2': X_embedded_PCA[:, 1], 'label': y})
sns.scatterplot(data=df_embedded, x='x1', y='x2',
                hue=df_embedded["label"], palette=palette, ax=ax[1], legend='full')
ax[1].set_title('PCA')


# Display non-Gaussian distribution (or Gausssian mixture) regardless of target class
ax[2].hist(X_class0[:, 2], bins=50, density=True)    # plot the histogram
ax[2].hist(X_class1[:, 2], bins=50, density=True)    # plot the histogram
ax[2].hist(X_class2[:, 2], bins=50, density=True)    # plot the histogram

stat, p = shapiro(X[:, 2])
print('Normality test on X')
print('X: Statistics=%.3f, p=%.3f\n' % (stat, p)) # Non-Gaussian if p < 0.05
if p < 0.05:
    ax[2].set_title('Non-Gaussian feature: '+ data_iris.feature_names[2])
else:
    ax[2].set_title('Gaussian feature: '+ data_iris.feature_names[2])


#%% Compare two classes

test_x1 = X_class1[:, 2]
test_x2 = X_class2[:, 2]

fig, ax = plt.subplots(1, 2)
ax[0].boxplot([test_x1, test_x2])
ax[0].set_xticklabels(['Class 1', 'Class 2'])

ax[1].hist(test_x1, density=True)    # plot the histogram
ax[1].hist(test_x2, density=True)    # plot the histogram
ax[1].legend(['Class 1', 'Class 2'])
ax[1].axvline(np.mean(test_x1), color='b') # Average
ax[1].axvline(np.median(test_x1), color='b', linestyle='--') # Median

ax[1].axvline(np.mean(test_x2), color='r') # Average
ax[1].axvline(np.median(test_x2), color='r', linestyle='--') # Median

stat, p = shapiro(test_x1)
print('X1: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05
stat, p = shapiro(test_x2)
print('X2: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05

res_ttest_ind = ttest_ind(test_x1, test_x2) # diff. btw. two group 
print('Unpaired t-test')
print('p-val: ', res_ttest_ind.pvalue)
print('t-val: ', res_ttest_ind.statistic)

res_ttest_rel = ttest_rel(test_x1, test_x2) # average of diff.
print('Paired t-test')
print('p-val: ', res_ttest_rel.pvalue)
print('t-val: ', res_ttest_rel.statistic)

res_wilcoxon_signed_rank = wilcoxon(test_x1, test_x2)
print('Wilcoxon signed rank test')
print('p-val: ', res_wilcoxon_signed_rank.pvalue)

res_wilcoxon_ranksum = ranksums(test_x1, test_x2)
print('Wilcoxon rank sum test')         # Mann-Whitney U test
print('p-val: ', res_wilcoxon_ranksum.pvalue) 

#%% Simulate data difficiency

def choose_subsamples_rows(X_in, M, seed=None):
    np.random.seed(seed)
    indicies_sub = np.random.choice(X_in.shape[0], size=M, replace=False)
    return X_in[indicies_sub]

M = 10
X_class1_sub = choose_subsamples_rows(test_x1, M)
X_class2_sub = choose_subsamples_rows(test_x2, M)

fig, ax = plt.subplots(1, 2)
ax[0].boxplot([X_class1_sub, X_class2_sub])
ax[0].set_xticklabels(['Class 1-sub', 'Class 2-sub'])

ax[1].hist(X_class1_sub, density=True)    # plot the histogram
ax[1].hist(X_class2_sub, density=True)    # plot the histogram
ax[1].legend(['Class 1-sub', 'Class 2-sub'])
ax[1].axvline(np.mean(X_class1_sub), color='b') # Average
ax[1].axvline(np.median(X_class1_sub), color='b', linestyle='--') # Median

ax[1].axvline(np.mean(X_class2_sub), color='r') # Average
ax[1].axvline(np.median(X_class2_sub), color='r', linestyle='--') # Median

# Not reliable when data is not enough
stat, p = shapiro(X_class1_sub)
print('Shapiro test on X1: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05
stat, p = shapiro(X_class2_sub)
print('Shapiro test on X2: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05

res_ttest_ind = ttest_ind(X_class1_sub, X_class2_sub)
print('Unpaired t-test')
print('p-val: ', res_ttest_ind.pvalue)
print('t-val: ', res_ttest_ind.statistic)


res_ttest_rel = ttest_rel(X_class1_sub, X_class2_sub)
print('Paired t-test')
print('p-val: ', res_ttest_rel.pvalue)
print('t-val: ', res_ttest_rel.statistic)

res_wilcoxon_signed_rank = wilcoxon(X_class1_sub, X_class2_sub)
print('Wilcoxon signed rank test')
print('p-val: ', res_wilcoxon_signed_rank.pvalue)

res_wilcoxon_ranksum = ranksums(X_class1_sub, X_class2_sub)
print('Wilcoxon rank sum test')         # Mann-Whitney U test
print('p-val: ', res_wilcoxon_ranksum.pvalue) 


#%% Case 2: Similar data

# Split the same class data into two groups - they should be similar
X_class1_splitA = X_class0[:25, :]
X_class1_splitB = X_class0[25:, :]

X_partA = X_class1_splitA[:, 2]
X_partB = X_class1_splitB[:, 2]

fig, ax = plt.subplots(1, 2)
ax[0].boxplot([X_partA, X_partB])
ax[0].set_xticklabels(['Part A', 'Part B'])

ax[1].hist(X_partA, density=True)    # plot the histogram
ax[1].hist(X_partB, density=True)    # plot the histogram
ax[1].legend(['Class 1', 'Class 2'])
ax[1].axvline(np.mean(X_partA), color='b') # Average
ax[1].axvline(np.median(X_partA), color='b', linestyle='--') # Median

ax[1].axvline(np.mean(X_partB), color='r') # Average
ax[1].axvline(np.median(X_partB), color='r', linestyle='--') # Median


# Not reliable when data is not enough
stat, p = shapiro(X_partA)
print('Shapiro test on X1: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05
stat, p = shapiro(X_partB)
print('Shapiro test on X2: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05

res_ttest_ind = ttest_ind(X_partA, X_partB)
print('Unpaired t-test')
print('p-val: ', res_ttest_ind.pvalue)
print('t-val: ', res_ttest_ind.statistic)

res_ttest_rel = ttest_rel(X_partA, X_partB)
print('Paired t-test')
print('p-val: ', res_ttest_rel.pvalue)
print('t-val: ', res_ttest_rel.statistic)

res_wilcoxon_signed_rank = wilcoxon(X_partA, X_partB)
print('Wilcoxon signed rank test')
print('p-val: ', res_wilcoxon_signed_rank.pvalue)

res_wilcoxon_ranksum = ranksums(X_partA, X_partB)
print('Wilcoxon rank sum test')         # Mann-Whitney U test
print('p-val: ', res_wilcoxon_ranksum.pvalue) 


#%% Case 2: Similar data with fewer samples

# Choose part of the data
# Run several times until it shows significant difference by chance
M = 10
X_partA_sub = choose_subsamples_rows(X_partA, M)
X_partB_sub = choose_subsamples_rows(X_partB, M)

fig, ax = plt.subplots(1, 2)
ax[0].boxplot([X_partA_sub, X_partB_sub])
ax[0].set_xticklabels(['Part A', 'Part B'])

ax[1].hist(X_partA_sub, density=True)    # plot the histogram
ax[1].hist(X_partB_sub, density=True)    # plot the histogram
ax[1].legend(['Class 1', 'Class 2'])
ax[1].axvline(np.mean(X_partA_sub), color='b') # Average
ax[1].axvline(np.median(X_partA_sub), color='b', linestyle='--') # Median

ax[1].axvline(np.mean(X_partB_sub), color='r') # Average
ax[1].axvline(np.median(X_partB_sub), color='r', linestyle='--') # Median

# Not reliable when data is not enough
stat, p = shapiro(X_partA_sub)
print('Shapiro test on X1: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05
stat, p = shapiro(X_partB_sub)
print('Shapiro test on X2: Statistics=%.3f, p=%.3f' % (stat, p)) # Non-Gaussian if p < 0.05

res_ttest_ind = ttest_ind(X_partA_sub, X_partB_sub)
print('Unpaired t-test')
print('p-val: ', res_ttest_ind.pvalue)
print('t-val: ', res_ttest_ind.statistic)

res_ttest_rel = ttest_rel(X_partA_sub, X_partB_sub)
print('Paired t-test')
print('p-val: ', res_ttest_rel.pvalue)
print('t-val: ', res_ttest_rel.statistic)

res_wilcoxon_signed_rank = wilcoxon(X_partA_sub, X_partB_sub)
print('Wilcoxon signed rank test')
print('p-val: ', res_wilcoxon_signed_rank.pvalue)

res_wilcoxon_ranksum = ranksums(X_partA_sub, X_partB_sub)
print('Wilcoxon rank sum test')         # Mann-Whitney U test
print('p-val: ', res_wilcoxon_ranksum.pvalue) 
