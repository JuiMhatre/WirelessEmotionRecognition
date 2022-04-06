from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from scr.ReadData import ReadData
from scr.utils import Utils
import pandas as pd
import numpy as np
readData = ReadData()
utils = Utils()

X, Y = readData.readVideoDataHRV()
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2','pc3'])

plt.plot(principalDf['pc2'],principalDf['pc2'] * [1], "x")
plt.plot(principalDf['pc3'],principalDf['pc3'] * [1], "y")
# plt.plot(principalDf['pc3'],principalDf['pc3'] * [1], "z")

# colors = ['red','green','blue','purple','yellow','black','pink','cyan']
#
# fig = plt.figure()
# plt.scatter(principalDf['pc1'], principalDf['pc2'], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
#
# cb = plt.colorbar()
# loc = np.arange(0,max(Y),max(Y)/float(len(colors)))
# cb.set_ticks(loc)
# cb.set_ticklabels(colors)
plt.show()