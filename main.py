from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# remove warnings
pd.set_option('mode.chained_assignment', None)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Atlikite duomenų esančių lentelėje 2 išskaidymą į 3 klasterius. Pritaikykite k-means algoritmą.
# Kokie pacientai pataikė į rizikos grupę, tai yra į vieną klasterį su sergančiu pacientu "Vardas1"?

# import data
df = pd.read_csv('data.csv')

# print original table
print('\n-----------------------------------')
print('Original: \n')
print(df)
print('\n-----------------------------------')

# get all the symptoms columns
# symptoms = list(df.columns)[:-8]
# symptoms = list(df.columns)[:-2]
symptoms = list(df.columns)

# get the symptoms data
data = df[symptoms]
print("Data:\n", data)
print('\n-----------------------------------')

# perform clustering here
clustering_kmeans = KMeans(n_clusters=2, precompute_distances="auto", n_jobs=-1)
data['clusters'] = clustering_kmeans.fit_predict(data)
print("PREDICTED CLUSTERS \n", data)

pca_num_components = 2

# Well, you cannot do it directly if you have more than 3 columns.
# However, you can apply a Principal Component Analysis to reduce
# the space in 2 columns and visualize this instead.

# run PCA (Principal Component Analysis) on the data and reduce dimensions in pca_num_components dimensions
reduced_data = PCA(n_components=pca_num_components).fit_transform(data)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
