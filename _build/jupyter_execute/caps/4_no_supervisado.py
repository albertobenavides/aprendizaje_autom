#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje no supervisado
# 
# Los métodos no supervisados son aquellos que, sin conocer etiquetas o parámetros objetivo en los datos, se encargan de agrupar, crear subconjuntos o subordinaciones de subconjuntos de datos a partir de distancias entre los datos o subconjuntos de los mismos.

# ## Análisis de correspondencia
# 
# Uno de los métodos de aprendizaje no supervisado se vio al final de la sesión anterior, PCA. Ahora toca a su contraparte categórica, el **análisis de correspondencia**. Aquí, se estandarizan parámetros continuos y se reducen las dimensiones de los parámetros estandarizados mediante PCA para representar las variables categóricas.

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../data/results/df_pollutants_melt.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.timestamp = df.timestamp.values.astype(np.int64) / 10 ** 9
df


# In[3]:


df_sel = df[['timestamp', 'station', 'variable', 'value']]
df_sel


# In[4]:


print(df_sel.iloc[:10].to_latex(index=False))


# In[5]:


df_sel_x = df_sel[[
    #'timestamp', 
    'value']]
df_sel_x


# In[6]:


from sklearn.preprocessing import MinMaxScaler


# In[7]:


scaler = MinMaxScaler()
# https://stackoverflow.com/a/43383700
scaled = scaler.fit_transform(df_sel_x)
# https://datatofish.com/numpy-array-to-pandas-dataframe/
df_scaled = pd.DataFrame(scaled, columns = df_sel_x.columns)
df_scaled


# In[8]:


df_scaled['variable'] = df_sel['variable']
df_scaled['station'] = df_sel['station']
df_scaled


# In[9]:


df_dropna = df_scaled.dropna()
df_dropna


# In[10]:


gp = df_dropna.groupby(['station', 'variable']).mean()
gp_unstacked = gp.unstack().dropna()
gp_unstacked.columns = gp_unstacked.columns.droplevel()
gp_unstacked


# In[11]:


import prince


# In[12]:


ca = prince.CA(n_components=2)
ca = ca.fit(gp_unstacked)
ca.plot(gp_unstacked)


# In[13]:


# [ ] Comparación entre frecuencias de palabras para autores


# ## $K$-medias
# 
# Esta técnica no supervisada consiste en generar $K \in \mathbb{N}$ grupos para $n$ elementos que incluyan a los $n_k$ más cercanos (con base en cierta medida de distancia, generalmente euclidiana) respecto a un centroide $c_k = (\hat{x}, \hat{y})$ tal que
# $$\hat{x} = \frac{1}{n_k} \sum_{i \in K} x_i,$$ 
# $$\hat{y} = \frac{1}{n_k} \sum_{i \in K} y_i.$$ 
# 
# Formalmente, la distancia más cercana respecto a los centroides $c_k$ se define como la minimización del error cuadrado para cada grupo
# $$\text{SS}_k = \sum_{i \in K} (x_i - \hat{x}_k)^2 + (y_i - \hat{y}_k)^2.$$
# 
# Este algoritmo es iterativo y tiene como función objetivo
# 
# $$\min \sum_{i \in K} \text{SS}_k.$$

# In[14]:


from sklearn.cluster import KMeans


# In[15]:


df_dropna = df_sel[['timestamp', 'value']].dropna()
scaled = scaler.fit_transform(df_dropna)
# https://datatofish.com/numpy-array-to-pandas-dataframe/
df_scaled = pd.DataFrame(scaled, columns = df_dropna.columns)
df_scaled


# In[16]:


# Se generan 15 grupos, por tener 15 variables
kmeans = KMeans(n_clusters = 15, n_init = 'auto').fit(df_scaled)


# In[17]:


# Esto nos da las etiquetas de los datos
kmeans.labels_


# In[18]:


# En kmeans, el tamaño de los grupos es variable y depende SÓLO de la distancia entre sus características
df_centroids_freq = pd.DataFrame(np.unique(kmeans.labels_, return_counts = True)).T
df_centroids_freq.columns = ['ck', 'nk']
df_centroids_freq


# In[19]:


# Aquí están los centroides 😯
kmeans.cluster_centers_


# In[20]:


df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = ['x', 'y'])
df_centers


# In[21]:


plt.figure()
plt.scatter(
    df_scaled.timestamp, 
    df_scaled.value, 
    s = 0.1, # tamaño de los marcadores
    c = kmeans.labels_
)
plt.scatter(df_centers.x, df_centers.y, s=10, c='k')
plt.show()


# ### Paralelización
# 
# Esto toma tiempo, pues a más elementos y grupos, más tiempo de ejecución

# In[22]:


# inertia = []
# for n_clusters in range(2, 15):
#     kmeans = KMeans(n_clusters=n_clusters).fit(df_scaled)
#     inertia.append(kmeans.inertia_ / n_clusters)


# Pero se puede paralelizar.

# In[23]:


# https://coderzcolumn.com/tutorials/python/joblib-parallel-processing-in-python
from joblib import Parallel, delayed, effective_n_jobs


# In[24]:


# Número de procesadores para usar
effective_n_jobs()


# In[25]:


# Ejemplo mínimo
import math


# In[26]:


get_ipython().run_cell_magic('time', '', 'Parallel(n_jobs = 2)(\n    delayed(math.factorial)(i) # (función)(parametro)\n    for i in range(10000)\n)\nprint()\n')


# In[27]:


get_ipython().run_cell_magic('time', '', 'Parallel(n_jobs=8)(delayed(math.factorial)(i) for i in range(10000))\nprint()\n')


# In[28]:


# Ahora sí con kmeans
def f_kmeans(n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init = 'auto').fit(df_scaled)
    return kmeans.inertia_ / n_clusters


# In[29]:


get_ipython().run_cell_magic('time', '', 'inertia = Parallel(n_jobs = 2, prefer="threads")(delayed(f_kmeans)(cn) for cn in range(2, 15))\n')


# In[30]:


get_ipython().run_cell_magic('time', '', 'inertia = Parallel(n_jobs = 8, prefer="threads")(delayed(f_kmeans)(cn) for cn in range(2, 15))\n')


# ### Seleccionar número de grupos

# In[31]:


df_inertias = pd.DataFrame({'n_clusters': range(2, 15), 'inertia': inertia})
df_inertias


# In[32]:


plt.figure()
plt.plot(df_inertias.n_clusters, df_inertias.inertia)
plt.xlabel('Número de grupos')
plt.ylabel('Media de las distancias cuadradas entre grupos')
plt.ylim((0, 1.1 * df_inertias.inertia.max()))
plt.xticks(range(2, 15))
plt.show()


# Para calcular el número de grupos por el método del codo, se calculan las distancias entre cada punto y la recta que va del primero al último. Generalmente, la primer distancia mayor corresponde con el mejor número de grupos.

# In[33]:


# https://stackoverflow.com/a/21566184/3113008
# De aquí sacamos la ecuación de la recta entre el punto primero y último

from numpy import ones,vstack
from numpy.linalg import lstsq
points = [
    (df_inertias.iloc[0, :].n_clusters, df_inertias.iloc[0, :].inertia),
    (df_inertias.iloc[-1, :].n_clusters, df_inertias.iloc[-1, :].inertia)
]
x_coords, y_coords = zip(*points)
A = vstack([x_coords, ones(len(x_coords))]).T
m, b = lstsq(A, y_coords, rcond=None)[0]


# In[34]:


plt.figure()
plt.plot(df_inertias.n_clusters, df_inertias.inertia)
plt.plot(df_inertias.iloc[[0, -1], :].n_clusters, df_inertias.iloc[[0, -1], :].inertia, c='r')
for i, r in df_inertias.iterrows():
    d = '$d_{' + str(i) + '}' + f' = {abs(round(r.inertia - (m * r.n_clusters + b), 2)):,}$'
    plt.annotate(d, xytext=(r.n_clusters, m * r.n_clusters + b), xy=(r.n_clusters, m * r.n_clusters + b), rotation = 45)
    plt.plot([r.n_clusters, r.n_clusters], [r.inertia, m * r.n_clusters + b], '--k')
plt.xlabel('Número de grupos')
plt.ylabel('Media de las distancias cuadradas entre grupos')
plt.xlim((1.5, 16.5))
plt.ylim((0, 1.15 * df_inertias.inertia.max()))
plt.xticks(range(2, 15))
plt.show()


# La mayor distancia se obtiene con $k = 4$, por lo que se toma como mejor número de $k$ por el método del codo.

# In[35]:


kmeans = KMeans(n_clusters=4, n_init = 'auto').fit(df_scaled)


# También se puede usar el método **Silhouette** (pero es más pesado que el método del codo 😓) 
# 
# $$
# \text{s} = \frac{a-b}{\max(a,b)}
# $$ 
# 
# donde $a$ es la distancia media entre los puntos dentro del grupo más cercano y $b$ es la distancia media entre cada punto respecto a todos los demás que están en los otros grupos a los que no pertenece. Los resultados obtenidos van de $1$ (las medias parten de grupos bien diferenciados) a $-1$ (las medias de los grupos indican que sus elementos), pasando por $0$ (distancias entre grupos se considera no significativa).

# In[36]:


from sklearn.metrics import silhouette_score


# In[37]:


silhouette_score(df_scaled, kmeans.labels_, metric = 'euclidean', sample_size = len(df_scaled) // 30, n_jobs = 8)


# ## Agrupamiento jerárquico
# 
# Este tipo de agrupamiento parte de $n \in \mathbb{N}$ datos con $p$ parámetros entre los cuales se pueden medir distancias $d_{i,j}, i, j \in n, i \neq j$ para formar $N \in \mathbb{N}$ grupos $A, B, C, \ldots$ con distancias $D_{A,B}; D_{A,C}; D_{B,C}; \ldots$ usadas para medir la disimilitud entre grupos. Cabe señalar que existen [diferentes distancias entre elementos y grupos](https://scikit-learn.org/stable/modules/clustering.html#different-linkage-type-ward-complete-average-and-single-linkage) que se pueden usar.

# In[38]:


from sklearn.cluster import AgglomerativeClustering


# In[39]:


clustering = AgglomerativeClustering(distance_threshold = 0, n_clusters = None)
clustering = clustering.fit(df_scaled.iloc[:10000])


# Estos resultados se pueden mostrar en dendogramas.

# In[40]:


# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# In[41]:


plt.figure()
plot_dendrogram(clustering, truncate_mode = "level", p = 3)
plt.xticks(rotation = 90)
# plt.savefig('dendogram.pdf')
plt.show()


# ## Tarea en clase (2 puntos)
# 
# - Investigar sobre algún otro algoritmo no supervisado que pueda usarse en tu código. Da un panorama sobre el modelo matemático que emplea y explica por qué conviene aplicarlo a tus datos. Algunos modelos que no se vieron en clase son *Affinity Propagation*, BIRCH, DBSCAN, *Mean Shift*, *Nearest Neighbors*, OPTICS, *Spectral Clustering*, TSNE, entre otros
# - Investigar otras estrategias para determinar número de grupos en estos algoritmos (como los índices de Calinski-Harabasz o de Davies-Bouldin), elegir la más adecuada al método que elegiste

# ## Tarea (8 puntos)
# 
# - Aplicar al menos un algoritmo no supervisado a tus datos para encontrar estructuras subyacentes
# - Elegir alguna métrica para determinar número de grupos, usarla y discutirla
# - Busca alguna revista científica que publique trabajos relacionados con el tuyo
# - Crea artículo mediante Latex con base en los lineamientos de la revista elegida y redacta ahí tus resultados, discusiones y bibliografía
# - Sube el código de tu tarea, los archivos de Latex y el PDF del artículo en tu repositorio, claramente diferenciados

# ## Recursos
# 
# - https://fesan818181.medium.com/unsupervised-learning-algorithms-explanaition-and-simple-code-b7f695a9e2cd
# - https://machinelearningmastery.com/clustering-algorithms-with-python/
# - https://medium.com/imagescv/top-8-most-important-unsupervised-machine-learning-algorithms-with-python-code-references-1222393b5077
# - https://builtin.com/data-science/unsupervised-learning-python
# - https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/
# - https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5
# - https://medium.com/analytics-vidhya/concept-of-gowers-distance-and-it-s-application-using-python-b08cf6139ac2
# - https://github.com/wwwjk366/gower
