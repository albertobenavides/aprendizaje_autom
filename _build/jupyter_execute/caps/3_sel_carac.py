#!/usr/bin/env python
# coding: utf-8

# # Selección de características
# 
# Técnica utilizada para eliminar características que sean estadísticamente redundantes o aporten poca información a los modelos. Como consecuencia, suelen disminuir tiempos de entrenamiento de modelos con muchas características e incluso mejorar sus resultados.

# ## Métodos de filtro
# 
# Usan estadísticos para determinar umbrales sobre los que elegir características. Suelen ser más rápidos que otros métodos, mas no suelen incluir interacción entre variables.

# ### ANOVA de valor $F$
# 
# Determina linealidad entre variables de entrada y salida. Un valor $F$ alto, indica alta relación lineal; valores menores, lo contrario.

# Primero cargamos los paquetes.

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')


# Luego los datos.

# In[2]:


df_pollutants_coords = pd.read_csv('../data/results/df_pollutants_coords.csv')
df_pollutants_coords.timestamp = pd.to_datetime(df_pollutants_coords.timestamp)
df_pollutants_coords.dropna(inplace = True)
df_pollutants_coords


# Para esta sesión, se busca explicar el contaminante $\text{PM}_{10}$ como $Y$ a partir de las variables atmosféricas $X$, independientes del tiempo y geolocalización.

# In[3]:


y = df_pollutants_coords[['PM10']]
y


# In[4]:


x = df_pollutants_coords[['BP', 'RF', 'RH', 'SR', 'T', 'WV', 'WD']]
x


# En Python se puede usar `f_regression` para calcular el valor $F$ en regresiones, mientras que `f_classif` se usa para clasificaciones.

# In[5]:


from sklearn.feature_selection import f_regression


# In[6]:


f_value = f_regression(x, y)
# Regresa arreglo de estadístico F y valor p
f_value


# Determinar si las variables aportan linealmente información relevante al modelo.

# In[7]:


pass_test = []
not_pass_test = []
alpha = 0.05
for i in range(len(f_value[1])):
    print(x.columns[i], f_value[1][i])
    if f_value[1][i] < alpha:
        pass_test.append(x.columns[i])
    else:
        not_pass_test.append(x.columns[i])


# In[8]:


df_results = pd.DataFrame(f_value[0], index=x.columns)
df_results.columns = ['f_value']
df_results.sort_values('f_value', inplace = True, ascending = False)
df_results


# Los resultados. Como recordatorio: Barra más alta, más linealidad con $\text{PM}_{10}$.

# In[9]:


plt.figure()
plt.bar(df_results.drop(not_pass_test).index, df_results.drop(not_pass_test).f_value)
plt.show()


# La humedad relativa, dirección del viento y radiación solar encabezan las variables más linealmente relacionadas con $\text{PM}_{10}$.

# ### Valor $R$ de correlación
# 
# La correlación ya se estudió en el capítulo pasado. También se puede utilizar como selección de características.

# In[10]:


from sklearn.feature_selection import r_regression


# In[11]:


r_value = r_regression(x, y)
r_value


# In[12]:


df_results['r_value'] = r_value
colors = []
for v in df_results['r_value']:
    if v > 0:
        colors.append('b')
    else:
        colors.append('r')


# In[13]:


df_results['r_value_abs'] = df_results['r_value'].abs()
df_results.sort_values('r_value_abs', inplace = True, ascending = False)
plt.figure()
plt.bar(df_results.index, df_results.r_value_abs, color = colors)
plt.show()


# ### Umbral de varianza
# 
# Otro modelo de filtro para selección de características es el umbral de varianza, que consiste en descartar características con baja varianza, en el supuesto de que no aportan tanta información al modelo. Requiere que las características estén normalizadas.

# In[14]:


# Normalización de variables
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(scaled, columns = x.columns)
x_scaled


# In[15]:


from sklearn.feature_selection import VarianceThreshold


# In[16]:


selector = VarianceThreshold()
selector.fit_transform(x_scaled)
selector.variances_


# In[17]:


# Se agregan las varianzas a los resultados
df_results['variance'] = selector.variances_
df_results.sort_values('variance', ascending = False, inplace = True)
df_results


# In[18]:


plt.figure()
plt.bar(df_results.index, df_results.variance)
plt.show()


# Generalmente, se suelen eliminar características con varianza menor a $0.2$. Aquí todas serían eliminadas 😅

# ### Información mutua
# 
# Este modelo mide la dependencia entre variables. Un valor de $0$ indicaría que las variables son independientes. Este modelo captura relaciones no lineales entre las variables 👀
# 
# Aquí también existe la variante `classif` para modelos que impliquen clasificación.

# In[19]:


from sklearn.feature_selection import mutual_info_regression


# In[20]:


mi = mutual_info_regression(x, y, random_state=0)
mi


# In[21]:


# Agregarlo a los resultados
df_results['mi'] = mi
df_results.sort_values('mi', ascending = False, inplace = True)

plt.figure()
plt.bar(df_results.index, df_results.mi)
plt.show()


# Una vez que se tienen algunos resultados, es recomendable utilizar una métrica para tomar una decisión. Hay muchas maneras de hacer esto, por ejemplo mediante la media de los valores normalizados. De esta manera, tenemos sólo una variable de decisión.

# In[22]:


scaled = scaler.fit_transform(df_results)
df_results_scaled = pd.DataFrame(scaled, columns = df_results.columns)
df_results_scaled.set_index(df_results.index, inplace = True)
df_results_scaled['norm_mean'] = df_results_scaled.mean(axis = 1)
df_results_scaled.sort_values('norm_mean', ascending = False, inplace = True)

plt.figure()
plt.bar(df_results_scaled.index, df_results_scaled.mean(axis = 1))
plt.show()


# ## Métodos de envoltura (?) o *wrapper*

# Métodos que exploran subconjuntos de combinaciones de características que mejoren algún desempeño de modelos de AA, con la ventaja de que, al usar un modelo, se estudian las relaciones de las carracterísticas *en* el modelo, a diferencia de los métodos de filtro, donde la relación de características dependía de estadísticos. Estos métodos tienen la desventaja de que, a mayor complejidad del modelo y número de características, mayor consumo de recursos y tiempos de ejecución ⌛

# ### Selección de características exhaustiva
# 
# La Selección de características exhaustiva o EFS (*Exhaustive Feature Selection*) por sus siglas en inglés evalúa todas las combinaciones de características y devuelve los valores que optimizan el modelo. Como ejemplo, se usa una regresión lineal.

# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


# In[25]:


lr = LinearRegression()

efs = EFS(estimator = lr,        # Use logistic regression as the classifier/estimator
          min_features = 1,      # The minimum number of features to consider is 1
          max_features = 7,      # The maximum number of features to consider is 4
          scoring = 'neg_mean_absolute_error',  # The metric to use to evaluate the classifier is accuracy 
          cv = 5)


# In[26]:


efs = efs.fit(x, y)


# In[27]:


print('Best accuracy score: %.2f' % efs.best_score_)
# print('Best subset (indices):', efs.best_idx_)
print('Best subset (corresponding names):', efs.best_feature_names_)


# In[28]:


metric_dict = efs.get_metric_dict()
df_efs = pd.DataFrame(metric_dict).T
df_efs.sort_values('avg_score', ascending=False,  inplace = True)
df_efs_best_10 = df_efs.iloc[:10]
df_efs_best_10


# In[29]:


# https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/#example-2-visualizing-the-feature-selection-results

fig = plt.figure(figsize=(20, 4))

plt.plot(
    df_efs_best_10.feature_names.astype(str), 
    df_efs_best_10.avg_score, 
    color='blue', marker='o'
)
plt.ylabel('MAE')
plt.xlabel('Características')

plt.xticks(rotation = 90)

plt.show()


# ### Sequential Forward Selection (SFS)
# 
# Este modelo agrega en cada iteración una variable e identifica las variables que mejoran la métrica del modelo.

# In[30]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[31]:


sfs = SFS(estimator = lr,        # Use logistic regression as the classifier/estimator
          k_features = (1, 7),  # Consider any feature combination between 1 and 8
          forward = True,       # Set forward to True when we want to perform SFS
          scoring = 'neg_mean_absolute_error',  # The metric to use to evaluate the classifier is accuracy 
          cv=5)


# In[32]:


sfs = sfs.fit(x, y)


# In[33]:


print('Best accuracy score: %.2f' % sfs.k_score_)   # k_score_ shows the best score 
print('Best subset (corresponding names):', sfs.k_feature_names_) # k_feature_names_ shows the feature names 
                                                                  # that yield the best score


# In[34]:


subsets_ = sfs.subsets_
df_sfs = pd.DataFrame(subsets_).T
df_sfs.sort_values('avg_score', ascending=False,  inplace = True)
df_sfs


# In[35]:


fig = plt.figure(figsize=(20, 4))

plt.plot(
    df_sfs.feature_names.astype(str), 
    df_sfs.avg_score, 
    color='blue', marker='o'
)
plt.ylabel('MAE')
plt.xlabel('Características')

plt.xticks(rotation = 90)

plt.show()


# ### Sequential Backward Selection (SBS)
# 
# Lo mismo, pero al revés 😛

# In[36]:


sbs = SFS(estimator = lr,
          k_features=(1, 7),
          forward=True,
          scoring='neg_mean_absolute_error',
          cv=5)

sbs = sbs.fit(x, y)
subsets_ = sbs.subsets_
df_sbs = pd.DataFrame(subsets_).T
df_sbs.sort_values('avg_score', ascending=False,  inplace = True)

fig = plt.figure(figsize=(20, 4))
plt.plot(
    df_sbs.feature_names.astype(str), 
    df_sbs.avg_score, 
    color='blue', marker='o'
)
plt.ylabel('MAE')
plt.xlabel('Características')

plt.xticks(rotation = 90)

plt.show()


# ## PCA
# 
# El [análisis de componentes principales](https://www.cienciadedatos.net/documentos/py19-pca-python.html) es una especie de técnica de reducción de características que podría utilizarse como selección de características. Consiste en reducir la dimensionalidad de características mediante hiperparámetros que incluyan las características que más varianza tengan para explicar un modelo.

# In[37]:


from sklearn.decomposition import PCA
import numpy as np


# In[38]:


pca = PCA(n_components = 3)
pca_model = pca.fit(x_scaled)


# In[39]:


plt.figure()
plt.bar(np.arange(pca_model.n_components_) + 1, pca_model.explained_variance_ratio_)
prop_varianza_acum = pca_model.explained_variance_ratio_.cumsum()
plt.plot(range(1, 4),prop_varianza_acum, marker = 'o', c='orange', label='Var. acumulada')
plt.xticks(np.arange(pca_model.n_components_) + 1)
plt.ylim(0, 1.1)
plt.xlabel('Componente principal', fontsize=12)
plt.ylabel('Varianza explicada', fontsize=16)
plt.legend()
plt.show()


# Para conocer los coeficientes que utiliza PCA para sus componentes, se puede hacer lo siguiente.

# In[40]:


# Coeficientes del PCA
pca_coef = pd.DataFrame(
  data    = pca_model.components_,
  columns = x_scaled.columns,
  index = ['pca1', 'pca2', 'pca3']
).T.sort_values('pca1', ascending=False)
pca_coef


# In[41]:


formula = ''
for i, r in pca_coef.iterrows():
    formula = formula +  str(round(r.pca1, 4)) + ' \text{' + i + '} + '
formula


# $$\text{PCA}_1 = 0.506 \cdot \text{SR} + 0.4066 \cdot \text{T} + 0.2737 \cdot \text{WV} + -0.0013 \cdot \text{RF} + -0.3749 \cdot \text{BP} + -0.4157 \cdot \text{RH} + -0.4362 \cdot \text{WD}.$$

# - [ ] Métodos embebidos

# ## Tarea en clase (2 puntos)
# 
# - Aplica algún método de filtro a tus datos mediante el uso de [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

# ## Tarea (8 puntos)
# 
# - Aplica los modelos de selección de características cuidando los supuestos de cada modelo
# - Busca una o varias métricas para seleccionar características en literatura relacionada con tu problema (cita tus fuentes)
# - Con base en tu investigación, determina las características más relevantes de tu conjunto de datos
# - Discute por qué crees que las características seleccionadas son las más relevantes y por qué el resto quedaron excluidas en la selección

# ## Referencias
# - https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# - https://neptune.ai/blog/feature-selection-methods
# - https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning
# - https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
# - https://scikit-learn.org/stable/modules/feature_selection.html
# - https://www.kaggle.com/code/ar2017/basics-of-feature-selection-with-python
# - https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/
# - https://towardsdatascience.com/beyond-linear-regression-467a7fc3bafb
# - https://github.com/AutoViML/featurewiz ⭐
