#!/usr/bin/env python
# coding: utf-8

# # Datos
# 
# Parte fundamental de la ciencia de datos consiste en importar información y preprocesarla. La carga de información puede ser por varios métodos. El **preprocesamiento** suele tomar considerable tiempo en el proceso de análisis de datos. Éste proceso consiste en modificar la información recibida de tal manera que esté lista para utilizarse para posteriores análisis.
# 
# Como ejemplo para el curso, se usa un conjunto de datos georrerefenciados por trece estaciones de monitoreo del Área Metropolitana de Monterrey durante 2017 y 2019. Estas estaciones recogen datos meteorológicos y de contaminantes ambientales cada hora por parte del [Sistema Integral de Monitoreo Ambiental](http://aire.nl.gob.mx/).
# 
# A continuación se muestran los valores que se registran.

# In[1]:


import math


# In[2]:


# Arreglo de diccionarios
pollutants = [
  {
    'name' : '$\\text{PM}_{10}$',
    'min_value' : 2,
    'max_value' : 850,
    'unit' : '$\mu\\text{gr}$ / $\\text{m}^3$'
  },
  {
    'name' : '$\\text{PM}_{2.5}$',
    'min_value' : 2,
    'max_value' : 850,
    'unit' : '$\mu\\text{gr}$ / $\\text{m}^3$'
  },
  {
    'name' : '$\\text{O}_3$',
    'min_value' : 1,
    'max_value' : 200,
    'unit' : 'ppb'
  },
  {
    'name' : 'NO',
    'min_value' : 1,
    'max_value' : 350,
    'unit' : 'ppb'
  },
  {
    'name' : '$\\text{NO}_2$',
    'min_value' : 1,
    'max_value' : 150,
    'unit' : 'ppb'
  },
  {
    'name' : '$\\text{NO}_X$',
    'min_value' : 1,
    'max_value' : 350,
    'unit' : 'ppb'
  },
  {
    'name' : '$\\text{SO}_2$',
    'min_value' : 1,
    'max_value' : 200,
    'unit' : 'ppb'
  },
  {
    'name' : 'CO',
    'min_value' : 0.050,
    'max_value' : 15,
    'unit' : 'ppm'
  },
  {
    'name' : 'Temperature',
    'min_value' : -15,
    'max_value' : 50,
    'unit' : 'C'
  },
  {
    'name' : 'Relative humidity',
    'min_value' : 0,
    'max_value' : 100,
    'unit' : '%'
  },
  {
    'name' : 'Barometric pressure',
    'min_value' : 650,
    'max_value' : 750,
    'unit' : 'mmHg'
  },
  {
    'name' : 'Solar radiation',
    'min_value' : 0,
    'max_value' : 1.2,
    'unit' : 'Langley / h'
  },
  {
    'name' : 'Rainfall',
    'min_value' : 0,
    'max_value' : math.inf,
    'unit' : 'mm / h'
  },
  {
    'name' : 'Wind velocity',
    'min_value' : 0,
    'max_value' : 60,
    'unit' : 'Km / h'
  },
  {
    'name' : 'Wind direction',
    'min_value' : 0,
    'max_value' : 360,
    'unit' : 'Azimutales'
  },
]


# [`pandas`](https://pandas.pydata.org/) es una librería de Python que permite manipular fuentes de datos como tablas (o _dataframes_).

# In[3]:


import pandas as pd

# Permite mostrar Markdown como salida en Jupyter
# https://stackoverflow.com/a/36313217
from IPython.display import Markdown


# In[4]:


df_pollutants_def = pd.DataFrame(pollutants)
Markdown(df_pollutants_def.to_markdown(index = False))


# El objetivo de este estudio radica en hacer interpolación espacio temporal mediante métodos de aprendizaje automático.

# Un archivo CSV tiene una parte de la base de datos, que se cargan con `pd.read_csv`.
# 
# Como una columna de datos contiene información temporal en un formato que no se apega a los estándares de `pandas`, hace falta transformarla. Se usa la librería [`datetime`](https://docs.python.org/3/library/datetime.html) y aprovechamos de usar el maravilloso `lambda`.

# In[5]:


# https://towardsdatascience.com/4-tricks-you-should-know-to-parse-date-columns-with-pandas-read-csv-27355bb2ad0e
from datetime import datetime
custom_date_parser = lambda x: datetime.strptime(x, '%d-%b-%y %H')


# In[6]:


df_pollutants_all = pd.read_csv("../data/all.csv",
  # Especifica qué columnas se van a leer
  usecols=[
    'timestamp',
    'station',
    'CO',
    'NO',
    'NO2',
    'NOX',
    'O3',
    'PM10',
    'PM2_5',
    'pressure',
    'rainfall',
    'humidity',
    'SO2',
    'solar',
    'temperature',
    'velocity',
    'direction',
    'valid',
  ],
  # Qué tipo de dato corresponde a cada columna
  dtype={
    'timestamp' : str,
    'station' : str,
    'CO' : float,
    'NO' : float,
    'NO2' : float,
    'NOX' : float,
    'O3' : float,
    'PM10' : float,
    'PM2_5' : float,
    'pressure' : float,
    'rainfall' : float,
    'humidity' : float,
    'SO2' : float,
    'solar' : float,
    'temperature' : float,
    'velocity' : float,
    'direction' : float,
    'valid' : int,
  },
  # En la columna timestamp
  parse_dates = ['timestamp'],
  # Guardar las fechas preprocesadas
  date_parser = custom_date_parser
)


# In[7]:


# Total de registros
len(df_pollutants_all)


# In[8]:


# Primeros datos
df_pollutants_all.head()


# In[9]:


# Últimos registros; en ambos se pueden especificar cuántos mostrar
df_pollutants_all.tail(3)


# In[10]:


# Muestra de 4 individuos sin reemplazo
df_pollutants_all.sample(4, replace = False)


# Los datos de `df_pollutants_all` van de 1993 a 2018. En un archivo XLSX se encuentran los datos de 2019.

# In[11]:


df_pollutants_2019 = pd.read_excel("../data/2019_pollutants.xlsx")
df_pollutants_2019


# Esta base de datos tiene la información organizada de una manera muy distinta a como aparece en `df_pollutants_all`. Las primeras dos columnas muetran la fecha del registro. Las siguientes llevan el nombre de la estación donde se hace el resgistro, seguida por un número entero. La primera fila muestra, para cada una de esas estaciones, la variable que están registrando. Hay que lidiar con todo eso.

# In[12]:


# Columnas
df_pollutants_2019.columns


# In[13]:


# Columnas que contienen Unnamed
unnamed = [c for c in df_pollutants_2019.columns if 'Unnamed' in c]
unnamed


# In[14]:


# Filtro para mostrar sólo columnas Unnamed
df_pollutants_2019[unnamed]


# Se comparan los valores entre la columna con índice 0 y la 1.

# In[15]:


# Selector de filas y columnas por índice
df_pollutants_2019.iloc[:, 0:2] # [filas : columnas]


# In[16]:


# Filas que son diferentes entre las columnas 0 y 1
filtro = df_pollutants_2019.iloc[:, 0] != df_pollutants_2019.iloc[:, 1]
filtro


# In[17]:


# Filtro a partir de comparación booleana
df_pollutants_2019[filtro]


# Se copia la fila $6312$ de la columna con índice 1 a la 0.

# In[18]:


df_pollutants_2019.iloc[6312, 1]


# Antes de modificar una tabla, recomiendo hacer una copia de la misma.

# In[19]:


# Copiar dataframe
df_pollutants_2019_mod = df_pollutants_2019.copy()


# In[20]:


# Copiar valores por posición
df_pollutants_2019_mod.iloc[6312, 0] = df_pollutants_2019_mod.iloc[6312, 1]
df_pollutants_2019_mod.iloc[6312, 0:2]


# In[21]:


# No nulos por columna
df_pollutants_2019_mod[unnamed].count()


# In[22]:


# Se eliminan las columnas con Unnamed
df_pollutants_2019_mod = df_pollutants_2019.drop(columns = unnamed)
df_pollutants_2019_mod.head()


# La fila 1 sólo aporta la unidad en que se mide la variable de los datos, así que se elimina.

# In[23]:


# Elimina fila por índice
df_pollutants_2019_mod = df_pollutants_2019_mod.drop(1)
df_pollutants_2019_mod.head()


# In[24]:


# Separación de cadena por .
df_pollutants_2019_mod.columns[2].split('.')


# In[25]:


columnas = list( # 2: A lista
    set( # 1: Sin repetidos
        [x.split('.')[0].title() for x in list(df_pollutants_2019_mod.columns)] # 0: Todos los nombres de columna antes del punto
    )
)
columnas


# In[26]:


# De ahí se eliminan la columna del tiempo
columnas.remove('Ce-Met')
sorted(columnas)


# In[27]:


# Se crea una base de datos vacía
df_pollutants_2019_def = pd.DataFrame()
for c in columnas:
  # Devuelve todas las columnas con el nombre de la estacion
  estacion = [col for col in df_pollutants_2019.columns if c == col.split('.')[0].title()]
  estacion.append('CE-MET')
  # Se toman todas las columnas de esa estación
  t = df_pollutants_2019[estacion].copy()
  # Se ponen como encabezados las variables
  t.columns = t.iloc[0]
  # Se quita la fila de encabezados que fue promovida
  t = t.drop(0)
  # y la de unidades de medida
  t = t.drop(1)
  # Se agrega la estacion y se eliminan espacios entre números y nombre
  t['station'] = c.replace(" ", "")
  # Se reinician los índices
  t = t.reset_index(drop=True)
  # Se agregan al df nuevo
  df_pollutants_2019_def = pd.concat([t, df_pollutants_2019_def], ignore_index = True)
df_pollutants_2019_def


# `WDV` y `WDR` miden grados, por lo que se pueden combinar.

# In[28]:


# WDV no es nulo mientras WDR es nulo
filtro = (~df_pollutants_2019_def.WDV.isna()) & (df_pollutants_2019_def.WDR.isna())
# Cantidad de True
filtro.sum()


# In[29]:


# Copiar en WDR los valores de WDV
df_pollutants_2019_def.loc[filtro, 'WDR'] = df_pollutants_2019_def.loc[filtro, 'WDV']


# In[30]:


# Eliminar columna en sitio
df_pollutants_2019_def.drop(columns = 'WDV', inplace = True)
df_pollutants_2019_def.head()


# Se emparejan los nombres de columnas.

# In[31]:


df_pollutants_2019_def.columns


# In[32]:


df_pollutants_all.columns


# In[33]:


df_pollutants_all.columns = [
    'timestamp', 'station', 'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2_5',
    'BP', 'RF', 'RH', 'SO2', 'SR', 'T',
    'WV', 'WD', 'valid'
]
df_pollutants_all.head()


# In[34]:


df_pollutants_2019_def.columns = [
    'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2_5', 'BP', 'RF', 'RH',
       'SO2', 'SR', 'T', 'WV', 'timestamp', 'station', 'WD'
]
df_pollutants_2019_def.head()


# In[35]:


# Unir dos dataframes
df_pollutants = pd.concat([df_pollutants_all.drop(columns = ['valid']), df_pollutants_2019_def])
df_pollutants.head()


# In[36]:


# Guardar registros con tiempo
df_pollutants = df_pollutants[~df_pollutants.timestamp.isna()]


# In[37]:


# Convertir a tipo datetime
df_pollutants[df_pollutants.columns[0]] = pd.to_datetime(df_pollutants.timestamp)


# In[38]:


# Primer y último registro
df_pollutants.iloc[[0, -1], ]


# Valores fuera de rango

# In[39]:


df_pollutants_def


# In[40]:


# Nombre de variables de sensores
sensor_variables = ['PM10', 'PM2_5', 'O3', 'NO', 'NO2', 'NOX','SO2', 'CO', 'T', 'RH', 'BP', 'SR', 'RF', 'WV', 'WD']


# In[41]:


# Agregar nombre de variable en df_pollutants_def
df_pollutants_def['var_name'] = sensor_variables
df_pollutants_def


# [`numpy`](https://numpy.org/) es una librería de Python para realizar operaciones matemáticas con arreglos y matrices.

# In[42]:


import numpy as np


# In[43]:


# Copia de dataframe
df_pollutants_copy = df_pollutants.copy()
# Contador de valores fuera de rango
errors = 0
for l, s in df_pollutants_def.iterrows():
    # Valores por debajo
    errors += len(df_pollutants_copy.loc[df_pollutants_copy[s.var_name] < s.min_value])
    errors += len(df_pollutants_copy.loc[df_pollutants_copy[s.var_name] > s.max_value])
    # Se usa np.nan para dejar vacíos esos lugares con valores fuera de rango
    df_pollutants_copy.loc[df_pollutants_copy[s.var_name] < s.min_value, s.var_name] = np.nan
    df_pollutants_copy.loc[df_pollutants_copy[s.var_name] > s.max_value, s.var_name] = np.nan
errors


# In[44]:


df_pollutants_copy


# Visualización

# In[45]:


import matplotlib.pyplot as plt


# In[46]:


t = df_pollutants_copy.groupby('station')['timestamp']
cols = ['station', 'min', 'max']
rangos = pd.DataFrame(columns=cols)

i = 0
for a, b in t:
  rangos.loc[i] = [a, min(b), max(b)]
  i += 1
  
fig, ax = plt.subplots(figsize=(15,6))
cmap = plt.cm.tab20
for i, r in rangos.iterrows():
  ax.broken_barh([(rangos.iloc[i, 1], rangos.iloc[i, 2] - rangos.iloc[i, 1])], [i, 0.5], facecolors=cmap(i), label=rangos.iloc[i, 0])

ax.set_yticks([])
plt.xticks(pd.date_range(start="1993-01-01", end="2020-01-01", normalize = True, freq='YS'), labels=range(1993, 2021))
plt.grid(axis='x')
plt.legend(bbox_to_anchor=(1.12, 0.8))
plt.gca().invert_yaxis()
plt.show()


# Se selecciona un rango con el que trabajar. En este caso, de 01-01-2017 a 31-12-2019.

# In[47]:


df_pollutants_2017 = df_pollutants[df_pollutants.timestamp.dt.year >= 2017]
df_pollutants_2017


# In[48]:


def imputations(cols, df):
  t = df.groupby('station')

  for c in cols:
    i = 0
    fig, ax = plt.subplots(figsize=(16,6))
    cmap = plt.cm.tab20
    for station, ts in t:
      # https://stackoverflow.com/questions/21402384/how-to-split-a-pandas-time-series-by-nan-values/66015224#66015224
      # Convert to sparse then query index to find block locations
      # different way of converting to sparse in pandas>=0.25.0
      sparse_ts = ts[c].astype(pd.SparseDtype('float'))
      # we need to use .values.sp_index.to_block_index() in this version of pandas
      block_locs = zip(
          sparse_ts.values.sp_index.to_block_index().blocs,
          sparse_ts.values.sp_index.to_block_index().blengths,
      )
      # Map the sparse blocks back to the dense timeseries
      blocks = [
          ts.iloc[start : (start + length - 1)]
          for (start, length) in block_locs
      ]

      j = 0
      for block in blocks:
        if block.empty:
          continue
        if j == 0:
          ax.broken_barh([(min(block.timestamp), max(block.timestamp) - min(block.timestamp))], [i, 0.5], facecolors=cmap(i), label = station)
          j += 1
        else:
          ax.broken_barh([(min(block.timestamp), max(block.timestamp) - min(block.timestamp))], [i, 0.5], facecolors=cmap(i))
      i += 1
    
    print(c)
    ax.set_yticks([])
    plt.xticks(pd.date_range(start="2017-01-01", end="2020-01-01", normalize = True, freq='YS'), labels=range(2017, 2021))
    plt.grid(axis='x')
    plt.legend(bbox_to_anchor=(1.12, 0.8))
    plt.gca().invert_yaxis()
    plt.show()


# In[49]:


imputations(['CO'], df_pollutants_2017)


# Ahora se agrega información geográfica de las estaciones de monitoreo.

# In[50]:


df_stations = pd.read_csv('../data/estaciones_coords.csv')
df_stations


# In[51]:


df_pollutants_coords = df_stations.merge(
    df_pollutants_2017,
    # unir DataFrames de datos con coordenadas
    on = 'station'
)
df_pollutants_coords = df_pollutants_coords.drop(columns = ['c'])
df_pollutants_coords.head()


# Dada la naturaleza de los datos, se sabe que $\text{NO}_x$ es la suma de $\text{NO}$ y $\text{NO}_2$. Se rellenan valores de alguna de estas variables, con la suma o resta de las dos restantes.

# In[52]:


# Conteo de vacíos
nox_na = df_pollutants_coords[['NO', 'NO2', 'NOX']].isna().sum()
nox_na


# In[53]:


# NOx vacíos
df_nox_na = df_pollutants_coords[(~df_pollutants_coords.NO.isna()) & (~df_pollutants_coords.NO2.isna()) & (df_pollutants_coords.NOX.isna())]
# NOx = NO + NO2
df_pollutants_coords.loc[df_nox_na.index, 'NOX'] = df_pollutants_coords.loc[df_nox_na.index, 'NO'] + df_pollutants_coords.loc[df_nox_na.index, 'NO2']


# In[54]:


# Con NO
df_no_na = df_pollutants_coords[(~df_pollutants_coords.NOX.isna()) & (~df_pollutants_coords.NO2.isna()) & (df_pollutants_coords.NO.isna())]
# NO = NOx - NO2
df_pollutants_coords.loc[df_no_na.index, 'NO'] = df_pollutants_coords.loc[df_no_na.index, 'NOX'] - df_pollutants_coords.loc[df_no_na.index, 'NO2']


# In[55]:


# Y NO2
df_no2_na = df_pollutants_coords[(~df_pollutants_coords.NOX.isna()) & (~df_pollutants_coords.NO.isna()) & (df_pollutants_coords.NO2.isna())]
# NO2 = NOx - NO
df_pollutants_coords.loc[df_no2_na.index, 'NO2'] = df_pollutants_coords.loc[df_no2_na.index, 'NOX'] - df_pollutants_coords.loc[df_no2_na.index, 'NO']


# In[56]:


# Reconstrucción
nox_na_new = df_pollutants_coords[['NO', 'NO2', 'NOX']].isna().sum()
nox_na - nox_na_new


# Ahora, se convierten todas las columnas de variables y sus valores, en una columna llamada variable y otra llamada valor, que se asignan a cada una de ellas, con los demás atributos.

# In[57]:


# https://pandas.pydata.org/docs/reference/api/pandas.melt.html
df_pollutants_melt = pd.melt(
    df_pollutants_coords, 
    id_vars=['timestamp', 'station', 'abbr', 'lat', 'lon', 'h'], # Variables que se mantienen
    value_vars = df_pollutants_def.var_name # Variables que se compactan
)
df_pollutants_melt


# Para interpolar, no hace falta el nombre de la estación ni su abreviatura.

# In[58]:


df_sel = df_pollutants_melt[['timestamp', 'lat', 'lon', 'h', 'variable', 'value']]
df_sel


# Más preprocesamiento, pero ahora enfocado a los métodos que nos interesan. Como los métodos de AA no trabajan con fechas, se convierten las fechas a enteros basados en la medida de [inicio de época de Unix](https://docs.python.org/3/library/time.html).

# In[59]:


# https://stackoverflow.com/a/54312941
# https://stackoverflow.com/a/51245631
df_sel.timestamp = df_sel.timestamp.values.astype(np.int64) / 10 ** 9
df_sel


# Ahora, para tomar en cuenta las variables como parámetros de los modelos, se utiliza el método de one-hot endoding.

# In[60]:


# https://towardsdatascience.com/ways-to-handle-categorical-data-before-train-ml-models-with-implementation-ffc213dc84ec
df_sel = pd.concat([df_sel, pd.get_dummies(df_sel.variable)], axis=1)
df_sel = df_sel.drop(columns= ['variable'])
df_sel


# Muchos de los algoritmos de AA tienen como supuestos que los datos estén normalizados. Existen muchas maneras de hacerlo. Aquí se usa [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), que devuelve cada columna a un rango $[0, 1] \in \mathbb{R}$,

# In[61]:


from sklearn.preprocessing import MinMaxScaler


# In[62]:


scaler = MinMaxScaler()
# https://stackoverflow.com/a/43383700
scaled = scaler.fit_transform(df_sel)
# https://datatofish.com/numpy-array-to-pandas-dataframe/
df_scaled = pd.DataFrame(scaled, columns = df_sel.columns)
df_scaled


# Selección de valores no vacíos.

# In[63]:


df_dropna = df_scaled.dropna()
df_dropna


# Separación en conjuntos de entrenamiento y prueba.

# Entrenamiento

# In[64]:


# https://stackoverflow.com/a/35531218
df_train = df_dropna.sample(frac = 0.7)
x_train = df_train[df_pollutants_def.var_name]
y_train = df_train[['value']]


# Prueba

# In[65]:


df_test = df_dropna.drop(df_train.index)

x_test = df_test[df_pollutants_def.var_name]
y_test = df_test[['value']]

