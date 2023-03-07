#!/usr/bin/env python
# coding: utf-8

# # Estadística descriptiva básica

# ## Estadísticos

# In[1]:


import pandas as pd


# In[2]:


df_sel = pd.read_csv('../data/results/df_sel.csv')
df_sel


# In[3]:


# Tipo de dato
type(df_sel)


# In[4]:


# Tipo de datos de las columnas
df_sel.dtypes


# In[5]:


# A dato temporal
df_sel.timestamp = pd.to_datetime(df_sel.timestamp)


# In[6]:


df_sel.dtypes


# Rápidamente se pueden encontrar descriptores estadísticos de los datos con `describe`.

# In[7]:


df_sel.describe()


# Pero esto no es muy informativo dado que hay diferentes variables, estaciones de monitoreo y tiempos de lectura. Para ello se puede separar el conjunto de datos en subconjuntos. Por ejemplo, por variable.

# In[8]:


gp_sel_variable = df_sel.groupby('variable')


# Esto se puede iterar (se muestra sólo una iteración como ejemplo).

# In[9]:


for name, group in gp_sel_variable:
    print(name)
    display(group)
    print('Etcétera')
    break


# También es posible filtrar resultados.

# In[10]:


df_pm10 = df_sel[df_sel.variable == 'PM10']


# Ya con esto, se pueden obtener el resumen estadístico de los datos.

# In[11]:


df_pm10.value.describe()


# ## Medidas de tendencia central

# ### Media
# 
# Suma de todos los elementos $x_i$ de conjunto $X$ dividida entre el número total de elementos $n$,
# 
# $$\sum_i^n x_i / n.$$

# Aunque se pueden usar funciones nativas de Python o integradas de `pandas` o `numpy` para los cálculos de los estadísticos, aprovecharemos esta sección para conocer la librería [`statistics`](https://docs.python.org/3/library/statistics.html).

# In[12]:


import statistics


# In[13]:


statistics.mean(df_pm10.value)


# Aquí tenemos un problema, pues aparece `nan` como resultado de la media. Esto sucede porque en el conjunto de registros de `value` hay valores no numéricos (_not a number_ o `nan`). Veamos cuántos `nan` hay.

# In[14]:


# Qué celdas son nan
df_pm10.isna()


# In[15]:


# Total de nan por columna
df_pm10.isna().sum(axis=0)


# In[16]:


# Total de nan por fila (aquí es trivial calcularlo)
df_pm10.isna().sum(axis=1)


# In[17]:


# Con esto vemos cuáles valores son nan
filtro = df_pm10.value.isna()
filtro


# In[18]:


# Esto nos devuelve las filas que son True en el filtro anterior
# O sea, todos los nan en value
df_pm10_nan = df_pm10[filtro]
df_pm10_nan


# Podemos contarlos.

# In[19]:


len(df_pm10_nan)


# También se puede encontrar la proporción de valores faltantes.

# In[20]:


len(df_pm10_nan) / len(df_pm10)


# Ahora, pueden realizarse varias estrategias para calcular la media de estos valores. La más común es ignorar los `nan`.

# In[21]:


# Elimina los valores nan
df_pm10.dropna()


# In[22]:


# Media sin nan
statistics.mean(df_pm10.value.dropna())


# In[23]:


# Cálculo de media más rápida para grandes conjuntos de datos
statistics.fmean(df_pm10.value.dropna())


# La **media ponderada** es una medida de tendencia central que da a los $n$ valores de un conjunto de datos $x$ un peso $w_i; i = [1, n]$, para luego calcular la media de los valores multiplicados por su peso asignado, tal que 
# 
# $$\frac{\sum_i w_i x_i}{\sum_i w_i}.$$
# 
# Es recomendable que $\sum_i w_i = 1$. En la librería `statistics`, la función `fmean` puede tomar como parámetro los pesos.

# In[24]:


df_pm10_dropna = df_pm10.dropna()


# Con `pandas` se pueden usar filtros para series de tiempo. En el siguiente ejemplo, se utiliza `dt` para acceder a las propiedades de la serie de tiempo. El parámetro `day_of_week` (hay más selectores) asigna `0` al lunes, `1` al martes... `6` al domingo.

# In[25]:


df_pm10_dropna[df_pm10_dropna.timestamp.dt.day_of_week == 6]


# Supongamos que queremos dar un peso por cada día de la semana, de manera incremental. El lunes pesaría $1$, el martes $2$... el domingo $7$. Como es recomendable que estos pesos sumen $1$, se puede hacer una función para obtener la porción equivalente.

# In[26]:


days_int = list(range(1, 8))
days_int_sum = sum(days_int)
for i in range(7):
    days_int[i] = days_int[i] / days_int_sum
sum(days_int)


# A partir de eso, se pueden asignar pesos a los días de la semana con un ciclo.

# In[27]:


df_pm10_dropna['w'] = -1 # Se añade una columna para los pesos
for i in range(7): # Para cada día de la semana
    df_pm10_dropna.loc[ # vamos a asignar
        df_pm10_dropna.timestamp.dt.day_of_week == i, # a las filas de ese día
        'w' # en la columna para los pesos
    ] = days_int[i] # el valor del peso normalizado correspondiente
df_pm10_dropna


# In[28]:


# Ahora ya se puede calcular esta media ponderada
statistics.fmean(df_pm10_dropna.value, weights = df_pm10_dropna.w)


# In[29]:


# Media armónica


# In[30]:


# Media geométrica


# ### Mediana
# 
# Tras ordenar un conjunto de $n$ datos, la mediana es el elemento que se encuentra en la posición $0.5 (n+1)$ si $n$ es impar, mientras que para $n$ par la mediana es la media de las posiciones $0.5 (n)$ y $0.5 (n+1)$.

# In[31]:


statistics.median(df_pm10_dropna.value)


# Exploremos la cantidad de valores $n$ de $\text{PM}_{10}$.

# In[32]:


len(df_pm10_dropna.value)


# Como $n$ es par, entonces podemos pedir los números sobre los que se calcula la media con

# In[33]:


statistics.median_low(df_pm10_dropna.value)


# In[34]:


statistics.median_high(df_pm10_dropna.value)


# Ja, de todas formas parece que hay varios $49$ por ahí. Lo veremos más adelante cuando hagamos representaciones visuales de estos datos.

# ### Moda
# 
# La moda representa el valor que más se repite en un conjunto de datos.

# In[35]:


statistics.mode(df_pm10_dropna.value)


# Si sólo hay un valor que represente la moda, los datos son unimodales, mientras que si hay empates los datos son multimodales (con dos o más valores que representan la moda). Eso se puede calcular también (aunque aquí nomás hay uno, jeje).

# In[36]:


statistics.multimode(df_pm10_dropna.value)


# ## Medidas de dispersión

# ### Varianza
# 
# La varianza de una muestra $s^2$ refleja lo alejados que se encuentran $n$ datos de un conjunto de datos $x$ de su media $\bar{x}$, o sea
# 
# $$s^2 = \frac{\sum_i (x_i - \bar{x})^2}{n-1}.$$

# In[37]:


statistics.variance(df_pm10_dropna.value)


# La varianza poblacional $\sigma^2$ se mide a partir de $n$ datos de una población $x$ y su distancia respecto a la media poblacional $\mu$, a saber
# 
# $$\sigma^2 = \frac{\sum_i (x_i - \mu)^2}{n}.$$

# In[38]:


statistics.pvariance(df_pm10_dropna.value)


# Cuando se tiene toda la población, se usa la varianza poblacional. En este caso, tenemos una muestra (quitamos los `nan`), así que se usaría la muestral.

# ### Desviación estándar
# 
# Raíz cuadrada de la varianza muestral y poblacional, que transforma el estadístico a la misma unidad de medida de los datos.

# In[39]:


statistics.stdev(df_pm10_dropna.value)


# In[40]:


statistics.pstdev(df_pm10_dropna.value)


# ### Sesgo (_bias_)
# Dirección de desviación de los datos respecto a la media.
# 
# $$\frac{\sum_i(x_i − \bar{x})^3 n}{(n − 1)(n − 2)s^3}.$$
# 
# Valores de sesgo negativos indican predominancia de elementos menores a la media, sesgo positivo son mayor cantidad de elementos mayores a la media, y sesgo cercano a $0$ (entre $-0.5, 0.5$) cuenta con elementos distribuidos simétricamente. No existe esta medida en `statistics`, pero sí en `pandas`.

# In[41]:


df_pm10_dropna.value.skew()


# ### Cuantiles
# Valores que representan divisiones de $n$ datos. Usualmente se dividen los datos en cuatro (cuartiles), diez (deciles), cien (percentiles), pero pueden dividirse por cualquir número $q$. Una vez determinadas las $q$ divisiones, se pueden calcular el cuantil $Q_k$ de una posición $k: 0 \leq k \leq q$ mediante
# $$Q_k = \frac{kn}{q}.$$

# In[42]:


# Esto nos da los cuartiles de la muestra
statistics.quantiles(df_pm10_dropna.value, n = 4)


# In[43]:


# Si tuviéramos una población, se usaría
statistics.quantiles(df_pm10_dropna.value, n = 10, method='inclusive')


# ### Rangos
# Mínimos y máximos.

# In[44]:


min(df_pm10_dropna.value)


# In[45]:


max(df_pm10_dropna.value)


# ## Visualización
# 
# Todos estos estadísticos pueden representarse en algunos gráficos. La librería [`matplotlib`](https://matplotlib.org/) permite generar gráficos en Python.

# In[46]:


import matplotlib.pyplot as plt


# Los histogramas permiten visualizar la distribución de los datos, agrupando el conteo de valores de datos en barras de rangos definidos.

# In[47]:


plt.figure(figsize = (6.4, 4.8)) # figura con tamaño por defecto
plt.hist(df_pm10_dropna.value, color='pink') # El histograma 🥳
plt.axvline( # Dibujar una línea vertical
    x = df_pm10_dropna.value.mean(), # En la media de los datos
    c = '#ff0000', # De color hexadecimal rojo
    label = '$\\bar{x}$' # con este identificador
)

plt.title('No me gusta añadir títulos')
plt.suptitle('Mucho menos súper títulos')

plt.xlabel('PM$_{10}$') # Título del eje horizontal
plt.ylabel('Frecuencia o conteo') # Título del eje vertical

plt.legend() # Muestra identificadores
plt.tight_layout() # Que quepa todo en la imagen guardada
plt.savefig('../imgs/histogram_example.png')
plt.show()


# Un diagrama de dispersión nos permite ver todos los datos como puntos individuales.

# In[48]:


plt.figure(figsize = (20, 4)) # figura con tamaño horizontal en 20 y vertical 4
plt.scatter(
    df_pm10_dropna.timestamp, # Valores del eje horizontal
    df_pm10_dropna.value, # del eje vertical
    marker='.' # y tipo de marcador
)
plt.show()


# Como esto es una serie de tiempo, tiene más sentido verla como un gráfico lineal.

# In[49]:


plt.figure(figsize = (20, 4)) # figura con tamaño horizontal en 20 y vertical 4
plt.plot(df_pm10_dropna.timestamp, df_pm10_dropna.value)
plt.show()


# Se ve feooooo 🤮, pero se puede arreglar ordenando los valores por `timestamp`. También se puede poner como índice, para no batallar.

# In[50]:


df_pm10_dropna.sort_values('timestamp', inplace=True)
df_pm10_dropna.reset_index(inplace=True, drop=True)
df_pm10_dropna


# In[51]:


plt.figure(figsize = (20, 4)) # figura con tamaño horizontal en 20 y vertical 4
plt.plot(df_pm10_dropna.timestamp, df_pm10_dropna.value,
    linestyle = 'dashed',
    linewidth = 0.5 # Ancho de línea
)
plt.show()


# Las gráficas de barras muestran información agrupada en rangos ordenados según un criterio.

# In[52]:


# Reajuste a media por semana
t = df_pm10_dropna.resample('W', on='timestamp').mean()
t


# In[53]:


plt.figure(figsize = (20, 4)) # figura con tamaño horizontal en 20 y vertical 4
plt.bar(t.index, t.value, width=5)
plt.show()


# Un diagrama de cajas y bigotes (o _boxplot_ en inglés) muestra los valores mínimo, máximo, primer cuartil, segundo cuartil (que corresponde con la mediana), tercer cuartil y _outliers_.

# In[54]:


plt.figure(figsize=(20, 4))
plt.boxplot(df_pm10_dropna.value, vert = False)

ax = plt.gca() # Esto separa los ejes de la figura
ax.annotate(
    'Mínimo', 
    xy=(min(df_pm10_dropna.value), 1.05), 
    xytext=(min(df_pm10_dropna.value), 1.3), 
    ha="center",
    arrowprops=dict(
        facecolor = 'black', 
        width = 0.5,
        headwidth = 6,
        connectionstyle="arc3, rad = -0.01"
    )
)

ax.annotate(
    '$Q_1$', 
    xy=(df_pm10_dropna.value.quantile(.25), 0.9), 
    xytext=(df_pm10_dropna.value.quantile(.25), 0.7), 
    ha="center",
    arrowprops=dict(
        facecolor = 'black', 
        width = 0.5,
        headwidth = 6,
        connectionstyle="arc3,rad=-0.01"
    )
)

ax.annotate(
    '$Q_2 =$ Mediana', 
    xy=(df_pm10_dropna.value.quantile(.5), 1.1), 
    xytext=(df_pm10_dropna.value.quantile(.5), 1.3), 
    ha="center",
    arrowprops=dict(
        facecolor = 'black', 
        width = 0.5,
        headwidth = 6,
        connectionstyle="arc3,rad=-0.01"
    )
)

ax.annotate(
    '$Q_3$', 
    xy=(df_pm10_dropna.value.quantile(.75), 0.9), 
    xytext=(df_pm10_dropna.value.quantile(.75), 0.7), 
    ha="center",
    arrowprops=dict(
        facecolor = 'black', 
        width = 0.5,
        headwidth = 6,
        connectionstyle="arc3,rad=-0.01"
    )
)

ax.annotate(
    '$Máximo$', 
    xy=(max(df_pm10_dropna.value), 1.1), 
    xytext=(max(df_pm10_dropna.value), 1.3), 
    ha="center",
    arrowprops=dict(
        facecolor = 'black', 
        width = 0.5,
        headwidth = 6,
        connectionstyle="arc3,rad=-0.01"
    )
)
plt.show()


# Cuando se tienen tantos valores atípicos, es mejor usar un diagrama de violín, que muestra la distribución de los datos espejeada, sobre los valores mínimos y máximos.

# In[55]:


plt.figure()
plt.violinplot(df_pm10_dropna.value, vert=False)
plt.axvline( # Dibujar una línea vertical
    x = df_pm10_dropna.value.mean(), # En la media de los datos
    c = '#ff0000', # De color hexadecimal rojo
    label = '$\\bar{x}$' # con este identificador
)
plt.yticks([1], labels=['PM$_{10}$'])
plt.show()


# Para comparar distribuciones, me agradan los diagramas de surcos (o _ridgelines_), a los que también les llaman _Joyplots_ por la portada del álbum [_Unknown pleasures_ de Joy Division](https://www.indierocks.mx/musica/articulos/a-40-anos-del-unknown-pleasures-de-joy-division/).

# In[56]:


# https://www.analyticsvidhya.com/blog/2021/06/ridgeline-plots-visualize-data-with-a-joy/
from joypy import joyplot


# In[57]:


plt.figure()
joyplot(df_pm10_dropna, by='lat', column='value')
plt.show()


# Para series de tiempo, las vistas de calendario son geniales.

# In[58]:


import calplot


# In[59]:


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np


# In[60]:


viridis = cm.get_cmap('viridis', 340) # Se genera un mapa de color de hasta 340 colores
newcolors = viridis(np.linspace(0, 1, 340)) # Se hace una función lineal del mapa de colores de 0 hasta 340 de 1 en 1

# Se definen rangos de colores por RGBA normalizado
newcolors[:50, :] = np.array([21/340, 176/340, 26/340, 1]) # Bueno
newcolors[50:75, :] = np.array([255/340, 255/340, 20/340, 1]) # Regular
newcolors[75:155, :] = np.array([249/340, 115/340, 6/340, 1]) # Malo
newcolors[155:235, :] = np.array([299/340, 0/340, 0/340, 1]) # Muy malo
newcolors[235:340, :] = np.array([126/340, 30/340, 156/340, 1]) # Extremadamente malo
newcmp = ListedColormap(newcolors) # A partir de esos colores, se fija un nuevo mapa de color

calplot.calplot(df_pm10_dropna.set_index('timestamp').value, cmap = newcmp, 
    yearlabel_kws=dict( # Esto es nomás porque no tengo Helvetica instalada 😅
        fontsize=30,
        color='gray',
        fontname='sans-serif',
        fontweight='bold',
        ha='center')
)


# ## Pruebas de correlación
# 
# La correlación $\rho$ mide la relación lineal entre dos variables $x, y$ normalizadas por sus desviaciones estándar $s_x, s_y$. Se calcula mediante
# $$\rho = \frac{\sum(x - \bar{x}) (y - \bar{y})}{s_x s_y}.$$

# In[61]:


df_sel.columns


# In[62]:


df_by_cols = df_sel.pivot_table('value', ['timestamp', 'lat', 'lon', 'h'], 'variable')
df_by_cols


# In[63]:


corr_ = df_by_cols[['PM10', 'PM2_5']].corr()
corr_


# Para este tipo de representaciones, un mapa de calor funciona bastante bien.

# In[64]:


plt.figure()
plt.imshow(corr_)
plt.xticks(ticks=[0, 1], labels=['PM$_{10}$', 'PM$_{2.5}$'])
plt.yticks(ticks=[0, 1], labels=['PM$_{10}$', 'PM$_{2.5}$'])
plt.clim(vmin = -1, vmax = 1)
plt.colorbar()
plt.show()


# In[65]:


corr_ = df_by_cols.corr()
ticks_ = list(range(len(df_by_cols.columns)))
plt.figure()
plt.imshow(corr_)
plt.xticks(ticks = ticks_, labels = df_by_cols.columns, rotation = 90)
plt.yticks(ticks = ticks_, labels = df_by_cols.columns)
plt.clim(vmin = -1, vmax = 1)
plt.colorbar()
plt.show()


# ## Pruebas de normalidad
# 
# Los datos pueden seguir una distribución gaussiana o normal. A este tipo de datos se les llama datos paramétricos. Cuando no siguen esta distribución, se consideran no paramétricos.

# In[66]:


# Generar datos aleatorios
mu = 10
sigma = 2
n = 1000
r_norm = np.random.normal(mu, sigma, 10000)


# In[67]:


statistics.mean(r_norm)


# In[68]:


statistics.stdev(r_norm)


# Se mantienen relativamente constantes los valores de media y desviación estándar. Visualmente se puede comprobar con histogramas.

# In[69]:


# Nos ayudamos de la librería scipy para generar una función de distribución de probabilidad o pdf
from scipy.stats import norm


# In[70]:


plt.figure()
plt.hist(r_norm, density=True)

plt.plot(np.arange(0, 20, 0.1), norm.pdf(np.arange(0, 20, 0.1), mu, sigma), label = 'Curva gaussiana')
plt.legend()
plt.show()


# También los diagramas de cuantil-cuantil son útiles para intuir la normalidad de los datos. Mientras la relación de los cuantiles teóricos y los de la muestra se mantenga sobre la regresión lineal, se considera que los datos son normales.

# In[71]:


from statsmodels.graphics.gofplots import qqplot


# In[72]:


plt.figure()
qqplot(r_norm, line='s')
plt.show()


# Formalmente, esto se realiza con pruebas de normalidad estadísticas. El más usado es el de Shapiro-Wilk, con la salvedad de que se usa en muestras de pocos datos (mil observaciones o menos). La hipótesis nula $H_0$ de esta prueba supone que una muestra $x$ proviene de una distribución de probabilidad normalmente distribuida. La prueba se rechaza si el estadístico (con valores entre 0 y 1) es demasiado pequeño (o si el valor $p$ es mejor que determinado valor $\alpha$ generalmente $\alpha = 0.05$).

# In[73]:


from scipy.stats import shapiro


# In[74]:


stat, p = shapiro(r_norm)
alpha = 0.05
if p < alpha:
    print(f'La distribución no parece normal con p = {round(p, 4)}')
else:
    print(f'La distribución parece normal con p = {round(p, 4)} ')


# Cuando los datos siguen una distribución normal, se pueden usar [pruebas estadísticas paramétricas](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/) para realizar inferencias sobre ellos. De otra manera, se usan pruebas [no paramétricas](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/).

# ## Tarea en clase (2 puntos)
# 
# - Crear funciones para calcular las medidas de tendencia central y de dispersión con Python puro (sin librerías)
# - Comprueba si tus variables de interés son conjuntos de datos paramétricos o no paramétricos

# ## Tarea (8 puntos)
# 
# - Calcula estadísticos descriptivos básicos para tus datos
# - Haz una matriz de correlación de tus datos y escribe algunas interpretaciones de la misma
# - Realiza alguna prueba de hipótesis a partir de las conclusiones que hayas sacado de la matriz de correlación
# - Presenta tus resultados gráficamente

# - https://realpython.com/python-statistics/#calculating-descriptive-statistics
# - https://machinelearningmastery.com/statistical-hypothesis-tests/
# - https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
