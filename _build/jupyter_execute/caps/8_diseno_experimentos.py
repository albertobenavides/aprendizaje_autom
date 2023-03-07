#!/usr/bin/env python
# coding: utf-8

# # Diseño de experimentos
# 
# 

# El diseño de experimentos consiste en idear una manera de generar nuevos conocimientos sobre relaciones causa-efecto entre variables de un problema científico. 
# 
# En este contexto, un **experimento** representa un cambio en las condiciones del problema de estudio. En el problema hay **variables independientes** y **variables dependientes**. Para el diseño de experimentos, también se consideran **niveles** y **tratamientos** de estas variables. Los niveles son los valores que toman las variables independientes, mientras que los tratamientos son los valores únicos que se asignan a cada combinación de niveles en el problema.
# 
# Por ejemplo, supongamos que se tiene un conjunto de datos como el que sigue.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.DataFrame({
    "x1": np.random.randint(0, 3, size=20),
    'x2' : np.random.randint(0, 360, size=20)
})
df['y'] = df.x1 * np.deg2rad(df.x2) + np.random.random()
df


# Donde $x_1$ corresponde con valores aleatorios en el conjunto $\{0, 1, 2\}$, mientras que los valoes que puede tomar $x_2$ se encuentran en el rango $[0, 360] \in \mathbb{N}$. Para este caso, los niveles para $x_1$ podrían ser los valores $\{0, 1, 2\}$. Sin embargo, para $x_2$, se pueden especificar niveles en rangos que vayan en incrementos de cada $10, 20, 45\ldots$ La siguiente tabla muestra un ejemplo de estos niveles asociados a sus tratamientos.

# In[3]:


niveles1 = [0, 1, 2]
niveles2 = [0, 45, 90, 135, 180, 225, 270, 315, 360]

n1 = [] # Niveles para x1
n2 = [] # Niveles para x2
t = [] # Tratamientos
k = 0
for i in range(len(niveles1)):
    for j in range(len(niveles2)):
        n1.append(niveles1[i])
        n2.append(niveles2[j])
        t.append(k)
        k += 1

df_tratamientos = pd.DataFrame({
    'n1' : n1,
    'n2' : n2,
    't' : t
})

df_tratamientos


# La cantidad de tratamientos es la combinación de elementos de $n_1$ y $n_2$, es decir $2 \times 9 = 27$. Estas combinaciones de niveles, dados por los tratamientos, pueden procesarse mediante algún modelo y estudiar los resultados obtenidos de manera estadística. 
# 
# En el ámbito del aprendizaje automático, es común realizar un diseño de experimentos que incluya métricas de evaluación como variable de estudio.

# ## Tarea (10 puntos)
# 
# - Realizar un diseño de experimentos para tu problema de estudio en la que indiques los niveles y tratamientos de interés.
# - Reporta en la metodología de tu artículo el diseño de experimentos que realizaste.
