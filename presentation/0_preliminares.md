---
marp: true
class: lead
header: 'UANL - FCFM - MCD | Aprendizaje automático - Alberto Benavides'
footer: 
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# Aprendizaje automático

Presentación

---

# Ponderación del curso

<center>
  <table>
    <tr>
      <th>Descripción</th>
      <th>Cantidad</th>
      <th>Puntos</th>
      <th>Total</th>
    </tr>
    <tr>
      <td>Actividades en clase</td>
      <td style="text-align:center">10</td>
      <td style="text-align:center">2</td>
      <td style="text-align:center">20</td>
    </tr>
    <tr>
      <td>Tareas</td>
      <td style="text-align:center">10</td>
      <td style="text-align:center">8</td>
      <td style="text-align:center">80</td>
    </tr>
    <tr>
      <td>PIA (Artículo)</td>
      <td style="text-align:center">10</td>
      <td style="text-align:center">1</td>
      <td style="text-align:center">10</td>
    </tr>
    <tr>
      <td colspan="3">Total</td>
      <td style="text-align:center">110</td>
    </tr>
  </table>
</center>

---

# Sesiones

<div class="columns">
<div>

**9 enero**: Presentación del curso
**16 enero**: Carga de datos y preprocesamiento
**23 enero**: Estadística descriptiva básica
**30 enero**: Selección de características
**6 febrero (asueto)**: Aprendizaje no supervisado

</div>
<div>

**13 febrero**: Clasificación
**20 febrero**: Regresión
**27 febrero**: Diseño de experimentos
**6 marzo**: Comparativas de desempeño
**20 marzo (asueto)**: Latex y artículo
**27 marzo**: Artículo

</div>
</div>

---

# Aprendizaje automático

- Técnicas computacionales
- Bases
    - Estadística
    - Minería de datos
    - Ciencia de datos
- Usos
    - Predicción
    - Clasificación
- [The human regression ensemble](https://justindomke.wordpress.com/2021/09/28/the-human-regression-ensemble/)

--- 

# Control de versiones

- Técnica computacional
- Seguir cambios en proyectos informáticos 
    - Individuales 
    - Colaborativos
- Almacenados virtualmente en **repositorios**
- Versión $\approx$ Instantánea :camera:
- Versiones pueden 
    - compararse o recuperarse
    - ramificarse, modificarse independientemente e implementarse de manera ordenada

--- 

# Herramientas para el control de versiones

[Git](https://git-scm.com/) es el sistema más usado para control de versiones

[Github](https://github.com/) es un servidor que hospeda repositorios de manera gratuita

---

# Git
<div class="columns">
<div>

```cmd
rem Iniciar un repositorio
git init

rem Clonar un repositorio existente
git clone url

rem Estado del repositorio
git status

rem Preparar un archivo o directorio
git add <archivo_o_directorio>

rem Desagregar un archivo o directorio
git reset arch

rem Crear un commit
git commit -m "Mensaje"
```

</div>

```cmd
rem Crea una rama
git branch <rama>

rem Cambia a una rama
git checkout <rama>

rem Une una rama a la actual
git merge <rama>

rem Añadir la url como remoto
git remote add <nombre> <url>

rem Enviar actualizaciones a un remoto
git push <remoto> <rama>

rem Obtiene cambios de remoto
git pull <remoto>
```
</div>
</div>

---

# Git (cont.)

```cmd
rem Deshacer cambios
git reset –hard

rem Borrar archivos nuevos
git clean -fxd
```

## `.gitignore`

Especifica archivos o carpetas excluidos del control de versiones

---

# Markdown

```markdown
# Encabezado 1
# Encabezado 2

Párrafo con *cursiva*, **negritas**, ~~tachados~~.

1. Lista
2. Numerada

* Lista
* No ordenada

[Vínculo a Google](www.google.com)

![Imagen](https://fakeimg.pl/150/)

`código`
```

---

# Python

## [Google Colab](https://colab.research.google.com/)

## 🤓 [Tutorial](https://colab.research.google.com/drive/1uoxgduAnH3e4Pz0YHLAyMnw2Cpilqqpy?usp=sharing)

Ver integración entre Google Collab y Github.

---

# Actividad en clase

- 2 puntos extra
- Crear repositorio para el curso que contenga 
  - en el `read.me`, una breve descripción de tu repositorio para este curso (primer *commit*)
  - un archivo Jupyter para la primera tarea con un ejemplo de regresión lineal o correlación hecho en puro Python (segundo *commit*)

---

# Tarea

- 8 puntos
- Definir un conjunto de datos sobre el que se trabajará durante el curso
- Justificar su elección
- Establecer objetivos a lograr en el curso
- Subir explicación en el repositorio en un cuaderno de Jupyter