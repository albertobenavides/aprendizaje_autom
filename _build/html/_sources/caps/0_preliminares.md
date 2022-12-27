# Introducción

## ¿Qué es el aprendizaje automático?

- Se usa para predecir y clasificar.

- Usado en estadística, minería de datos, inteligencia artificial.

## Prerrequisitos

### Control de versiones

Fuente: https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud

El **control de versiones** se usa para seguir cambios en proyectos informáticos individuales o colaborativos. Los cambios se almacenan virtualmente en un *repositorio*. Las versiones del código se guardan mediante *commits* o instantáneas, que pueden compararse o recuperarse. Estas versiones pueden ramificarse, modificarse independientemente e implementarse de manera ordenada.

(Git)[https://git-scm.com/] es el sistema más usado para control de versiones. (Github)[https://github.com/] es un servidor que hospeda repositorios de manera gratuita.

#### Instrucciones esenciales de Git

```
rem Iniciar un repositorio en una carpeta
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

rem Obtiene cambios de un repositorio remoto
git pull <remoto>

rem Deshacer cambios
git reset –hard

rem Borrar archivos nuevos
git clean -fxd
```

En el archivo `.gitignore` se especifican qué archivos o carpetas deben ser excluidos del control de versiones.

### Markdown

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

### Python