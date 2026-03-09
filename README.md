# Laboratorios Geología Numérica G-4101

Material de laboratorio para el curso de **Geología Numérica G-4101**, de la [Escuela Centroamericana de Geología](https://www.geologia.ucr.ac.cr/), Universidad de Costa Rica.

## Descripción

Este repositorio contiene las guías de laboratorio del curso, implementadas como documentos [Quarto](https://quarto.org/) (`.qmd`) y cuadernos Jupyter (`.ipynb`). Los laboratorios cubren temas de estadística, álgebra y métodos numéricos aplicados a las geociencias, usando **Python** como lenguaje de programación principal.

## Contenido

| Laboratorio | Tema |
|---|---|
| `G4101_Intro_Python` | Introducción a Python para geociencias |
| `G4101_Intro_jupyter` | Introducción a Jupyter Notebooks |
| `G4101_Intro_Quarto` | Introducción a Quarto |
| `G4101_lab01_algebra` | Álgebra lineal y operaciones matriciales |
| `G4101_lab02_estadistica_descriptiva1` | Estadística descriptiva – Parte 1 |
| `G4101_lab03_estadistica_descriptiva2` | Estadística descriptiva – Parte 2 |
| `G4101_lab04_estadistica_distribuciones` | Distribuciones de probabilidad |
| `G4101_lab05_estadistica_intro-inferencial` | Introducción a la estadística inferencial |
| `G4101_lab06_estadistica_estimacion` | Estimación de parámetros |
| `G4101_lab07_estadistica_pruebas1` | Pruebas de hipótesis – Parte 1 |
| `G4101_lab08_estadistica_pruebas2` | Pruebas de hipótesis – Parte 2 |
| `G4101_lab09_estadistica_noparametrica` | Estadística no paramétrica |
| `G4101_lab10_estadistica_direccional` | Estadística direccional |
| `G4101_lab11_secuencias` | Análisis de secuencias |
| `G4101_lab12_geoestadistica` | Geostadística |

## Herramientas y paquetes principales

- **Python 3.13** con entorno conda (`environment_geo.yml`)
- [`numpy`](https://numpy.org/), [`pandas`](https://pandas.pydata.org/), [`scipy`](https://scipy.org/), [`statsmodels`](https://www.statsmodels.org/) — cálculo numérico y estadística
- [`matplotlib`](https://matplotlib.org/), [`seaborn`](https://seaborn.pydata.org/), [`plotnine`](https://plotnine.readthedocs.io/), [`plotly`](https://plotly.com/python/) — visualización
- [`geopandas`](https://geopandas.org/), [`rasterio`](https://rasterio.readthedocs.io/)  — datos geoespaciales
- [`gstools`](https://gstools.readthedocs.io/), [`pykrige`](https://geostat-framework.readthedocs.io/projects/pykrige/), [`scikit-gstat`](https://scikit-gstat.readthedocs.io/) — geoestadística
- [`pingouin`](https://pingouin-stats.org/), [`scikit-posthocs`](https://scikit-posthocs.readthedocs.io/) — pruebas estadísticas avanzadas
- [`mplstereonet`](https://mplstereonet.readthedocs.io/), [`mpltern`](https://mpltern.readthedocs.io/) — gráficos geológicos especializados
- [`polars`](https://docs.pola.rs/) — manipulación de datos de alto rendimiento
- [Quarto](https://quarto.org/) — generación de documentos reproducibles

## Configuración del entorno

```bash
conda env create -f environment_geo.yml
conda activate geo
```

## Uso

Cada laboratorio está disponible en dos formatos equivalentes:

- **`.qmd`** — documento Quarto, recomendado para renderizar como HTML o PDF
- **`.ipynb`** — cuaderno Jupyter, para ejecución interactiva celda a celda

Para renderizar un documento Quarto:

```bash
quarto render G4101_lab01_algebra.qmd
```

Los archivos generados se guardan en la carpeta `outputs/` según la configuración en `_quarto.yml`.