<div align="center">

# 🐍 Python para Minería de Datos

### Chuleta rápida: de cero a EDA, preprocesamiento y primer modelo

*pandas · NumPy · matplotlib · seaborn · SciPy · scikit-learn*

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Colab%20%7C%20Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Licencia](https://img.shields.io/badge/uso-educativo-2ea44f?style=for-the-badge)

<br>

**[Inicio rápido](#inicio-rápido)** ·
**[Python básico](#python-básico)** ·
**[pandas](#pandas-trabajar-con-tablas)** ·
**[Gráficas](#gráficas-matplotlib-y-seaborn)** ·
**[Estadística](#estadística-con-scipy)** ·
**[scikit-learn](#scikit-learn-preprocesamiento-y-modelos)** ·
**[Receta EDA](#receta-eda-en-10-pasos)** ·
**[Errores](#errores-típicos)**

</div>

---

> [!TIP]
> Esta chuleta está pensada para **leerla con el cuaderno abierto**. Copia un bloque en una celda de Colab o Jupyter, ejecútalo con <kbd>Mayús</kbd> + <kbd>Intro</kbd> y cambia valores para ver qué pasa.

## Contenido

| # | Sección | Para qué sirve |
|:-:|---|---|
| 1 | [Inicio rápido](#inicio-rápido) | Entorno, imports y primer vistazo a los datos |
| 2 | [Python básico](#python-básico) | Los 6 patrones que verás en cualquier cuaderno |
| 3 | [pandas](#pandas-trabajar-con-tablas) | Leer, filtrar, resumir y limpiar tablas |
| 4 | [Gráficas](#gráficas-matplotlib-y-seaborn) | Qué gráfica usar y cómo leerla |
| 5 | [Estadística](#estadística-con-scipy) | Normalidad, tests y tamaño del efecto |
| 6 | [scikit-learn](#scikit-learn-preprocesamiento-y-modelos) | Escalar, imputar, codificar, Pipeline, modelos y métricas |
| 7 | [Receta EDA](#receta-eda-en-10-pasos) | Orden de trabajo de un análisis exploratorio |
| 8 | [Errores](#errores-típicos) | Lo que más suele salir mal |
| 9 | [Glosario](#glosario) | Vocabulario esencial |

---

## Inicio rápido

### Dónde ejecutar Python

| Opción | Cómo | Cuándo |
|---|---|---|
| **Google Colab** | [colab.research.google.com](https://colab.research.google.com) → *Archivo → Subir cuaderno* | Sin instalar nada, solo cuenta de Google |
| **Jupyter / Anaconda** | Instalar [Anaconda](https://www.anaconda.com/download) y abrir *Jupyter Notebook* | Trabajar en local |
| **Script `.py`** | `python mi_script.py` en la terminal | Automatizar o desplegar |

### Instalar librerías

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
# en Colab ya vienen instaladas; para imbalanced-learn:
pip install imbalanced-learn
```

### Imports estándar (pégalos siempre al inicio)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
```

### Las librerías de un vistazo

| Librería | Alias | Se usa para |
|---|:-:|---|
| pandas | `pd` | Tablas (`DataFrame`): cargar, filtrar, agrupar, resumir |
| NumPy | `np` | Cálculo numérico, logaritmos, aleatorios |
| matplotlib | `plt` | Gráficas base |
| seaborn | `sns` | Gráficas estadísticas sobre matplotlib |
| SciPy (`stats`) | `stats` | Tests estadísticos y distribuciones |
| scikit-learn | `sklearn` | Preprocesamiento, modelos, validación y métricas |
| imbalanced-learn | `imblearn` | Remuestreo (SMOTE, undersampling) |

### Primer vistazo a unos datos (dataset de ejemplo incluido en sklearn)

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame                                       # tabla con 30 atributos + 'target'
df["diagnosis"] = df["target"].map({0: "maligno", 1: "benigno"})
df = df.drop(columns="target")

df.shape      # (569, 31) → filas y columnas
df.head()     # primeras 5 filas
df.info()     # tipos y valores no nulos
```

> [!NOTE]
> En scikit-learn `target = 0` significa **maligno** y `1` **benigno**, al revés de lo intuitivo. Por eso se recodifica: la clase "positiva" debe ser la que quieres detectar.

---

## Python básico

<details open>
<summary><b>Los 6 patrones que aparecen en todos los cuadernos</b></summary>

<br>

**1. Variables.** `=` guarda el resultado de la derecha con el nombre de la izquierda.

```python
n = 569
nombre = "mean area"           # texto entre comillas
atributos = ["radius", "area"] # lista entre corchetes
```

**2. Comentarios.** Lo que sigue a `#` lo ignora Python.

**3. Funciones y métodos.** Una acción con paréntesis; las opciones van dentro con nombre.

```python
len(atributos)                          # función: cuántos elementos tiene
df.describe()                           # método: se aplica al objeto df
sns.histplot(data=df, x="mean area", bins=40)   # opciones con nombre
```

**4. Encadenar métodos.** Se lee de izquierda a derecha.

```python
df.isna().sum().sum()   # marca huecos → cuenta por columna → suma total
```

**5. Bucles.** Repetir algo para cada elemento.

```python
for col in ["mean radius", "mean area"]:
    print(col, df[col].mean().round(2))
```

**6. Listas por comprensión.** Crear una lista filtrando en una línea.

```python
means = [c for c in df.columns if c.startswith("mean")]   # columnas que empiezan por 'mean'
```

</details>

<details>
<summary><b>Condiciones y operadores</b></summary>

<br>

| Operador | Significa | Ejemplo |
|:-:|---|---|
| `==` / `!=` | igual / distinto | `df["diagnosis"] == "maligno"` |
| `>` `<` `>=` `<=` | comparaciones | `df["mean area"] > 1000` |
| `&` | y (entre paréntesis) | `(a > 1) & (b < 2)` |
| `\|` | o (entre paréntesis) | `(a > 1) \| (b < 2)` |
| `~` | negación | `~df["x"].isna()` |
| `in` / `.isin()` | pertenece a | `df["g"].isin(["A", "B"])` |

</details>

---

## pandas: trabajar con tablas

### Leer y guardar

```python
df = pd.read_csv("datos.csv")              # leer CSV (sep=";" si usa punto y coma)
df = pd.read_excel("datos.xlsx")           # leer Excel
df.to_csv("salida.csv", index=False)       # guardar sin la columna de índice
```

### Explorar la estructura

| Comando | Devuelve |
|---|---|
| `df.shape` | (filas, columnas) |
| `df.head(n)` / `df.tail(n)` | primeras / últimas filas |
| `df.info()` | tipos, no nulos y memoria |
| `df.dtypes` | tipo de cada columna |
| `df.columns` | nombres de columnas |
| `df.describe()` | resumen estadístico numérico |
| `df.nunique()` | valores distintos por columna |
| `df["col"].value_counts()` | cuántas veces aparece cada valor |
| `df["col"].value_counts(normalize=True)` | lo mismo en proporción |

### Seleccionar y filtrar

```python
df["mean area"]                            # una columna (Series)
df[["mean area", "mean radius"]]           # varias columnas (DataFrame)
df[df["mean area"] > 1000]                 # filas que cumplen una condición
df[(df["mean area"] > 1000) & (df["diagnosis"] == "maligno")]
df.loc[df["mean area"] > 1000, "diagnosis"]   # filas por condición, columnas por nombre
df.iloc[0:5, 0:3]                          # por posición: 5 primeras filas, 3 primeras columnas
```

### Resumir y agrupar

```python
df.groupby("diagnosis")["mean area"].median()          # una cifra por grupo
df.groupby("diagnosis")[["mean area", "mean radius"]].agg(["median", "std"])
pd.crosstab(df["diagnosis"], df["mean area"] > 1000)   # tabla de contingencia
```

### Calidad de datos

```python
df.isna().sum()                  # huecos por columna
df.isna().sum().sum()            # huecos totales
df.duplicated().sum()            # filas repetidas
df = df.drop_duplicates()        # quitar repetidas
df.dropna()                      # quitar filas con algún hueco
df["mean area"].fillna(df["mean area"].median())   # rellenar con la mediana
(df.select_dtypes("number") < 0).sum().sum()       # ¿valores negativos imposibles?
```

> [!WARNING]
> **Ausentes escondidos:** un hueco puede ser un espacio en blanco, `"?"` o `"N/A"` y `isna()` no lo ve. Cuando una columna numérica sale como `object`, sospecha y conviértela:
> ```python
> df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # lo no numérico pasa a NaN
> ```

### Transformar columnas

```python
d = df.copy()                                               # trabaja sobre una copia para no estropear el original
d["log_area"] = np.log(d["mean area"])                      # nueva columna
d["abandona"] = (d["diagnosis"] == "maligno").astype(int)   # booleano → 0/1
d["diag_num"] = d["diagnosis"].map({"maligno": 1, "benigno": 0})
d["tramo"] = pd.cut(d["mean area"], bins=4)                 # igual anchura
d["cuartil"] = pd.qcut(d["mean area"], q=4)                 # igual frecuencia
d = d.drop(columns=["col_inutil"], errors="ignore")         # quitar columnas
d = d.rename(columns={"mean area": "area"})                 # renombrar
```

### Unir tablas

```python
medias = df[["mean area"]].assign(ID=range(len(df)))
etiquetas = pd.DataFrame({"ID": range(len(df)), "grupo": "A"})
medias.merge(etiquetas, on="ID", how="inner")      # horizontal: une columnas por clave (como un JOIN)
pd.concat([df, df], ignore_index=True)             # vertical: apila filas con las mismas columnas
```

---

## Gráficas: matplotlib y seaborn

### Qué gráfica usar

| Quiero ver... | Gráfica | Código |
|---|---|---|
| Forma de **una** variable numérica | Histograma | `sns.histplot(df["mean area"], kde=True)` |
| Mediana, cuartiles y atípicos | Boxplot | `sns.boxplot(x=df["mean area"])` |
| Una variable **por grupo** | Violín / boxplot | `sns.violinplot(data=df, x="diagnosis", y="mean area")` |
| Cuántos hay de cada categoría | Barras | `sns.countplot(data=df, x="diagnosis")` |
| Categoría **dividida por clase** | Barras con `hue` | `sns.countplot(data=df, x="Contract", hue="Churn")` |
| Relación entre **dos** numéricas | Dispersión | `sns.scatterplot(data=df, x="mean radius", y="mean area", hue="diagnosis")` |
| Correlaciones entre muchas | Heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="RdBu_r", vmin=-1, vmax=1)` |
| Todos los pares de variables | Pairplot | `sns.pairplot(df[cols], hue="diagnosis", corner=True)` |
| Dónde faltan datos | Heatmap de ausentes | `sns.heatmap(df.isna().T, cbar=False)` |
| ¿Es normal? | Gráfico Q-Q | `stats.probplot(df["mean area"], plot=plt)` |

### Varias gráficas a la vez

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 4))      # 1 fila, 3 gráficas
for a, c in zip(ax, ["mean radius", "mean area", "mean texture"]):
    sns.histplot(df[c], kde=True, ax=a)
    a.set_title(c)
plt.tight_layout()
plt.show()
```

### Cómo leer cada gráfica

<details>
<summary><b>Histograma · Boxplot · Violín · Heatmap · Pairplot · Q-Q</b></summary>

<br>

| Gráfica | Cómo se lee | Qué buscar |
|---|---|---|
| **Histograma** | Barras = cuántos datos caen en cada intervalo. La curva KDE es la versión suavizada | Forma (simétrica o con cola), varios picos (grupos mezclados) |
| **Boxplot** | La caja va de Q1 a Q3 (50 % central); la línea es la mediana; los bigotes llegan a 1,5·IQR | Puntos sueltos = candidatos a atípico; comparar cajas entre grupos |
| **Violín** | Histograma girado y simétrico; las líneas internas son cuartiles | Si los violines de dos clases se **solapan** o están separados |
| **Heatmap de correlación** | Cada celda = correlación entre fila y columna (−1 a 1) | Bloques rojos oscuros = variables redundantes |
| **Pairplot** | Diagonal: distribución por clase. Resto: un punto por caso en cada par | Si los colores de las clases se separan |
| **Q-Q plot** | Tus datos frente a una normal ideal | Puntos sobre la recta = normal; curva hacia arriba a la derecha = cola pesada |

</details>

---

## Estadística con SciPy

### Estadísticos básicos

| Qué | Comando | Cómo leerlo |
|---|---|---|
| Media / mediana | `s.mean()` / `s.median()` | Media > mediana ⇒ cola a la derecha |
| Desviación típica | `s.std()` | Dispersión en las unidades del dato |
| Cuartiles e IQR | `q1, q3 = s.quantile([.25, .75])`; `iqr = q3 - q1` | IQR = 50 % central |
| Coef. de variación | `s.std() / s.mean()` | Dispersión relativa, sin unidades |
| Asimetría | `s.skew()` | \|valor\| > 1 ⇒ asimetría fuerte |
| Curtosis | `s.kurt()` | > 0 ⇒ colas pesadas |
| Correlación | `df.corr()` · `df.corr("spearman")` | −1 a 1; Spearman si no es lineal |

### Atípicos: regla IQR y z-score

```python
s = df["mean area"]
q1, q3 = s.quantile([.25, .75])
iqr = q3 - q1
limite_sup = q3 + 1.5 * iqr
atipicos_iqr = s[(s < q1 - 1.5 * iqr) | (s > limite_sup)]    # IQR: robusta, no asume normalidad
atipicos_z = s[np.abs(stats.zscore(s)) > 3]                   # z-score: asume normalidad
len(atipicos_iqr), len(atipicos_z)
```

> [!IMPORTANT]
> Un atípico **no es un error** salvo que sea imposible. Antes de borrar nada pregúntate si es un caso real y si es justo la clase que te interesa.

### Contrastes de hipótesis

```python
# ¿Es normal?   H0: los datos son normales; p < 0,05 ⇒ se rechaza
stats.normaltest(df["mean area"])
stats.shapiro(df["mean area"])

# ¿Difieren dos grupos?  (no paramétrico: no exige normalidad)
a = df.loc[df["diagnosis"] == "maligno", "mean area"]
b = df.loc[df["diagnosis"] == "benigno", "mean area"]
U, p = stats.mannwhitneyu(a, b)

# Tamaño del efecto (d de Cohen): |d| > 0,8 ⇒ diferencia grande
d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
print(p, d)

# Dos categóricas: chi-cuadrado sobre una tabla de contingencia
chi2, p, gl, esperado = stats.chi2_contingency(pd.crosstab(df["diagnosis"], df["mean area"] > 1000))
```

> [!NOTE]
> **p-valor:** probabilidad de ver una diferencia así si en realidad no hubiese ninguna. Con muchos datos casi todo sale "significativo", así que mira siempre también el **tamaño del efecto** (d de Cohen).

---

## scikit-learn: preprocesamiento y modelos

### El esquema universal

```python
from sklearn.preprocessing import StandardScaler

esc = StandardScaler()            # 1. crear la herramienta
esc.fit(X_train)                  # 2. fit: APRENDE (calcula media y desviación) → SOLO con entrenamiento
X_tr = esc.transform(X_train)     # 3. transform: APLICA lo aprendido
X_te = esc.transform(X_test)      #    …y se aplica igual al test
# atajo para entrenamiento: esc.fit_transform(X_train)
# modelos: modelo.fit(X_train, y_train)  →  modelo.predict(X_test)
```

> [!CAUTION]
> **Regla de oro:** `fit` solo con los datos de **entrenamiento**. Si lo haces con todo el dataset, el test "se cuela" en el aprendizaje (fuga de información) y tus resultados salen más optimistas de lo real.

### Preparar datos: separar atributos, clase y partición

```python
X = df.drop(columns="diagnosis")
y = (df["diagnosis"] == "maligno").astype(int)       # clase positiva = 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
# stratify=y mantiene la proporción de clases · random_state fija el azar (repetible)
```

### Piezas de preprocesamiento

| Tarea | Pieza | Nota |
|---|---|---|
| Normalizar a [0, 1] | `MinMaxScaler()` | Sensible a atípicos |
| Media 0, desviación 1 | `StandardScaler()` | El más usado |
| Escalado robusto | `RobustScaler()` | Usa mediana e IQR; poco sensible a atípicos |
| Corregir asimetría | `PowerTransformer(method="yeo-johnson")` | Admite ceros y negativos; además estandariza |
| Rellenar ausentes (media/mediana) | `SimpleImputer(strategy="median")` | Reduce la varianza |
| Rellenar con vecinos | `KNNImputer(n_neighbors=5)` | Usa otros atributos |
| Rellenar con regresión | `IterativeImputer()` | Requiere `enable_iterative_imputer` |
| Categoría ordenada → número | `OrdinalEncoder(categories=[[...]])` | Peque < mediano < grande → 0, 1, 2 |
| Categoría sin orden → 0/1 | `OneHotEncoder()` · `pd.get_dummies()` | Una columna por categoría |
| Intervalos | `KBinsDiscretizer(n_bins=4)` · `pd.cut` · `pd.qcut` | Convierte número en categoría |
| Reducir dimensiones | `PCA(n_components=10)` | Estandariza antes |
| Seleccionar atributos | `SelectKBest(mutual_info_classif, k=10)` | Filtro |
| Seleccionar con el modelo | `SequentialFeatureSelector(modelo, n_features_to_select=5)` | Wrapper, más costoso |
| Anómalos multivariantes | `IsolationForest(contamination=0.05)` | Marca con −1 |

### Pipeline: encadenar todo sin fuga de información

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

DESCARTAR = ["worst area", "worst radius", "mean perimeter", "mean radius", "mean area"]

pipe = Pipeline([
    ("sel", ColumnTransformer([("fuera", "drop", DESCARTAR)], remainder="passthrough")),
    ("pt",  PowerTransformer(method="yeo-johnson")),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1", "recall", "roc_auc"])
print({k: round(v.mean(), 3) for k, v in res.items() if k.startswith("test_")})

pipe.fit(X_train, y_train)            # ajuste final con todo el entrenamiento
y_pred = pipe.predict(X_test)         # evaluación sobre el test, que no se ha tocado
```

### Modelos habituales

| Modelo | Código | Necesita escalar |
|---|---|:-:|
| Regresión logística | `LogisticRegression()` | Sí |
| k-NN | `KNeighborsClassifier(n_neighbors=5)` | **Sí** |
| SVM | `SVC()` | Sí |
| Árbol de decisión | `DecisionTreeClassifier(max_depth=3)` | No |
| Random Forest | `RandomForestClassifier()` | No |
| Línea base trivial | `DummyClassifier(strategy="most_frequent")` | No |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_test, y_test)   # línea base: ¿qué acierta sin aprender?
DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train).score(X_test, y_test)
```

### Métricas

```python
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

confusion_matrix(y_test, y_pred)             # [[VN, FP], [FN, VP]]
print(classification_report(y_test, y_pred))
recall_score(y_test, y_pred)                 # de los malignos reales, ¿cuántos detecto?
```

| Métrica | Pregunta que responde | Úsala cuando |
|---|---|---|
| **Accuracy** | ¿Qué % acierto? | Clases equilibradas (engaña si no) |
| **Recall** | De los positivos reales, ¿cuántos detecto? | Un falso negativo es grave (diagnóstico, fraude) |
| **Precisión** | De los que marco positivos, ¿cuántos lo son? | Un falso positivo es caro |
| **F1** | Equilibrio entre precisión y recall | Clases desbalanceadas |
| **AUC** | ¿Ordena bien positivos antes que negativos? (1 = perfecto, 0,5 = azar) | Comparar modelos sin fijar umbral |

### Clases desbalanceadas

```python
LogisticRegression(class_weight="balanced")        # opción más simple: pesos
# Remuestreo (SOLO sobre entrenamiento, nunca en test):
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
```

---

## Receta EDA en 10 pasos

```text
 1. Entender el significado y unidades de cada columna
 2. df.shape · df.info() · df.head()                       ¿qué tengo?
 3. df.isna().sum() · df.duplicated().sum()                ¿hay huecos? ¿escondidos?
 4. Reglas de dominio (negativos, rangos, coherencia)      ¿son datos posibles?
 5. df.describe() · skew() · histogramas                   ¿cómo es cada variable?
 6. Boxplot + regla IQR                                    ¿atípicos? ¿errores o reales?
 7. value_counts() de la clase                             ¿desbalanceo?
 8. Cada variable frente a la clase (violín/barras + test) ¿cuáles discriminan?
 9. Correlaciones y heatmap                                ¿redundancia?
10. Tabla: hallazgo → evidencia → decisión                 ¿qué decido?
```

<details>
<summary><b>Plantilla de la tabla de hallazgos</b></summary>

<br>

| Hallazgo | Evidencia | Decisión de preprocesamiento |
|---|---|---|
| Sin ausentes ni duplicados | `isna()`, `duplicated()` | No imputar |
| Escalas muy distintas | `describe()` | Escalar (k-NN, SVM, redes) |
| Asimetría positiva | `skew() > 1`, Q-Q | `PowerTransformer` o `log` |
| Atípicos reales | IQR, Isolation Forest | No eliminar |
| Variables redundantes | Correlación > 0,9 | Selección de atributos o PCA |
| Clase desbalanceada | `value_counts()` | Estratificar, F1/recall, pesos |

</details>

---

## Errores típicos

| Error | Cómo evitarlo |
|---|---|
| Mirar solo la media | Compara con la mediana y dibuja la distribución |
| Borrar atípicos "porque sí" | Averigua si son errores o casos reales |
| Creer que correlación = causalidad | Una correlación solo describe, no explica |
| Fiarse de p-valores con muchos datos | Calcula también el tamaño del efecto |
| Dar por bueno `isna() == 0` | Revisa tipos: un `object` donde esperabas número delata huecos escondidos |
| **Fuga de información** | `fit` solo con entrenamiento; usa `Pipeline` |
| Evaluar con accuracy en clases desbalanceadas | Usa recall, F1 o AUC |
| Partición sin estratificar | `train_test_split(..., stratify=y)` |
| Resultados no repetibles | Fija `random_state` y documenta las versiones |

---

## Glosario

<details>
<summary><b>Ver vocabulario esencial</b></summary>

<br>

| Término | Significado |
|---|---|
| **Instancia** | Una fila: un caso observado |
| **Atributo** | Una columna: una propiedad medida |
| **Clase / target** | Lo que se quiere predecir |
| **Outlier (atípico)** | Valor correcto pero muy distinto del resto |
| **Ruido** | Valor erróneo |
| **IQR** | Rango intercuartílico, Q3 − Q1 |
| **Asimetría (skew)** | Hacia qué lado se alarga la cola de una distribución |
| **Multicolinealidad** | Atributos que se pueden predecir unos con otros |
| **VIF** | Factor de inflación de la varianza; > 10 indica multicolinealidad severa |
| **Estratificar** | Dividir manteniendo la proporción de cada clase |
| **Validación cruzada** | Entrenar y probar rotando partes de los datos |
| **Pipeline** | Cadena de preprocesamiento + modelo ajustada sin fuga |
| **Fuga de información** | Usar al entrenar datos que no tendrías al predecir |
| **PCA** | Crear pocas variables nuevas que resumen la información de muchas |
| **Recall** | Proporción de positivos reales que el modelo detecta |
| **AUC** | Probabilidad de puntuar más alto a un positivo que a un negativo |

</details>

---

<div align="center">

**Hecha para estudiar Minería de Datos con Python** 🐍

<sub>Si te sirve, deja una ⭐ en el repositorio</sub>

</div>
