# Predicción de abandono de clientes (Telco Customer Churn)

**Asignatura:** Machine Learning (MLY1101) · **Evaluación Parcial N°2** · **Caso A**
**Integrantes:** Kimberly Bobadilla, Luna Cortés y Millaray Paillafil

---

## Resumen ejecutivo

Una empresa de telecomunicaciones pierde ingresos cada vez que un cliente se va (*churn*). En este proyecto usamos el dataset **Telco Customer Churn (IBM)**, con 7.043 clientes, para:

1. entender qué características tienen los clientes que abandonan,
2. construir y comparar **3 modelos supervisados de clasificación** (Regresión Logística, Random Forest y Gradient Boosting),
3. **segmentar** a los clientes con modelos no supervisados (K-Means y agrupamiento jerárquico), y
4. traducir los resultados en recomendaciones para el negocio, revisando además el impacto ético del modelo.

Los tres modelos rindieron de forma muy parecida (ROC-AUC en validación cruzada entre 0,845 y 0,848). Seleccionamos **Gradient Boosting con balanceo de clases** por sus resultados de validación cruzada (ROC-AUC CV = 0,848; Average Precision CV = 0,663; ROC-AUC en test reportado = 0,845). Con el umbral estándar de 0,50, el notebook reporta que detecta **302 de los 374 clientes que abandonaron** en el conjunto de prueba (Recall = 0,807), con 291 falsos positivos. La segmentación identificó **4 perfiles** de cliente, con tasas de churn reportadas entre 7,4 % y 47,3 %.

---

## Tabla de contenidos

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [Metodología](#3-metodología)
4. [Descripción de las fuentes de datos](#4-descripción-de-las-fuentes-de-datos)
5. [Análisis exploratorio de los datos (EDA)](#5-análisis-exploratorio-de-los-datos-eda)
6. [Preparación y transformación de los datos](#6-preparación-y-transformación-de-los-datos)
7. [Justificación de la selección de modelos](#7-justificación-de-la-selección-de-modelos)
8. [Variables finales incluidas por modelo](#8-variables-finales-incluidas-por-modelo)
9. [Implementación de los modelos supervisados](#9-implementación-de-los-modelos-supervisados)
10. [Evaluación y selección del modelo](#10-evaluación-y-selección-del-modelo)
11. [Variables más importantes](#11-variables-más-importantes)
12. [Revisión ética y de sesgos](#12-revisión-ética-y-de-sesgos)
13. [Aprendizaje no supervisado: segmentación de clientes](#13-aprendizaje-no-supervisado-segmentación-de-clientes)
14. [Recomendaciones de negocio](#14-recomendaciones-de-negocio)
15. [Limitaciones y trabajo futuro](#15-limitaciones-y-trabajo-futuro)
16. [Instrucciones de ejecución](#16-instrucciones-de-ejecución)\n17. [Conclusiones generales](#conclusiones-generales)

---

## 1. Descripción del problema de negocio

Una empresa de telecomunicaciones pierde ingresos cuando un cliente cancela el servicio. Retener a un cliente cuesta menos que conseguir uno nuevo, por lo que conviene **anticiparse** e identificar quiénes tienen más probabilidad de irse para enfocar en ellos las ofertas de retención.

El problema se plantea como una **clasificación binaria**: `0` = el cliente se queda y `1` = el cliente abandona. Además de predecir, queremos **agrupar** a los clientes en segmentos con características parecidas, porque no todos se van por las mismas razones y una sola campaña para toda la cartera sería poco eficiente.

El modelo no busca reemplazar una decisión humana, sino **apoyar la priorización** de a quién contactar.

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar y evaluar modelos de Machine Learning que identifiquen clientes con riesgo de abandono y los segmenten según sus características, para apoyar decisiones de retención.

**Objetivos específicos:**

1. Analizar qué características tienen los clientes que abandonan (EDA) y revisar la calidad de los datos.
2. Preparar los datos para el modelamiento (codificación, división train/test, balanceo de clases).
3. Construir y comparar 3 modelos supervisados de clasificación con validación cruzada y ajuste de hiperparámetros.
4. Agrupar a los clientes con un modelo no supervisado (K-Means), comparándolo con agrupamiento jerárquico.
5. Evaluar con métricas adecuadas a un problema desbalanceado e interpretarlas para el negocio.
6. Revisar el impacto ético del modelo (sesgo por género y por adultos mayores).
7. Entregar recomendaciones de negocio.

## 3. Metodología

Seguimos una adaptación de **CRISP-DM**. La siguiente tabla muestra en qué parte de este informe y del notebook se desarrolla cada etapa:

| Etapa CRISP-DM | Qué hicimos | Sección del informe |
|---|---|---|
| Comprensión del negocio | Definimos el problema de churn, los objetivos y el tipo de modelos | 1, 2 |
| Comprensión de los datos | Calidad de datos, distribución de `Churn`, EDA numérico y categórico, correlaciones | 4, 5 |
| Preparación de los datos | Codificación *one-hot*, exclusión de `gender`, división estratificada 80/20, manejo del desbalance | 6 |
| Modelado | 3 modelos supervisados con `GridSearchCV` y 2 algoritmos de clustering | 7, 8, 9, 13 |
| Evaluación | Validación cruzada, evaluación en test, curvas ROC y PR, auditoría ética, estabilidad del clustering | 10, 11, 12, 13 |
| Despliegue / uso | Recomendaciones de negocio, limitaciones y guardado de modelos durante la ejecución del notebook | 14, 15 |

### Justificación de la metodología

Se utilizó CRISP-DM porque permite organizar el desarrollo de la solución desde la comprensión del problema comercial hasta la evaluación de los modelos y la formulación de recomendaciones. En este proyecto, la comprensión del negocio define la necesidad de anticipar el abandono; el análisis exploratorio permite identificar patrones relevantes; la preparación transforma los datos para el modelamiento; y la evaluación permite comparar el desempeño de los algoritmos según las necesidades del problema.

El proceso también considera la revisión ética y las limitaciones de los resultados. De esta manera, el objetivo no es únicamente obtener un buen indicador predictivo, sino construir una solución documentada que pueda interpretarse y utilizarse como apoyo para decisiones de retención.


**Herramientas:** Python en Google Colab con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y Joblib. Usamos `random_state = 42` en todo el proyecto para que los resultados sean reproducibles.

## 4. Descripción de las fuentes de datos

**Fuente:** *Telco Customer Churn* (IBM). Usamos una **versión limpia** del dataset, que preparamos en la evaluación anterior y que está guardada en `data/Telco_Customer_Churn_Clean.csv`.

- **Tamaño:** 7.043 clientes y 20 variables.
- **Identificador:** el dataset no incluye `customerID`.
- **Variable objetivo:** `Churn` (`Yes` / `No`).

| Grupo | Variables |
|---|---|
| Demográficas | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Servicios contratados | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contrato y pago | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Numéricas | `tenure` (meses de antigüedad), `MonthlyCharges`, `TotalCharges` |
| Objetivo | `Churn` |

## 5. Análisis exploratorio de los datos (EDA)

### 5.1 Calidad de los datos

Revisamos el tamaño, los tipos de datos, los nulos, los duplicados y los estadísticos descriptivos.

- **Nulos:** en la versión limpia `TotalCharges` ya está en formato numérico y los faltantes fueron tratados en la evaluación anterior.
- **Duplicados:** se detectaron **22 filas con todos sus valores coincidentes (aproximadamente 0,3 %)**. 
- **`TotalCharges = 0`:** corresponde a **11 clientes con `tenure = 0`** (clientes nuevos que aún no reciben su primera factura). Los mantuvimos porque son registros válidos.

**Justificación del tratamiento de duplicados**

Durante la evaluación anterior se realizó el proceso de limpieza del conjunto de datos. En la revisión actual se identificaron 22 filas idénticas (aproximadamente el 0,3 % del total). De acuerdo con el procedimiento de limpieza realizado previamente por el equipo, estos registros corresponden a clientes distintos y se conservaron para evitar la eliminación de observaciones consideradas válidas. Esta decisión mantiene la continuidad del conjunto de datos entre ambas evaluaciones.

Se reconoce que el archivo actual no incluye un identificador de cliente que permita comprobar esta distinción únicamente a partir de sus columnas. Por ello, la decisión se fundamenta en la revisión previa del dataset y no exclusivamente en el conteo de filas duplicadas.

### 5.2 Variable objetivo

| Churn | Clientes | Porcentaje |
|---|---|---|
| No | 5.174 | 73,5 % |
| Sí | 1.869 | 26,5 % |

![Distribución de Churn](images/distribucion_churn.png)

Las clases están **moderadamente desbalanceadas**. Por eso `Accuracy` no basta para evaluar los modelos (un modelo que dijera siempre "no se va" tendría ~73,5 % de accuracy sin detectar a nadie), y complementamos con Precision, Recall, F1, ROC-AUC y Average Precision.

### 5.3 Variables numéricas

![Variables numéricas según Churn](images/variables_numericas_churn.png)

| Grupo | Tenure promedio (meses) | MonthlyCharges promedio | TotalCharges promedio |
|---|---|---|---|
| Se quedan | 37,6 | 61,3 | 2.550 |
| Abandonan | 18,0 | 74,4 | 1.532 |

Los clientes que abandonan presentan, en promedio, menor antigüedad y mayores cargos mensuales. Su `TotalCharges` promedio es menor, lo que puede relacionarse con un menor tiempo de permanencia. Estos patrones sugieren que los primeros meses de relación con el cliente merecen especial atención, aunque una mayor antigüedad no elimina por completo el riesgo de abandono.

### 5.4 Variables categóricas

![Variables categóricas según Churn](images/variables_categoricas_churn.png)

| Variable | Tasa de churn |
|---|---|
| **Contrato** | Mes a mes: 42,7 % · 1 año: 11,3 % · 2 años: 2,8 % |
| **Internet** | Fibra óptica: 41,9 % · DSL: 19,0 % · Sin internet: 7,4 % |
| **Método de pago** | Cheque electrónico: 45,3 % · otros métodos: entre 15,2 % y 19,1 % |
| **Soporte técnico** | Sin soporte: 41,6 % · con soporte: 15,2 % |
| **Adulto mayor** | Sí: 41,7 % · No: 23,6 % |
| **Género** | Mujeres: 26,9 % · Hombres: 26,2 % (sin diferencia relevante) |

### 5.5 Correlaciones

![Matriz de correlación](images/matriz_correlacion.png)

- `tenure` tiene una correlación negativa moderada con el churn (**−0,35**): los clientes nuevos son más propensos a irse.
- `MonthlyCharges` tiene una relación positiva leve con el churn (**0,19**).
- `TotalCharges` y `tenure` están muy correlacionadas (**0,83**). Esta multicolinealidad puede distorsionar a los modelos lineales, así que más adelante excluimos `TotalCharges` de la Regresión Logística.

**Interpretación del análisis exploratorio**

Los resultados sugieren que la antigüedad, el tipo de contrato y las características de los servicios contratados permiten distinguir grupos con diferentes tasas de abandono. En particular, el contrato mes a mes presenta una tasa de churn del 42,7 %, mientras que los contratos de uno y dos años presentan tasas de 11,3 % y 2,8 %, respectivamente.

Estas diferencias justifican incluir las variables de contrato y servicios en el modelamiento. Sin embargo, los porcentajes describen asociaciones observadas en el dataset y no demuestran que una característica sea, por sí sola, la causa del abandono. La utilidad predictiva de estas variables debe evaluarse mediante los modelos y las métricas posteriores.

> Estos resultados muestran **asociaciones**, no causalidad.

## 6. Preparación y transformación de los datos

1. **Variable objetivo:** `Churn` se transformó a binario (`No` = 0, `Yes` = 1).
2. **Exclusión de `gender`:** la sacamos de los predictores por criterio ético, pero la conservamos aparte para la auditoría de sesgo (sección 12).
3. **Codificación:** las variables categóricas se convirtieron con *One-Hot Encoding* (`pd.get_dummies(..., drop_first=True)`). El resultado son **29 variables predictoras**.
4. **División train/test:** 80 % entrenamiento (5.634 clientes) y 20 % prueba (1.409 clientes), con `stratify=y` para mantener la misma proporción de churn (26,5 %) en ambos conjuntos.
5. **Escalado:** `StandardScaler` solo para la Regresión Logística (dentro de un *pipeline*, para no filtrar información del test). Random Forest y Gradient Boosting no lo necesitan.
6. **Desbalance de clases:** `class_weight='balanced'` en Regresión Logística y Random Forest, y `sample_weight` balanceado en Gradient Boosting (que no tiene `class_weight`).
7. **Validación:** los tres modelos usan los mismos **5 folds estratificados** (`StratifiedKFold`, `shuffle=True`, `random_state=42`).

## 7. Justificación de la selección de modelos

Elegimos tres clasificadores que representan enfoques distintos, para comparar un modelo lineal con modelos basados en árboles:

| Modelo | Por qué lo elegimos |
|---|---|
| **Regresión Logística** | Línea base simple e interpretable. Sus coeficientes permiten explicar en qué dirección influye cada variable. |
| **Random Forest** | Captura relaciones no lineales e interacciones entre variables, es robusto al ruido y entrega una importancia de variables (orientativa). |
| **Gradient Boosting** | Suele tener muy buen rendimiento en datos tabulares porque corrige errores de forma secuencial con árboles pequeños. |

**Justificación de la comparación**

La comparación de estos tres algoritmos permite evaluar si los modelos basados en árboles aportan mejoras frente a un modelo lineal interpretable. La Regresión Logística sirve como referencia para analizar la dirección de las relaciones entre las variables y el abandono; Random Forest permite capturar interacciones y patrones no lineales; y Gradient Boosting construye sucesivamente árboles que buscan mejorar los errores del modelo anterior.

Se utiliza el mismo esquema de validación cruzada estratificada y ROC-AUC como criterio de ajuste para mantener una comparación consistente. La selección final considera también Average Precision y Recall, debido al desbalance de clases y al interés comercial por detectar clientes que podrían abandonar.

Como referencia, entrenamos además un **Gradient Boosting sin balancear**, solo para mostrar el efecto del balanceo de clases.

## 8. Variables finales incluidas por modelo

| Modelo | N° de variables | Variable excluida | Escalado | Tratamiento del desbalance |
|---|---|---|---|---|
| Regresión Logística | 28 | `TotalCharges` (colinealidad con `tenure`, 0,83) | `StandardScaler` (en pipeline) | `class_weight='balanced'` |
| Random Forest | 29 | ninguna | no necesita | `class_weight='balanced'` |
| Gradient Boosting | 29 | ninguna | no necesita | `sample_weight` balanceado |

Las 29 variables corresponden a las numéricas (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) y a las variables categóricas codificadas (por ejemplo `Contract_One year`, `Contract_Two year`, `InternetService_Fiber optic`, `InternetService_No`, `PaymentMethod_Electronic check`, `OnlineSecurity_Yes`, `TechSupport_Yes`, etc.). El listado completo se imprime en el notebook (sección *Variables Finales por Modelo*).

### Justificación de las decisiones de preparación

La variable objetivo `Churn` se transformó a formato numérico para permitir el entrenamiento de los clasificadores. Las variables categóricas se transformaron mediante codificación One-Hot, utilizando `drop_first=True` para evitar categorías redundantes en la representación resultante.

La variable `gender` se excluyó de las variables predictoras por criterio ético, pero se conservó para examinar posteriormente posibles diferencias de desempeño entre grupos. Esta exclusión no demuestra por sí sola que el modelo esté libre de sesgos ni que su rendimiento permanezca inalterado.

En la Regresión Logística se excluyó `TotalCharges` por su alta correlación con `tenure` (0,83), con el propósito de reducir la redundancia entre variables. Esta decisión se aplicó específicamente a dicho modelo; Random Forest y Gradient Boosting conservaron esa variable porque los modelos basados en árboles no requieren la misma consideración de escalado y colinealidad que la regresión lineal.

La división de entrenamiento y prueba se realizó en proporción 80/20 y de forma estratificada, manteniendo una distribución comparable de la variable objetivo en ambos conjuntos. La validación cruzada estratificada se utilizó posteriormente para comparar los modelos y ajustar sus hiperparámetros.

## 9. Implementación de los modelos supervisados

Los hiperparámetros se ajustaron con `GridSearchCV` usando los 5 folds estratificados y **ROC-AUC** como métrica, de modo que la elección no depende de una sola división de los datos.

| Modelo | Malla de búsqueda | Mejores hiperparámetros | AUC (GridSearchCV) |
|---|---|---|---|
| Regresión Logística | `C` ∈ {0,01; 0,1; 1; 10} | `C = 10` | 0,845 |
| Random Forest | `n_estimators` ∈ {100, 200}; `max_depth` ∈ {5, 8, 12}; `min_samples_leaf` ∈ {1, 5, 10} | `n_estimators=200`, `max_depth=8`, `min_samples_leaf=10` | 0,848 |
| Gradient Boosting | `n_estimators` ∈ {100, 200}; `learning_rate` ∈ {0,05; 0,1}; `max_depth` ∈ {2, 3} | `n_estimators=200`, `learning_rate=0,05`, `max_depth=2` | 0,849 |

### Justificación de la selección de modelos

Se seleccionaron tres algoritmos complementarios para comparar distintas formas de modelar el abandono de clientes.

La Regresión Logística se utiliza como modelo de referencia por su interpretabilidad y porque permite analizar la dirección de la asociación entre las variables predictoras y la probabilidad estimada de abandono. Random Forest combina múltiples árboles de decisión y permite representar relaciones no lineales e interacciones entre variables. Gradient Boosting construye árboles de forma secuencial, buscando mejorar el desempeño de los modelos anteriores, y constituye una alternativa apropiada para datos tabulares.

Los tres modelos se compararon mediante validación cruzada estratificada de cinco particiones, utilizando ROC-AUC como criterio de ajuste de hiperparámetros. Además, se evaluaron Precision, Recall, F1 y Average Precision para analizar aspectos complementarios del desempeño, especialmente relevantes cuando la clase de abandono es minoritaria.

La selección no se basa únicamente en la diferencia numérica entre los modelos. También considera la capacidad de identificar clientes que abandonan, el volumen de falsas alarmas y la posible utilidad de las predicciones para orientar acciones de retención.

## 10. Evaluación y selección del modelo

### 10.1 Selección en validación cruzada

Para comparar los tres modelos bajo el mismo esquema, generamos **predicciones fuera de fold** (`cross_val_predict`) sobre el conjunto de entrenamiento. Así evitamos mirar el test para decidir.

| Modelo | ROC-AUC (CV) | Average Precision (CV) |
|---|---|---|
| Regresión Logística | 0,845 | 0,655 |
| Random Forest | 0,847 | 0,661 |
| **Gradient Boosting (balanceado)** | **0,848** | **0,663** |

### Interpretación de la validación cruzada

Los resultados de validación cruzada permiten comparar el comportamiento de los modelos utilizando diferentes particiones de los datos de entrenamiento. Las métricas ROC-AUC y Average Precision aportan información complementaria: la primera evalúa la capacidad de ordenar correctamente los casos positivos respecto de los negativos a través de distintos umbrales; la segunda se concentra en la relación entre Precision y Recall para la clase positiva.

Cuando las diferencias entre modelos son pequeñas, no corresponde afirmar que uno es claramente superior en términos generales. La elección debe considerar las métricas obtenidas, el objetivo comercial y las limitaciones de cada algoritmo.

Los resultados de `GridSearchCV` corresponden al ajuste de hiperparámetros, mientras que las métricas calculadas mediante predicciones fuera de fold constituyen una evaluación adicional de las predicciones sobre las particiones de entrenamiento. Los resultados del conjunto de prueba se utilizan para evaluar el comportamiento final de los modelos en observaciones que no participaron en el entrenamiento.

Por ello, estas métricas deben presentarse por separado, identificando el procedimiento del que provienen y evitando compararlas como si fueran resultados obtenidos mediante el mismo cálculo.

### 10.2 Evaluación final en el conjunto de prueba (umbral 0,50)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión Logística | 0,742 | 0,509 | 0,775 | 0,614 | 0,838 |
| Random Forest | 0,754 | 0,525 | 0,783 | 0,629 | 0,845 |
| **Gradient Boosting (balanceado)** | 0,742 | 0,509 | **0,807** | 0,625 | 0,845 |
| Gradient Boosting sin balancear (referencia) | 0,798 | 0,656 | 0,500 | 0,568 | 0,846 |

![Matrices de confusión](images/matrices_confusion_comparacion.png)

![Curvas ROC y Precision-Recall](images/curvas_roc_precision_recall.png)

### Interpretación comercial del umbral de decisión

El umbral de 0,50 se utiliza como referencia para comparar los modelos bajo una regla de clasificación común. Con este umbral, el Recall representa la proporción de clientes que realmente abandonaron y fueron identificados por el modelo, mientras que los falsos positivos representan clientes que fueron marcados como posibles casos de abandono, aunque no abandonaron en el conjunto de prueba.

En una estrategia comercial, un Recall alto puede ayudar a identificar una mayor proporción de clientes en riesgo, pero también puede aumentar la cantidad de clientes contactados innecesariamente. Por este motivo, el modelo debe interpretarse como una herramienta de priorización y no como una certeza sobre el comportamiento futuro de cada persona.

La selección definitiva del umbral debería considerar los costos de contacto, el costo de los incentivos y el beneficio esperado de retener a un cliente. Como estos costos no se modelaron directamente en el presente proyecto, el umbral de 0,50 se considera una referencia inicial y no un óptimo económico demostrado.

### 10.3 Interpretación de las métricas

- **ROC-AUC ≈ 0,845:** si tomamos un cliente que se fue y uno que se quedó al azar, el modelo le asigna mayor probabilidad de churn al que se fue en cerca del 84,5 % de los casos. Las curvas ROC de los tres modelos son casi idénticas, coherente con el empate en AUC.
- **Average Precision (CV) = 0,663:** la proporción base de churn es 26,5 %, así que el modelo rinde unas 2,5 veces mejor que elegir al azar. En la curva Precision-Recall, Gradient Boosting y Random Forest mantienen una precisión algo mayor que la Regresión Logística en recall bajo y medio (entre 0,0 y 0,6).
- **Matriz de confusión de Gradient Boosting (umbral 0,50):** detecta **302 de 374** clientes que realmente se van (Recall = 0,807), deja pasar 72 y genera **291 falsos positivos**. Esto significa que, de los ~593 clientes marcados en riesgo, aproximadamente la mitad (Precision = 0,509) se iba a ir de verdad.
- **Efecto del balanceo:** sin balancear, el modelo tiene más accuracy (0,798) pero **solo detecta al 50 % de quienes se van** (Recall = 0,500). Para un problema de retención, perder la mitad de los clientes en fuga es un costo alto, por eso preferimos la versión balanceada.
- **Accuracy y F1 de Random Forest:** a umbral 0,50 Random Forest es levemente superior (0,754 y 0,629 contra 0,742 y 0,625). Mantenemos Gradient Boosting por el criterio definido de antemano en CV y porque tiene el mayor Recall, que es lo más importante para no dejar escapar clientes. Reconocemos que **las diferencias entre los tres modelos son pequeñas**.

**Justificación de la selección final**

Se seleccionó Gradient Boosting balanceado porque obtuvo los mejores resultados de ROC-AUC y Average Precision en la tabla de predicciones fuera de fold del conjunto de entrenamiento. La diferencia con los otros modelos es pequeña, por lo que no se considera una superioridad concluyente, sino una ventaja limitada según los criterios establecidos para la comparación.

En el conjunto de prueba, Gradient Boosting balanceado alcanzó un Recall de 0,807, identificando 302 de los 374 clientes que abandonaron. Esta capacidad de detección es relevante para una estrategia de retención, aunque genera 291 falsos positivos. Por tanto, el modelo permite priorizar posibles casos de abandono, pero no garantiza que todos los clientes contactados necesiten una oferta de retención.

La elección del umbral operativo debe revisarse cuando se disponga de información sobre los costos de las campañas, el margen de los clientes y la efectividad real de las ofertas.

### 10.4 Umbral de decisión

Se utiliza el umbral estándar de 0,50 como referencia para comparar los modelos bajo una misma regla de clasificación. Un umbral más alto podría reducir los falsos positivos, pero también disminuir el Recall y aumentar la cantidad de clientes que abandonan sin ser detectados. La selección del umbral definitivo debe considerar los costos de contacto, los incentivos comerciales y el beneficio esperado de retener a un cliente. Como estos costos no fueron modelados en el presente proyecto, el umbral de 0,50 no se considera un óptimo económico demostrado.

## 11. Variables más importantes

![Top 10 variables Random Forest](images/importancia_variables_rf.png)

Random Forest (importancia basada en impureza, solo orientativa) destaca estas variables:

| Variable | Importancia |
|---|---|
| `tenure` | 0,182 |
| `Contract_Two year` | 0,134 |
| `TotalCharges` | 0,120 |
| `InternetService_Fiber optic` | 0,104 |
| `PaymentMethod_Electronic check` | 0,067 |
| `MonthlyCharges` | 0,067 |
| `Contract_One year` | 0,047 |
| `OnlineSecurity_Yes` | 0,033 |
| `TechSupport_Yes` | 0,030 |
| `InternetService_No` | 0,023 |


En la Regresión Logística (variables estandarizadas) los coeficientes confirman la dirección de las relaciones: la **antigüedad (−0,76)** y los **contratos de uno (−0,30) y dos años (−0,62)** reducen el riesgo de churn, mientras que la **fibra óptica (+1,29)** lo aumenta.

Ambos modelos coinciden con el EDA: el riesgo se concentra en clientes nuevos, con contrato mes a mes y fibra óptica. Son relaciones **predictivas**, no necesariamente causales.

## 12. Revisión ética y de sesgos

### 12.1 Decisiones tomadas

- La variable `gender` se excluyó de los predictores del modelo por criterio ético y se conservó para auditar diferencias de desempeño entre grupos.
- `SeniorCitizen` se mantuvo como predictor por su relación observada con el abandono en el EDA. También se utiliza para revisar posibles diferencias de desempeño entre adultos mayores y el resto de los clientes.
- La auditoría se realiza sobre el conjunto de prueba, usando el modelo **Gradient Boosting (balanceado)** y el umbral de clasificación de **0,50**.

### 12.2 Métricas auditadas

El notebook calcula por grupo el tamaño de la muestra, el porcentaje de churn real, el porcentaje marcado en riesgo, Recall, Precision y la tasa de falsos positivos. Los valores finales de esta sección deben transcribirse desde la tabla que imprime la celda de auditoría ética después de ejecutar el notebook completo en Colab.

| Grupo auditado | N° de clientes | Tasa real de abandono | Clientes marcados en riesgo | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Mujeres | 687 | 28,09 % | 41,48 % | 78,00 % | 53,00 % | 27,13 % |
| Hombres | 722 | 25,07 % | 42,66 % | 83,00 % | 49,00 % | 29,02 % |
| Adultos mayores (SeniorCitizen = 1) | 222 | 44,14 % | 68,02 % | 93,00 % | 60,00 % | 48,39 % |
| No adultos mayores (SeniorCitizen = 0) | 1.187 | 23,25 % | 37,24 % | 76,00 % | 48,00 % | 25,36 % |

Los resultados muestran diferencias en el desempeño del modelo entre los grupos auditados. Por género, el Recall es de 78 % en mujeres y 83 % en hombres, mientras que la tasa de falsos positivos es de 27,13 % y 29,02 %, respectivamente.

Por condición de adulto mayor, el modelo alcanza un Recall de 93 % en el grupo SeniorCitizen = 1, frente a 76 % en el grupo SeniorCitizen = 0. Sin embargo, la tasa de falsos positivos también es considerablemente mayor en adultos mayores (48,39 % frente a 25,36 %).

Estos resultados muestran que el modelo identifica una proporción mayor de los casos reales de abandono entre los adultos mayores, pero también clasifica incorrectamente como clientes en riesgo a una proporción mayor de quienes no abandonan. Por ello, se recomienda revisar estos resultados antes de utilizar las predicciones para orientar campañas comerciales. Las diferencias observadas son descriptivas y no demuestran, por sí solas, discriminación causal.

### Auditoría de desempeño según condición de adulto mayor

La revisión por condición de adulto mayor muestra diferencias en la tasa de falsos positivos del modelo. En los resultados registrados, el grupo `SeniorCitizen = 1` presenta una tasa de falsos positivos de 48,39 %, frente a 25,36 % en el grupo `SeniorCitizen = 0`. Las tasas reales de abandono reportadas son 44,14 % y 23,25 %, respectivamente.

La mayor tasa de falsos positivos en el primer grupo requiere atención, ya que puede implicar un mayor volumen de contactos u ofertas de retención que no resulten necesarios. Aunque las diferencias en las tasas reales de abandono aportan contexto, no permiten concluir que toda la disparidad se explique por ellas.

Como trabajo futuro, se propone evaluar modelos alternativos, analizar los errores por grupo y revisar los umbrales de decisión. Cualquier ajuste deberá comprobarse mediante métricas comparables y considerando tanto la equidad como los objetivos comerciales. La utilización de umbrales diferenciados requiere una evaluación específica y no debe asumirse como una solución automática al problema de sesgo.

### 12.3 Interpretación responsable

La tasa de falsos positivos se calcula como `FP / (FP + TN)`: representa la proporción de clientes que realmente no abandonaron, pero fueron clasificados como posibles casos de abandono. Debe distinguirse de la proporción de clientes que el modelo marcó en riesgo (`pred.mean()`).

Las diferencias entre grupos deben interpretarse considerando tanto el desempeño del modelo como la tasa real de abandono de cada grupo. Una disparidad es una señal que merece revisión, pero por sí sola no demuestra discriminación causal. La exclusión de una variable sensible tampoco garantiza que el modelo esté libre de sesgos, ya que otras variables pueden estar relacionadas con ella.

### 12.4 Medidas de mitigación y privacidad

Antes de un uso real, se recomienda:
- comparar modelos con y sin `gender` para cuantificar el efecto de esa decisión sobre las métricas;
- analizar las diferencias de Recall, Precision y tasa de falsos positivos entre grupos;
- evaluar medidas de mitigación apropiadas y documentar sus efectos;
- mantener revisión humana antes de activar campañas comerciales;
- usar únicamente los datos necesarios, limitar los accesos y monitorear el desempeño con datos nuevos.

El modelo debe apoyar la priorización de acciones de retención; no debe utilizarse para negar servicios ni aplicar sanciones a los clientes.

## 13. Aprendizaje no supervisado: segmentación de clientes

**Objetivo:** encontrar grupos de clientes con características parecidas para diseñar acciones distintas según el perfil. Es un complemento a la predicción: el modelo supervisado dice *quién* tiene riesgo y la segmentación ayuda a entender *qué tipo de cliente* es.

**Datos y algoritmos:** usamos las 28 variables de la Regresión Logística, escaladas con `StandardScaler`, y comparamos **K-Means** con **agrupamiento jerárquico aglomerativo (enlace Ward)**. La variable `Churn` **no** se usó para construir los clusters, solo para describirlos después.

### 13.1 Elección de k

Probamos k entre 2 y 8 con inercia (método del codo), Silhouette y Davies-Bouldin.

![Selección de k](images/seleccion_k_clustering.png)

- **k = 2** tiene el mejor Silhouette (0,353) y Davies-Bouldin (1,048), pero su partición es poco útil: solo separa a los clientes **sin internet** (1.526 clientes, 7,4 % de churn) de un único grupo de 5.517 clientes (78 %) con 31,8 % de churn que mezcla perfiles muy distintos. Dentro de ese grupo grande, el churn de los segmentos de k = 4 va de 14,9 % a 47,3 %, así que una sola estrategia no sería adecuada.
- **k = 4** fue nuestra elección: en el codo muestra el último descenso relevante de inercia (**11,55 %**) y divide al grupo grande en perfiles con riesgo, antigüedad y facturación claramente distintos.

### Justificación de la selección de cuatro segmentos

La selección de \(k=4\) responde a un criterio de utilidad comercial, no a que este valor obtenga los mejores indicadores de separación. En los resultados registrados, \(k=2\) alcanza un Silhouette de 0,353 y un Davies-Bouldin de 1,048, mientras que K-Means con \(k=4\) obtiene un Silhouette de 0,235.

Aunque dos clusters presentan una mejor separación según estas métricas, la segmentación en cuatro grupos permite distinguir perfiles con diferencias relevantes en antigüedad, cargos mensuales, tipo de contrato y tasas observadas de abandono. Esta diferenciación puede ser útil para formular estrategias de retención específicas para distintos grupos de clientes.

La decisión implica un compromiso entre separación matemática e interpretabilidad comercial. El Silhouette de 0,235 indica que la estructura de cuatro grupos presenta una separación limitada y posible solapamiento. Por tanto, los segmentos se interpretan como perfiles orientativos y no como categorías perfectamente diferenciadas.

La variable `Churn` se utiliza después del ajuste para caracterizar y ordenar los clusters según la tasa observada de abandono; no se incorpora como predictor en la matriz utilizada para construirlos. De este modo, se distingue la formación de los grupos de su interpretación posterior.

### 13.2 Estabilidad de los clusters

| Verificación | Resultado | Interpretación |
|---|---|---|
| ARI de K-Means entre 10 semillas | promedio 0,999 | K-Means es muy estable |
| ARI K-Means vs. jerárquico | 0,664 | Concordancia moderada: los segmentos tienen cierto respaldo, aunque no son idénticos |
| Silhouette K-Means / jerárquico | 0,235 / 0,222 | Grupos con solapamiento |

Para visualizar usamos **PCA de 2 componentes**, que explican solo el **45,5 %** de la varianza. Por eso el solapamiento del gráfico es esperable; el PCA se usó únicamente para graficar, no para segmentar.

![Clusters en PCA](images/visualizacion_clusters_pca.png)

### 13.3 Perfil de los segmentos (K-Means, k = 4)

Después de entrenar el clustering, se reordenan las etiquetas de los grupos de mayor a menor tasa de churn para facilitar su interpretación. El churn no se incorpora a las variables usadas para ajustar K-Means ni el modelo jerárquico; se utiliza después para describir y ordenar los perfiles.

| Cluster | Clientes | Tenure prom. | Cargo mensual | Churn | Perfil principal |
|---|---|---|---|---|---|
| **C0** | 2.670 | 16,2 | 74,2 | **47,3 %** | Clientes nuevos, 92,7 % con contrato mes a mes, mucha fibra óptica y pocos servicios adicionales (1,5 en promedio) |
| **C1** | 682 | 31,7 | 42,0 | 24,9 % | Clientes con DSL, cargos mensuales intermedios y riesgo moderado |
| **C2** | 2.165 | 53,8 | 91,1 | 14,9 % | Clientes antiguos, de alta facturación y con varios servicios agregados |
| **C3** | 1.526 | 30,5 | 21,1 | **7,4 %** | Clientes sin internet, bajo costo mensual y el riesgo más bajo |

### Interpretación comercial de los perfiles

Los perfiles permiten observar que los grupos difieren en antigüedad, facturación y características de los servicios contratados. Estas diferencias aportan una perspectiva complementaria al modelo supervisado: mientras el clasificador estima el riesgo individual de abandono, el clustering permite describir segmentos de clientes con características similares.

El segmento C0, con la mayor tasa observada de churn, constituye una prioridad para explorar acciones preventivas de retención. C1 presenta un riesgo intermedio y podría beneficiarse de estrategias orientadas a comprender las necesidades de sus clientes. C2 agrupa clientes con mayor antigüedad y facturación, lo que permite explorar iniciativas de fidelización. C3 presenta la menor tasa de abandono observada y podría requerir acciones de seguimiento menos intensivas.

Estas propuestas son hipótesis de aplicación comercial derivadas de los perfiles. No se ha demostrado que pertenecer a un cluster cause el abandono ni que una campaña específica vaya a reducirlo. La efectividad de las acciones debe comprobarse mediante pruebas piloto y la comparación de sus resultados con un grupo de control.

**Justificación de la segmentación final**

La elección de cuatro segmentos no se basa en que K-Means obtenga las mejores métricas de separación, ya que para k = 2 se obtuvieron mejores valores de Silhouette y Davies-Bouldin. Se priorizó k = 4 porque permite distinguir perfiles con diferencias relevantes de antigüedad, facturación y tasa de abandono, lo que facilita proponer estrategias de retención diferenciadas.

El segmento C0 concentra la mayor tasa de abandono, por lo que constituye una prioridad para evaluar acciones preventivas. C2 presenta clientes con mayor antigüedad y facturación, por lo que resulta pertinente explorar estrategias de fidelización. C3 tiene la menor tasa de abandono, lo que sugiere que podría bastar con un seguimiento de menor intensidad.

La segmentación presenta limitaciones: el Silhouette de K-Means es 0,235 y los grupos tienen cierto solapamiento. Por ello, los segmentos se utilizan como una herramienta de interpretación y apoyo comercial, no como categorías exactas ni como evidencia de las causas del abandono.

## 14. Recomendaciones de negocio

| Segmento | Recomendación |
|---|---|
| **C0 – Alto riesgo (47,3 %)** | Priorizar campañas de retención. Incentivar el paso de contrato mes a mes a contratos de 1 o 2 años (que tienen 11,3 % y 2,8 % de churn) y revisar la experiencia de los clientes con fibra óptica (41,9 % de churn). Ofrecer soporte técnico, ya que sin él el churn es 41,6 %. |
| **C1 – Riesgo moderado (24,9 %)** | Campañas de fidelización moderadas y evaluar servicios adicionales que aumenten la permanencia. |
| **C2 – Alto valor, bajo riesgo (14,9 %)** | Evitar descuentos masivos innecesarios; usar beneficios de fidelización para mantener la relación. |
| **C3 – Bajo riesgo (7,4 %)** | Monitoreo de bajo costo y evitar campañas agresivas. |

**Uso del modelo predictivo:**

- Usar el *score* de Gradient Boosting para ordenar a los clientes por riesgo y contactar primero a los de mayor probabilidad, combinándolo con el segmento al que pertenecen.
- Con el umbral 0,50 la campaña alcanzaría a ~81 % de quienes se van, pero la mitad de los contactados no se iría. Si el costo de la oferta es alto, conviene **subir el umbral** para contactar a menos clientes con mayor certeza.
- Mantener **revisión humana** antes de aplicar acciones comerciales, sobre todo en adultos mayores (ver sección 12).

**Validación de las recomendaciones**

Las recomendaciones se derivan de los patrones observados en el análisis exploratorio, de las predicciones del modelo supervisado y de los perfiles obtenidos mediante clustering. No se ha demostrado todavía que las acciones propuestas reduzcan efectivamente el abandono.

Como siguiente paso, se recomienda realizar pruebas piloto de las campañas de retención y comparar sus resultados con un grupo de control. Esto permitiría evaluar si las ofertas generan una mejora real en la permanencia de los clientes y si el costo de las campañas se justifica por los beneficios obtenidos.

## 15. Limitaciones y trabajo futuro

**Limitaciones**

- El dataset es histórico y puede no representar a los clientes actuales de una empresa real.
- Los resultados muestran asociaciones predictivas, no relaciones causales.
- Las diferencias entre los tres modelos son pequeñas; la elección de Gradient Boosting se apoya en diferencias mínimas de CV.
- El clustering tiene un Silhouette de 0,235 (estructura moderada a débil) y K-Means usa distancia euclidiana sobre variables *dummy*, que no representa perfectamente variables categóricas.
- Existe una disparidad de falsos positivos en adultos mayores.
- El umbral de 0,50 no se optimizó con costos reales del negocio.
- Las recomendaciones comerciales se derivan de patrones observados y predicciones; no se ha demostrado mediante una prueba experimental que las campañas propuestas reduzcan efectivamente el abandono.

**Trabajo futuro**

- Calibrar el umbral de decisión con costos, márgenes y tasas de aceptación reales de la empresa.
- Probar mitigaciones de sesgo: excluir `SeniorCitizen`, umbrales por grupo o revisión humana.
- Validar el modelo con datos nuevos y monitorear cambios en el comportamiento de los clientes.
- Comparar otras técnicas de segmentación pensadas para variables mixtas (por ejemplo K-Prototypes).

## 16. Instrucciones de ejecución

### Requisitos

- Python y un entorno compatible con las bibliotecas utilizadas.
- Google Colab o Jupyter Notebook.
- El archivo `Telco_Customer_Churn_Clean.csv`.
- Las bibliotecas Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y Joblib.

### Pasos para ejecutar el proyecto

1. Abrir el notebook del proyecto en Google Colab o Jupyter.
2. Ubicar el archivo `Telco_Customer_Churn_Clean.csv` en la carpeta de datos correspondiente.
3. Verificar que la ruta de `pd.read_csv()` coincida con la ubicación real del archivo. En la versión actual del notebook se utiliza `/content/data/Telco_Customer_Churn_Clean.csv`.
4. Ejecutar las celdas en orden y comprobar que no existan errores.
5. Revisar las métricas de clasificación, las tablas de auditoría ética y los perfiles de clustering que se generan.


## Conclusiones generales

El proyecto permitió analizar el abandono de clientes mediante análisis exploratorio, clasificación supervisada y segmentación no supervisada. El EDA identificó diferencias relevantes en las tasas de churn según antigüedad, tipo de contrato, servicios y método de pago. Estos patrones orientaron la selección de variables y la construcción de los modelos.

Entre los clasificadores comparados, Gradient Boosting balanceado fue seleccionado por sus resultados de validación cruzada y su capacidad para detectar clientes que abandonan. En el conjunto de prueba, el Recall registrado fue de 0,807, lo que representa la detección de 302 de los 374 casos reales de abandono. Sin embargo, también se generaron 291 falsos positivos, por lo que la aplicación comercial requiere considerar el costo de contactar clientes que finalmente no abandonan.

En el análisis no supervisado, K-Means con cuatro clusters permitió describir segmentos con diferencias en antigüedad, facturación y tasa de abandono. La selección de cuatro grupos prioriza la utilidad comercial de los perfiles, reconociendo que su separación matemática es limitada y que los segmentos no constituyen categorías exactas.

La revisión ética identificó diferencias de desempeño por género y una disparidad importante en la tasa de falsos positivos de adultos mayores. Estos resultados deben considerarse en futuras evaluaciones del modelo, sin asumir que excluir una variable sensible elimina todos los posibles sesgos.

En conjunto, la solución puede apoyar la priorización de clientes para estrategias de retención, pero no demuestra causalidad ni la efectividad de campañas específicas. Como trabajo futuro, se propone validar las estrategias mediante pruebas piloto, revisar los umbrales de decisión, profundizar el análisis de equidad y comprobar el comportamiento del modelo con nuevos datos.
