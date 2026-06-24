# GalaxyMorph ML

**Analisis y clasificacion de la morfologia galactica utilizando tecnicas de aprendizaje profundo basadas en redes neuronales convolucionales**

Proyecto academico — Universidad Privada Antenor Orrego (UPAO) · Curso: Inteligencia Artificial Principios y Tecnicas

**Equipo:** Morales Robles Jeancarlo · Leon Garcia Axel Erico · Tarazona Flores Jose Ricardo · Docente: Hernan Sagastegui Chigne

---

## Resumen

GalaxyMorph ML implementa un **experimento comparativo controlado** de cuatro arquitecturas de aprendizaje profundo — **ResNet-50**, **EfficientNet-B3**, **Swin-S** y **MaxViT-T** — para clasificar la morfologia de galaxias del catalogo **Galaxy Zoo 2** en **6 clases**. El dataset final comprende **111,129 imagenes** tras umbralizacion de confianza (theta >= 0.6) y soft cap por clase (25,000). Los cuatro modelos se entrenaron **localmente en GPU NVIDIA RTX 5060 Ti** bajo condiciones identicas de particion, perdida ponderada y criterio de checkpoint; la evaluacion final se realiza sobre un test set aislado de **16,670 galaxias** en `notebooks/09_evaluation.ipynb`.

> **Evolucion del proyecto.** En iteraciones tempranas se exploraron arquitecturas individuales (ResNet-50 basico, hibrido EfficientNet-B0 + Transformer, ConvNeXt-Tiny). El enfoque definitivo —documentado en el informe academico y en este repositorio— es la **comparativa sistematica de las cuatro arquitecturas** anteriores, entrenadas y evaluadas bajo el mismo protocolo experimental.

---

![Dataset Cover](docs/dataset-cover.png)

---

## Tabla de contenidos

0. [Resumen](#resumen)
1. [Contexto del proyecto](#1-contexto-del-proyecto)
2. [Problematica](#2-problematica)
3. [Fundamento cientifico](#3-fundamento-cientifico)
4. [Objetivos](#4-objetivos)
5. [Requisitos del sistema](#5-requisitos-del-sistema)
6. [Paradigma de aprendizaje y solucion propuesta](#6-paradigma-de-aprendizaje-y-solucion-propuesta)
7. [Dataset](#7-dataset)
8. [Pipeline de notebooks](#8-pipeline-de-notebooks)
9. [Modelos y arquitecturas](#9-modelos-y-arquitecturas)
10. [Resultados comparativos](#10-resultados-comparativos)
11. [Estructura del proyecto](#11-estructura-del-proyecto)
12. [Instalacion y uso](#12-instalacion-y-uso)
13. [Requisitos de hardware](#13-requisitos-de-hardware)
14. [Referencias](#14-referencias)

---

## 1. Contexto del proyecto

La clasificacion morfologica de galaxias es la piedra angular de la astrofisica extragalactica y la cosmologia observacional moderna. Lejos de ser un ejercicio taxonomico, la morfologia codifica informacion fisica vital sobre la historia de formacion de cada galaxia, la dinamica orbital de sus poblaciones estelares, su contenido de gas interestelar y la evolucion estructural del universo a lo largo del tiempo cosmico. La apariencia visual de una galaxia es el resultado de las relaciones entre materia oscura, materia baryonica y los procesos termodinamicos que operan a escala galactica: comprender si exhibe brazos espirales con formacion estelar activa o un elipsoide dominado por estrellas antiguas permite reconstruir historiales de fusiones, distribucion de materia oscura y tasas de formacion estelar a distintos corrimientos al rojo.

![Hubble-de Vaucouleurs classification scheme](docs/hubble-de-vaucouleurs.png)

### Historia y evolucion del esquema clasificatorio

La astrofisica extragalactica nacio cuando se comprendio que el universo se extiende mas alla de la Via Lactea. Con el telescopio y los grandes relevamientos fotograficos, Edwin Hubble (1926) introdujo el primer esquema formal —el "Diapason de Hubble"— dividiendo las galaxias en elipticas, espirales e irregulares. Gerhard de Vaucouleurs (1959) extendio este modelo con el indice continuo de tipo T y subtipos para espirales barradas y estructuras en anillo. Allan Sandage reconocio formalmente las lenticulares (S0) como clase de transicion. Esta evolucion refleja la naturaleza intrinsecamente continua de la morfologia galactica: las fronteras entre categorias son graduales, no discretas.

La forma de una galaxia es un registro de su historia. Las galaxias evolucionan por fusiones y colisiones; cuando dos espirales interactuan, sus orbitas ordenadas se vuelven caoticas y pueden originar elipsoides. Clasificar morfologias —detectar colas de marea, puentes de gas o asimetrias— es el primer paso para cuantificar la dinamica del universo.

### Escala de los relevamientos modernos

La tasa de captura fotografica de galaxias ha superado la capacidad cognitiva humana. El telescopio espacial **Euclid** (ESA, 2023) fotografio 1.2 millones de galaxias en su primer ano; en una sola liberacion anticipada presento 380,000 galaxias en apenas 63 grados cuadrados del cielo (~0.4% del cielo que mapeara hasta 2030). En superficie, el **Observatorio Vera C. Rubin** (LSST, Chile) producira **10 TB de datos crudos por noche**, emitira ~10 millones de alertas transitorias diarias y consolidara, en 10 anos de operacion, una base de datos de **15 petabytes** con **20,000 millones de galaxias** catalogadas. A esta escala, la clasificacion visual humana es matematicamente inviable.

---

## 2. Problematica

### 2.1. Problema a resolver

Los telescopios modernos como el LSST generaran catalogos de hasta **20,000 millones de galaxias**, mientras que toda la comunidad astronomica mundial (~200,000 profesionales) tardo **3 anos** en clasificar manualmente apenas **300,000 galaxias** en Galaxy Zoo. Este abismo hace inviable la clasificacion morfologica manual a escala actual.

Adicionalmente, el proceso humano introduce **sesgos sistematicos**. El corrimiento al rojo (redshift) cosmologico distorsiona la percepcion visual: las galaxias lejanas se ven mas pequenas y tenues, y los clasificadores pasan por alto brazos espirales finos clasificandolas erroneamente como esferas difusas. El catalogo de Hart et al. (2016) corrige este sesgo mediante fracciones de voto debiased, pero la escala del problema exige automatizacion.

La ausencia de un **sistema automatizado, preciso y escalable** constituye un cuello de botella critico que impide el aprovechamiento cientifico de los datos astronomicos modernos.

### 2.2. Demografia y capital humano

La astronomia profesional es una disciplina altamente especializada y demograficamente reducida. La Union Astronomica Internacional (IAU) registra ~256,000 miembros en 92 paises; la comunidad activa de investigadores ronda los **200,000 profesionales** — comparable al tamano de un pueblo mediano frente a la escala del cosmos.

Si el LSST catalogara 20,000 millones de galaxias, cada astronomo del planeta tendria que evaluar **100,000 imagenes** manualmente. Incluso dedicando 24 horas al dia sin descanso, el tiempo requerido superaria con creces la duracion de una carrera academica.

**Galaxy Zoo** (2007) demostro el limite de la fuerza bruta humana incluso con ciencia ciudadana: ~80,000 voluntarios clasificaron mas de 10 millones de imagenes en el primer proyecto; GZ2 movilizo a mas de 83,000 voluntarios con ~16 millones de clasificaciones sobre ~300,000 galaxias del SDSS. Aun asi, al ritmo de Galaxy Zoo, clasificar los catalogos del LSST tomaria **decenas de miles de anos**.

El costo de clasificacion manual profesional es igualmente prohibitivo: los astronomos especializados requieren doctorado y anos de formacion postdoctoral, con costos salariales elevados para las agencias espaciales e instituciones academicas. En un contexto de presupuestos ajustados —la NASA opero con ~24.4 mil millones de dolares en FY2026, con recortes propuestos a la Division de Astrofisica—, la clasificacion manual a escala de relevamiento no es sostenible.

### 2.3. Desafios tecnicos de la clasificacion automatica

La clasificacion automatica de morfologia galactica presenta desafios que la distinguen de la vision por computadora convencional:

**Desbalance de clases severo.** La distribucion de morfologias en el universo no es uniforme. En el dataset de este proyecto, tras el soft cap, la clase minoritaria (Irregular, 5,927) tiene **4.2x** menos representacion que las clases mayoritarias (25,000), sesgando los clasificadores si no se aplica correccion.

**Ambiguedad en la frontera E/S0.** La transicion entre elipticas (E) y lenticulares (S0) es intrinsecamente continua. Incluso clasificadores humanos entrenados discrepan en esta frontera. Es un limite fisico del problema, no un defecto del metodo.

**Sesgo por corrimiento al rojo.** Las galaxias distantes pierden detalle morfologico fino. El catalogo Hart et al. (2016) aplica correccion debiased, pero la ambiguedad residual persiste en las fronteras de clase.

**Invariancia a orientacion y escala.** Las galaxias no tienen orientacion canonica; una espiral vista de canto es morfologicamente indistinguible de una lenticular sin informacion espectroscopica complementaria.

**Gradiente de dificultad morfologica.** Edge_on tiene un sello visual inequivoco (disco fino, banda oscura central). Lenticular e Irregular requieren capturar caracteristicas de escala global inherentemente mas dificiles de codificar.

---

## 3. Fundamento cientifico

### 3.1. Esquema taxonomico y clases morfologicas

El esquema de clasificacion empleado sigue la secuencia de **Hubble-de Vaucouleurs**, con seis categorias derivadas del arbol de decision morfologico de Galaxy Zoo 2:

| Indice | Clase         | Descripcion morfologica                                                                                                                                                         |
| ------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | Elliptical    | Perfil de brillo suave tipo de Vaucouleurs (n~4), sin estructura de disco, sin brazos ni barra. Rotacion dominada por dispersion de velocidades. Poblaciones estelares antiguas |
| 1      | Lenticular    | Disco con abultamiento central (bulbo), sin brazos espirales detectables. Transicion dinamica entre elipticas y espirales. Poco gas interestelar                                |
| 2      | Spiral        | Brazos espirales bien definidos con formacion estelar activa, disco dominante, sin barra central evidente                                                                       |
| 3      | Barred_Spiral | Estructura de barra central con brazos espirales que emergen de sus extremos. La barra es una concentracion lineal de estrellas que atraviesa el nucleo                         |
| 4      | Edge_on       | Galaxia de disco vista de canto (~90 grados de inclinacion). El disco delgado y el abultamiento central son visibles, pero la presencia o ausencia de brazos no es distinguible |
| 5      | Irregular     | Morfologia perturbada, asimetrica o en interaccion. No encaja en ninguna categoria del diagrama de Hubble. Frecuentemente resultado de fusiones o mareas gravitacionales        |

### 3.2. Galaxy Zoo 2 y el catalogo Hart et al. (2016)

**Galaxy Zoo 2** amplio las clasificaciones de su predecesor para ~300,000 galaxias del SDSS, midiendo barras, brazos espirales, inclinacion, bulbos y otras caracteristicas mediante un arbol de decision jerarquico. Mas de 83,000 voluntarios realizaron ~16 millones de clasificaciones.

Las imagenes provienen del **Sloan Digital Sky Survey (SDSS DR7)**: telescopio de 2.5 m en Apache Point, imagenes JPEG de 424x424 px compuestas con filtros fotometricos (u, g, r, i, z). El reto de GZ2, documentado por Willett et al. (2013), fue el sesgo optico del redshift: galaxias lejanas parecen mas difusas y se confunden con elipticas.

Hart et al. (2016) publicaron fracciones de voto **debiased** para ~240,000 galaxias, ajustadas matematicamente para simular como habrian votado los humanos si todas estuvieran a distancia y claridad ideales. Cada galaxia se identifica por su `dr7objid` del SDSS.

### 3.3. Etiquetado y umbral de confianza

Las etiquetas se derivan aplicando el arbol de decision jerarquico de GZ2 con umbral **theta = 0.6** sobre las fracciones debiased: al menos el 60% de los voluntarios deben haber respondido coherentemente en la misma direccion del arbol. Las galaxias con votos difusos se excluyen como `Uncertain`. Los artefactos fotograficos reciben prioridad maxima y se descartan.

El sistema de prioridades jerarquicas asigna reglas morfologicas especificas antes que la etiqueta Irregular (prioridad mas baja), evitando que anomalias locales sobreescriban la morfologia principal.

---

## 4. Objetivos

### 4.1. Objetivo general

Disenar e implementar un sistema de clasificacion automatica de morfologia galactica basado en **redes neuronales convolucionales (CNN) y Vision Transformers**, capaz de procesar de forma masiva y reproducible los catalogos astronomicos modernos con precision comparable al consenso humano experto, **evaluando y comparando cuatro arquitecturas de aprendizaje profundo** bajo condiciones identicas de entrenamiento y evaluacion.

### 4.2. Objetivos especificos

**Construccion y curacion del dataset.** Consolidar un dataset astronomico etiquetado y balanceado a partir del catalogo Galaxy Zoo 2, aplicando umbralizacion de confianza (theta >= 0.6), correccion del sesgo por redshift (fracciones debiased) y particion estratificada en subconjuntos de entrenamiento, validacion y prueba.

**Diseno del pipeline de preprocesamiento.** Desarrollar un flujo de normalizacion de imagenes que corrija ruido luminico, artefactos instrumentales y desequilibrio de clases, garantizando la calidad y reproducibilidad de los datos de entrada al modelo.

**Implementacion de arquitecturas de aprendizaje profundo.** Implementar, configurar y entrenar **cuatro arquitecturas** — ResNet-50, EfficientNet-B3, Swin-S y MaxViT-T — adaptadas a la clasificacion de imagenes astronomicas, con tecnicas de aumento de datos y fine-tuning diferencial desde pesos preentrenados en ImageNet.

**Evaluacion y validacion comparativa.** Comparar el rendimiento de las arquitecturas implementadas mediante metricas de clasificacion (F1-macro, precision y recall por clase), determinando el modelo con mayor capacidad de generalizacion ante datos no vistos y analizando la relacion entre rendimiento y costo computacional.

---

## 5. Requisitos del sistema

### 5.1. Definicion del dominio

**Dominio astrofisico.** El sistema trabaja con imagenes fotometricas de galaxias extragalacticas del universo cercano, capturadas en el espectro visible e infrarrojo cercano (SDSS). El sistema taxonomico de referencia es el esquema continuo de Hubble-de Vaucouleurs.

**Dominio computacional.** El sistema opera sobre imagenes digitales de 424x424 pixeles en tres canales RGB. La tarea es una **clasificacion supervisada multiclase** de seis categorias morfologicas. Las tecnicas empleadas pertenecen al aprendizaje profundo: CNN y arquitecturas modernas derivadas (Vision Transformers).

### 5.2. Requisitos funcionales

| ID    | Requisito                                                                                                                                        |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| RF-01 | El sistema debe aceptar como entrada imagenes JPEG de galaxias de 424x424 pixeles                                                                |
| RF-02 | El sistema debe clasificar cada imagen en una de seis categorias morfologicas: Elliptical, Lenticular, Spiral, Barred_Spiral, Edge_on, Irregular |
| RF-03 | El sistema debe generar una probabilidad de pertenencia por clase para cada imagen analizada (vector de logits softmax)                          |
| RF-04 | El sistema debe poder procesar multiples imagenes en modo batch para inferencia eficiente                                                        |
| RF-05 | El sistema debe producir reportes de metricas de evaluacion: F1-macro, matriz de confusion, precision y recall por clase                         |

### 5.3. Requisitos no funcionales

| ID     | Categoria        | Requisito                                                                                                          |
| ------ | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| RNF-01 | Precision        | El modelo debe alcanzar un F1-macro >= 0.75 sobre el conjunto de prueba (objetivo de diseno; ver resultados en seccion 10) |
| RNF-02 | Escalabilidad    | El sistema debe ser capaz de procesar al menos una imagen por segundo en fase de inferencia sobre hardware con GPU |
| RNF-03 | Reproducibilidad | El pipeline de preprocesamiento y particion del dataset debe ser completamente determinista (semilla fija = 42)   |
| RNF-04 | Trazabilidad     | Cada muestra debe ser identificable hasta su fuente original mediante el identificador SDSS (dr7objid)             |

### 5.4. Requisitos de datos

| ID    | Requisito                                                                                                             |
| ----- | --------------------------------------------------------------------------------------------------------------------- |
| RD-01 | Dataset fuente: Galaxy Zoo 2 (catalogo de Hart et al., 2016), con imagenes del SDSS DR7                               |
| RD-02 | Minimo de muestras por clase segun disponibilidad del catalogo (clase Irregular: 5,927 muestras tras soft cap)        |
| RD-03 | Umbral de confianza para etiquetado: >= 60% de consenso entre clasificadores humanos sobre fracciones debiased        |
| RD-04 | Distribucion de particion: 70% entrenamiento / 15% validacion / 15% prueba, con estratificacion por clase morfologica |

---

## 6. Paradigma de aprendizaje y solucion propuesta

### 6.1. Contribuciones del proyecto

Este proyecto implementa y evalua un pipeline completo de clasificacion automatica de morfologia galactica:

- **Etiquetado sistematico** de 111,129 galaxias del catalogo Galaxy Zoo 2 aplicando el arbol de decision morfologico con umbral de confianza calibrado.
- **Particion estratificada** reproducible del dataset (70/15/15) preservando la distribucion de clases.
- **Comparativa de cuatro arquitecturas** (ResNet-50, EfficientNet-B3, Swin-S, MaxViT-T) entrenadas localmente bajo identicas condiciones de regularizacion, optimizacion y evaluacion.
- **Infraestructura de entrenamiento reproducible** con AMP (mixed precision), early stopping, checkpointing por epoca y reanudacion sin perdida de estado.
- **Evaluacion cuantitativa** sobre test set aislado con F1-macro, accuracy, matrices de confusion y analisis de eficiencia parametros/rendimiento.

### 6.2. El agente inteligente

El sistema se modela como un **agente de clasificacion perceptual**: recibe una imagen de galaxia como entrada sensorial y emite una etiqueta morfologica discreta como accion. No planifica secuencias ni razona simbolicamente; transforma una entrada de alta dimensionalidad en una decision categórica mediante una funcion no lineal aprendida (red neuronal profunda). El entorno es estatico, episodico y completamente observable — un escenario de clasificacion supervisada directa.

### 6.3. Descarte de alternativas

**IA simbolica / sistemas expertos.** Un sistema de reglas explicitas requeriria codificar condiciones del tipo «si fraccion de barra > 0.6 y apertura > 45°, clasificar como espiral barrada». Tres obstaculos lo hacen inviable: (1) ambiguedad intrinseca — las fracciones de voto son continuas, no etiquetas discretas; (2) explosion combinatoria del espacio de caracteristicas (orientacion, distancia, brillo, artefactos); (3) inconsistencia del criterio humano documentada por Willett et al. (2013) y Hart et al. (2016).

**Aprendizaje no supervisado.** Los metodos de clustering no garantizan correspondencia con las categorias de Hubble-de Vaucouleurs; podrian agrupar por brillo o tamano en lugar de morfologia intrinseca.

**Aprendizaje por refuerzo.** No existe un entorno interactivo donde el agente modifique el estado del mundo ni recompensas diferidas.

**Machine learning clasico.** SVM, Random Forest o Gradient Boosting requieren extraccion manual de caracteristicas (indice de concentracion C, asimetria A, Gini, elipticidad isofotal), reproduciendo el sesgo humano que el proyecto pretende eliminar. Una imagen 424x424x3 es un vector de 539,328 dimensiones donde los clasificadores clasicos sobre pixeles crudos no son viables.

### 6.4. Aprendizaje profundo supervisado

El problema se formaliza como **aprendizaje supervisado** sobre 111,129 pares (imagen, etiqueta_morfologica) derivados del consenso humano de Galaxy Zoo. Las **redes neuronales convolucionales (CNN)** aprenden representaciones jerarquicas directamente de los pixeles, con tres propiedades clave para imagenes astronomicas:

- **Invariancia a la traslacion:** una espiral se reconoce independientemente de la posicion del nucleo en la imagen.
- **Jerarquia de representaciones:** capas bajas detectan bordes y texturas; capas profundas integran patrones morfologicos (brazos, barras, bulbos).
- **Comparticion de parametros:** viabiliza el procesamiento de imagenes de alta resolucion con menor riesgo de sobreajuste.

La literatura respalda esta eleccion: Cheng et al. (2021, 2023) reportan F1 > 0.85 con CNN sobre el Dark Energy Survey; Katsaros et al. (2025) consolidan estos resultados en una revision sistematica.

### 6.5. Transfer learning

Entrenar desde cero requiere millones de imagenes; el conjunto de entrenamiento (77,789 muestras) es insuficiente para arquitecturas de esta escala. Las cuatro arquitecturas se inicializan con pesos **ImageNet-1K** (1.2 M imagenes, 1,000 clases): los detectores de bordes, texturas y gradientes de las capas bajas son representaciones visuales universales transferibles al dominio astronomico.

El fine-tuning reemplaza la cabeza clasificadora (1,000 → 6 salidas) y entrena la red completa: el backbone ajusta sus representaciones al dominio galactico mientras la cabeza aprende las fronteras morfologicas.

### 6.6. Seleccion de las cuatro arquitecturas

Se seleccionaron cuatro arquitecturas que representan paradigmas distintos, manteniendo constante el dataset, el protocolo de entrenamiento y los hiperparametros base:

| Arquitectura    | Paradigma                  | Parametros (~) | Entrada  | Justificacion                                                                 |
| --------------- | -------------------------- | -------------- | -------- | ----------------------------------------------------------------------------- |
| ResNet-50       | CNN residual clasica       | 25.6 M         | 224 px   | Baseline historico obligatorio; comparabilidad con la literatura astronomica  |
| EfficientNet-B3 | CNN de escalado compuesto  | 12.2 M         | 224 px   | Mejor relacion eficiencia/rendimiento por parametro (Cheng et al., 2021)      |
| Swin-S          | Vision Transformer jerarquico| 49.7 M       | 308 px   | Atencion por ventanas; captura dependencias espaciales de mediano alcance     |
| MaxViT-T        | Transformer multi-escala   | 30.9 M         | 224 px   | Atencion local + global simultanea; textura fina y forma global del objeto    |

La unica variable independiente del experimento es la **arquitectura**; las diferencias en metricas finales se atribuyen a diferencias arquitectonicas, no a artefactos del entrenamiento.

---

## 7. Dataset

**Galaxy Zoo 2** es un proyecto de ciencia ciudadana que recogio clasificaciones morfologicas detalladas de ~300,000 galaxias del Sloan Digital Sky Survey (SDSS). Las clasificaciones se generaron mediante votacion de voluntarios a traves de 11 preguntas jerarquicas sobre la morfologia de cada objeto. El catalogo publicado por Hart et al. (2016) aplica una correccion del sesgo por corrimiento al rojo, produciendo fracciones de voto debiased que simulan como habrian votado los clasificadores si todas las galaxias estuvieran a una distancia estandar.

| Propiedad                              | Valor                                              |
| -------------------------------------- | -------------------------------------------------- |
| Fuente                                 | Galaxy Zoo 2 (Hart et al., 2016)                   |
| Catalogo base                          | SDSS DR7                                           |
| Galaxias en catalogo Hart16            | 239,695                                            |
| Galaxias con etiqueta valida + imagen  | 169,531 (antes del soft cap)                       |
| Galaxias tras soft cap (theta=0.6)     | **111,129**                                        |
| Imagenes JPEG disponibles (Kaggle)     | ~243,000 archivos (424x424 px, 3 canales RGB)      |
| Clases morfologicas                    | 6                                                  |
| Particion entrenamiento                | 77,789 (70.0%)                                     |
| Particion validacion                   | 16,670 (15.0%)                                     |
| Particion test                         | 16,670 (15.0%)                                     |
| Estratificacion                        | Si, por clase morfologica                          |

**Pipeline de curacion del dataset** (`notebooks/02_dataset_preparation.ipynb`):

1. Carga del catalogo `gz2_hart16.csv` (239,695 galaxias) y asignacion de etiquetas mediante el arbol de decision GZ2 con umbral theta = 0.6.
2. Filtrado de galaxias sin imagen en disco o con etiqueta `Uncertain`/`Artifact`.
3. Cruce con el mapeo `gz2_filename_mapping.csv` (sample `original`) → **169,531** galaxias con etiqueta valida e imagen disponible.
4. Soft cap de 25,000 muestras por clase sobre las categorias dominantes → **111,129** galaxias en el dataset final.
5. Particion estratificada 70/15/15 exportada a `data/splits/`.

### 7.1. Metodologia de etiquetado

Cada galaxia recorre el arbol de decision jerarquico de GZ2. El umbral de confianza theta = 0.6 determina si una fraccion de voto debiased es suficientemente consensuada para propagar la clasificacion al siguiente nivel del arbol. Se aplica el siguiente sistema de prioridad jerarquica:

1. Si la fraccion de voto para "disco" supera theta, la galaxia entra en la rama de espirales o lenticulares.
2. Dentro de la rama de espirales, se determina si existe barra central (Barred_Spiral vs. Spiral).
3. Si la fraccion de voto para "forma redondeada" supera theta sin estructura de disco, la galaxia es clasificada como Elliptical.
4. Las galaxias con inclinacion >60 grados (mayoria de voto para "visto de canto") se asignan a Edge_on independientemente de sus brazos.
5. Irregular se asigna con la prioridad mas baja: galaxias que ninguna otra rama del arbol reclama con confianza suficiente.
6. Las galaxias que no superan theta en ninguna rama del arbol se excluyen como casos de ambiguedad intrinseca.

Para evitar que las clases mas frecuentes dominen el entrenamiento y distorsionen las representaciones aprendidas, se aplica un cap suave de 25,000 muestras por clase. Las clases con mas de 25,000 galaxias disponibles (Elliptical, Spiral, Barred_Spiral) se submuestrean mediante muestreo aleatorio estratificado. El resultado es una razon de desbalance residual de 4.2x entre la clase mayoritaria (25,000) y la clase minoritaria (Irregular, ~5,927), que se gestiona durante el entrenamiento mediante pesos de clase en CrossEntropyLoss.

### 7.2. Distribucion de clases

**Distribucion de clases (conjunto etiquetado, post-cap):**

| Clase         | n      | Fraccion | Peso de clase |
| ------------- | ------ | -------- | -------------- |
| Elliptical    | 25,000 | 22.5%    | 0.74           |
| Spiral        | 25,000 | 22.5%    | 0.74           |
| Barred_Spiral | 25,000 | 22.5%    | 0.74           |
| Lenticular    | 16,926 | 15.2%    | 1.09           |
| Edge_on       | 13,276 | 11.9%    | 1.40           |
| Irregular     | 5,927  | 5.3%     | 3.12           |

Los pesos de clase se calculan como la inversa de la frecuencia normalizada respecto a la clase mas frecuente y se aplican a la funcion de perdida (CrossEntropyLoss) durante el entrenamiento. Se prefirio este enfoque sobre el sobremuestreo sintetico de Irregular porque la clase Irregular es intrinsecamente heterogenea: cualquier muestra artificial generada no representaria la variabilidad real de morfologias perturbadas.

### 7.3. Particion y trazabilidad

La particion 70/15/15 se realiza mediante `train_test_split(stratify=morph_label)` para preservar la distribucion de clases en los tres subconjuntos. Los splits se exportan como `train.csv`, `val.csv` y `test.csv` con las columnas `[dr7objid, asset_id, img_filename, morph_label]`, garantizando la trazabilidad de cada muestra hasta su identificador SDSS original (requisito RNF-04). El conjunto de prueba (test set) se mantiene completamente aislado durante todo el proceso de entrenamiento y validacion; solo se utiliza en la evaluacion final del notebook `09_evaluation.ipynb`.

Los archivos del dataset deben colocarse en `data/` siguiendo la estructura indicada en la seccion de instalacion. Las imagenes no se incluyen en este repositorio por su volumen (~12 GB).

---

## 8. Pipeline de notebooks

El proyecto se organiza como una secuencia de notebooks Jupyter con responsabilidades separadas. Cada notebook es autocontenido y puede ejecutarse de forma independiente si sus dependencias (checkpoints, splits) estan disponibles.

### Pipeline principal (experimento comparativo)

| #   | Notebook                               | Descripcion                                                                                                                                                        | Estado   |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| 01  | `01_eda.ipynb`                         | Analisis exploratorio: inspeccion del catalogo CSV, mapeo de IDs, inventario de imagenes, arbol de decision morfologico, distribucion de clases, muestras visuales | Completo |
| 02  | `02_dataset_preparation.ipynb`         | Etiquetado duro (threshold=0.6), soft cap por clase, particion 70/15/15 estratificada, exportacion de splits CSV a `data/splits/`                                   | Completo |
| 03  | `03_image_preprocessing.ipynb`         | Definicion de `GalaxyDataset`, pipeline de transforms, DataLoaders, verificacion de pesos de clase                                                                 | Completo |
| 04L | `04_train_efficientnet_b3_local.ipynb` | Entrenamiento de EfficientNet-B3 (local, RTX 5060 Ti)                                                                                                              | Completo |
| 05  | `05_train_resnet50_local.ipynb`        | Entrenamiento de ResNet-50 (local)                                                                                                                                 | Completo |
| 06  | `06_train_swins_local.ipynb`           | Entrenamiento de Swin-S (local)                                                                                                                                     | Completo |
| 08  | `08_train_maxvit_local.ipynb`          | Entrenamiento de MaxViT-T (local)                                                                                                                                  | Completo |
| 09  | `09_evaluation.ipynb`                  | Evaluacion comparativa de los 4 modelos sobre test set                                                                                                             | Completo |

### Notebooks complementarios

| #   | Notebook                               | Descripcion                                                                                              | Estado      |
| --- | -------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------- |
| 04  | `04_train_efficientnet_b3.ipynb`       | Version historica del entrenamiento en Kaggle (2x GPU T4); conservada como referencia                   | Completo    |
| 07  | `07_train_swint_local.ipynb`           | Exploracion de Swin-T (variante mas ligera); **no forma parte de la comparativa principal**              | Completo    |
| 10  | `10_finetune_zoobot.ipynb`             | Fine-tuning de Zoobot (ConvNeXt-Nano) preentrenado en Galaxy Zoo; trabajo futuro                        | En progreso |

Los cuatro notebooks de entrenamiento del experimento comparativo (04L, 05, 06, 08) comparten la misma estructura: configuracion, pipeline de datos, definicion de modelo, infraestructura de entrenamiento con AMP y early stopping, reanudacion desde checkpoint, loop de entrenamiento, curvas de aprendizaje y resumen de artefactos.

---

## 9. Modelos y arquitecturas

Se entrenaron y compararon cuatro arquitecturas representando distintas familias de modelos para vision por computadora. Los notebooks de entrenamiento de referencia son `04_train_efficientnet_b3_local.ipynb`, `05_train_resnet50_local.ipynb`, `06_train_swins_local.ipynb` y `08_train_maxvit_local.ipynb`. Todos los modelos parten de pesos preentrenados en ImageNet-1K y se ajustan en dos fases: calentamiento de la cabeza de clasificacion seguido de fine-tuning completo del backbone con learning rate diferencial.

### EfficientNet-B3

EfficientNet (Tan y Le, 2019) introduce el escalado compuesto: profundidad, anchura y resolucion se escalan simultaneamente usando coeficientes derivados mediante NAS. B3 es la tercera variante ($\phi=3$), con ~10.7 M parametros. El bloque fundamental es el MBConv con Squeeze-and-Excitation (SE), que recalibra los canales de forma adaptativa.

**Cabeza personalizada:** `classifier[1] = nn.Linear(1536, 6)`

![EfficientNet-B3 Architecture](docs/efficientnet-b3-architecture-cnn.png)

| Parametro           | Valor                                              |
| ------------------- | -------------------------------------------------- |
| Pesos preentrenados | EfficientNet_B3_Weights.IMAGENET1K_V1              |
| IMAGE_SIZE          | 224 px                                             |
| CROP_SIZE           | 320 px                                             |
| BATCH_SIZE          | 128                                                |
| Parametros totales  | ~10.7 M                                            |
| Early stopping      | No (30 epocas completas)                           |
| Documentacion       | [docs/efficientnet_b3.md](docs/efficientnet_b3.md) |

---

### ResNet-50

ResNet (He et al., 2016) introdujo las conexiones residuales (skip connections), que resuelven la degradacion del gradiente en redes profundas permitiendo aprender el residuo $\mathcal{F}(x) = \mathcal{H}(x) - x$ en lugar de la transformacion completa. ResNet-50 usa bloques Bottleneck de tres convoluciones que reducen el costo computacional a ~23.5 M parametros.

**Cabeza personalizada:** `fc = nn.Linear(2048, 6)`

![ResNet-50 Architecture](docs/resnet50-architecture-cnn.png)

| Parametro           | Valor                                |
| ------------------- | ------------------------------------ |
| Pesos preentrenados | ResNet50_Weights.IMAGENET1K_V2       |
| IMAGE_SIZE          | 224 px                               |
| CROP_SIZE           | 280 px                               |
| BATCH_SIZE          | 128                                  |
| Parametros totales  | ~23.5 M                              |
| Early stopping      | Si (patience=5)                      |
| Documentacion       | [docs/resnet50.md](docs/resnet50.md) |

---

### Swin Transformer Small (Swin-S)

Swin Transformer (Liu et al., 2021, ICCV Best Paper) resuelve el coste cuadratico de la atencion global en ViT confinando la atencion dentro de ventanas locales de 7x7 tokens y alternando entre ventanas alineadas (W-MSA) y desplazadas (SW-MSA) para comunicar informacion entre ventanas. Swin-S tiene 18 bloques en el stage 3 (vs. 6 en Swin-T), capturando dependencias espaciales de mayor alcance relevantes para estructuras galacticas.

**Cabeza personalizada:** `head = nn.Linear(768, 6)`

![Swin-S Architecture](docs/swin-s-architecture-cnn.png)

| Parametro           | Valor                          |
| ------------------- | ------------------------------ |
| Pesos preentrenados | Swin_S_Weights.IMAGENET1K_V1   |
| IMAGE_SIZE          | 308 px                         |
| CROP_SIZE           | 380 px                         |
| BATCH_SIZE          | 32                             |
| Parametros totales  | ~49.6 M                        |
| Early stopping      | Si (patience=5)                |
| Documentacion       | [docs/swins.md](docs/swins.md) |

> Swin-S requiere una resolucion de entrada diferente (308 px en lugar de 224 px) porque su arquitectura jerarquica con ventanas de 7x7 tokens y patch size de 4x4 exige que la resolucion sea divisible por 4x7=28.

---

### MaxViT-T

MaxViT (Tu et al., 2022, ECCV) unifica en un unico bloque tres mecanismos: extraccion de caracteristicas convolucionales locales (MBConv), atencion local dentro de ventanas (Block Attention) y atencion global de coste lineal mediante muestreo dilatado (Grid Attention). Esta combinacion multi-eje permite capturar contexto local y global simultaneamente en cada capa, superando la limitacion de Swin que solo comunica ventanas adyacentes.

**Cabeza personalizada:** `classifier[5] = nn.Linear(512, 6)` (reemplaza el ultimo Linear del classifier Sequential)

![MaxViT-T Architecture](docs/maxvit-t-architecture-cnn.png)

| Parametro           | Valor                                |
| ------------------- | ------------------------------------ |
| Pesos preentrenados | MaxVit_T_Weights.IMAGENET1K_V1       |
| IMAGE_SIZE          | 224 px                               |
| CROP_SIZE           | 320 px                               |
| BATCH_SIZE          | 32                                   |
| Parametros totales  | ~30.9 M                              |
| Early stopping      | Si (patience=5)                      |
| Documentacion       | [docs/maxvit_t.md](docs/maxvit_t.md) |

---

## 10. Resultados comparativos

Todos los modelos se evaluan sobre el mismo **test set aislado de 16,670 galaxias** — datos que ningun modelo observo durante entrenamiento ni seleccion de hiperparametros. La metrica principal de comparacion es el **F1-macro en test** (promedio no ponderado del F1 por clase), complementada con accuracy global y recall por clase.

**Configuracion de entrenamiento compartida:**

| Componente               | Valor                                      |
| ------------------------ | ------------------------------------------ |
| Entorno                  | Local — NVIDIA RTX 5060 Ti, CUDA 12.8      |
| Optimizador              | AdamW                                      |
| Learning rate (cabeza)   | 1e-3                                       |
| Learning rate (backbone) | 1e-4                                       |
| Weight decay             | 1e-4                                       |
| Scheduler                | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| Early stopping           | patience=5 (excepto EfficientNet-B3)       |
| Funcion de perdida       | CrossEntropyLoss con pesos de clase        |
| Precision                | AMP float16                                |
| Epocas maximas           | 30                                         |

**Resultados en validacion (criterio de seleccion de `best.pth`):**

| Modelo          | Val F1-macro | Epoca mejor | Epocas totales  | Params  |
| --------------- | ------------ | ----------- | --------------- | ------- |
| Swin-S          | **0.6962**   | 25          | 30 (completo)   | ~49.6 M |
| MaxViT-T        | 0.6951       | 17          | 22 (early stop) | ~30.9 M |
| ResNet-50       | 0.6914       | 14          | 19 (early stop) | ~23.5 M |
| EfficientNet-B3 | 0.6894       | 16          | 30 (completo)   | ~10.7 M |

**Resultados en test set (evaluacion final, ordenado por Test F1-macro):**

| Modelo          | Val F1 | Test F1 | Test Acc. | Params | T. entrenamiento |
| --------------- | ------ | ------- | --------- | ------ | ---------------- |
| MaxViT-T        | 0.6951 | **0.6839** | 0.7130 | ~30.9 M | 6.1 h          |
| Swin-S          | 0.6962 | 0.6834  | 0.7128    | ~49.6 M | 15.9 h         |
| EfficientNet-B3 | 0.6894 | 0.6750  | 0.7013    | ~10.7 M | 10.8 h         |
| ResNet-50       | 0.6914 | 0.6706  | 0.6906    | ~23.5 M | 3.6 h          |

**Recall por clase en test set (mejores checkpoints):**

| Clase         | EfficientNet-B3 | ResNet-50 | Swin-S | MaxViT-T |
| ------------- | --------------- | --------- | ------ | -------- |
| Elliptical    | 0.786           | 0.626     | 0.793  | 0.789    |
| Lenticular    | 0.494           | 0.592     | 0.511  | 0.499    |
| Spiral        | 0.604           | 0.656     | 0.642  | 0.650    |
| Barred_Spiral | 0.744           | 0.770     | 0.762  | 0.780    |
| Edge_on       | 0.944           | 0.921     | 0.928  | 0.910    |
| Irregular     | 0.622           | 0.539     | 0.562  | 0.546    |

**Observaciones clave:**

- **MaxViT-T** obtiene el mayor Test F1-macro (0.6839) y accuracy (0.7130), a pesar de que Swin-S lideraba en validacion — su atencion multi-escala generaliza mejor a datos no vistos.
- **EfficientNet-B3** ofrece la mejor relacion rendimiento/parametros: Test F1 de 0.6750 con solo ~10.7 M parametros, ideal para despliegue con restricciones de computo.
- La brecha entre el mejor y el peor modelo es de solo **0.0133 puntos** de F1-macro en test, mientras el coste computacional varia hasta x4.4 en tiempo de entrenamiento.
- **Lenticular** es la clase mas dificil (recall <= 0.592): la ambiguedad morfologica con Elliptical es un limite intrinseco del problema, no de la arquitectura.
- **Edge_on** es la mas robusta en todos los modelos (recall >= 0.91) por su perfil discoidal inequivoco.

La evaluacion comparativa completa — matrices de confusion, curvas de entrenamiento y analisis de eficiencia — se realiza en `notebooks/09_evaluation.ipynb`.

---

## 11. Estructura del proyecto

```
galaxy-morph-ml/
|
+-- data/
|   +-- gz2_hart16.csv              # Catalogo GZ2 con fracciones de voto debiased
|   +-- gz2_filename_mapping.csv    # Mapeo objid -> asset_id (nombre de archivo)
|   +-- images_gz2/
|   |   +-- images/                 # Imagenes JPEG (424x424, ~243k archivos)
|   +-- splits/                     # Generado por 02_dataset_preparation.ipynb
|       +-- train.csv
|       +-- val.csv
|       +-- test.csv
|
+-- docs/
|   +-- efficientnet_b3.md          # Documentacion tecnica EfficientNet-B3
|   +-- resnet50.md                 # Documentacion tecnica ResNet-50
|   +-- swins.md                    # Documentacion tecnica Swin-S
|   +-- maxvit_t.md                 # Documentacion tecnica MaxViT-T
|   +-- efficientnet-b3-architecture-cnn.png
|   +-- resnet50-architecture-cnn.png
|   +-- swin-s-architecture-cnn.png
|   +-- maxvit-t-architecture-cnn.png
|   +-- dataset-cover.png
|   +-- hubble-de-vaucouleurs.png
|
+-- logs/
|   +-- {model_name}_log.csv        # Registro de entrenamiento por epoca
|   +-- {model_name}_training_curves.png
|   +-- {model_name}_confusion_matrix.png
|   +-- evaluation_*.png            # Graficas comparativas (09_evaluation.ipynb)
|   +-- evaluation_summary.csv
|
+-- models/
|   +-- checkpoints/
|       +-- efficientnet_b3/        # latest.pth, best.pth, epoch_XXX.pth
|       +-- resnet50/
|       +-- swin_s/
|       +-- swin_t/
|       +-- maxvit_t/
|
+-- notebooks/
|   +-- 01_eda.ipynb
|   +-- 02_dataset_preparation.ipynb
|   +-- 03_image_preprocessing.ipynb
|   +-- 04_train_efficientnet_b3_local.ipynb   # Experimento comparativo
|   +-- 04_train_efficientnet_b3.ipynb          # Version historica Kaggle
|   +-- 05_train_resnet50_local.ipynb
|   +-- 06_train_swins_local.ipynb
|   +-- 07_train_swint_local.ipynb              # Complementario (no en comparativa)
|   +-- 08_train_maxvit_local.ipynb
|   +-- 09_evaluation.ipynb
|   +-- 10_finetune_zoobot.ipynb                # Trabajo futuro
|
+-- requirements.txt
+-- README.md
```

Los checkpoints (`models/checkpoints/`) y las imagenes del dataset (`data/images_gz2/`) no se versionan en el repositorio por su volumen. Los splits CSV generados por el notebook 02 se incluyen en `data/splits/` para garantizar la reproducibilidad de los experimentos.

---

## 12. Instalacion y uso

### Clonar el repositorio

```bash
git clone https://github.com/jeancdevx/galaxy-morph-ml.git
cd galaxy-morph-ml
```

### Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy matplotlib seaborn scikit-learn Pillow tqdm jupyter
```

> El flag `--index-url https://download.pytorch.org/whl/cu128` instala PyTorch con soporte CUDA 12.8. Para otras versiones de CUDA, consultar [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Obtener el dataset

1. Descargar las imagenes de Galaxy Zoo 2 desde Kaggle:
   [https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)

2. Descargar el catalogo de clasificaciones (Hart et al., 2016):
   [https://data.galaxyzoo.org/](https://data.galaxyzoo.org/) (archivo `gz2_hart16.csv`)

3. Colocar los archivos segun la estructura indicada en la seccion anterior:

```bash
data/
  gz2_hart16.csv
  gz2_filename_mapping.csv
  images_gz2/images/   # ~243k archivos .jpg
```

### Obtener los modelos entrenados

Los checkpoints `best.pth` de los cuatro modelos del experimento comparativo estan publicados en **Kaggle Models** para facilitar su descarga y replicacion de la evaluacion. El entrenamiento reportado en el informe se realizo **localmente** en RTX 5060 Ti; Kaggle se utiliza unicamente como canal de distribucion de artefactos.

| Modelo          | Enlace                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------- |
| ResNet-50       | [galaxy-morph-resnet50](https://www.kaggle.com/models/jeancdevx/galaxy-morph-resnet50)             |
| EfficientNet-B3 | [galaxy-morph-efficientnet-b3](https://www.kaggle.com/models/jeancdevx/galaxy-morph-efficientnet-b3) |
| Swin-S          | [galaxy-morph-swin-s](https://www.kaggle.com/models/jeancdevx/galaxy-morph-swin-s)                 |
| MaxViT-T        | [galaxy-morph-maxvit-t](https://www.kaggle.com/models/jeancdevx/galaxy-morph-maxvit-t)             |

Descargar cada `best.pth` y colocarlo en el directorio correspondiente dentro de `models/checkpoints/`:

```
models/checkpoints/resnet50/best.pth
models/checkpoints/efficientnet_b3/best.pth
models/checkpoints/swin_s/best.pth
models/checkpoints/maxvit_t/best.pth
```

### Ejecutar el pipeline

Ejecutar los notebooks en orden desde Jupyter:

```bash
jupyter lab
```

Los notebooks 01 y 02 deben ejecutarse primero para generar los splits en `data/splits/`. Los notebooks de entrenamiento del experimento comparativo (`04_train_efficientnet_b3_local`, `05`, `06`, `08`) son independientes entre si una vez que los splits existen. El notebook `09_evaluation.ipynb` requiere los checkpoints `best.pth` de los cuatro modelos en `models/checkpoints/` (ver seccion [Obtener los modelos entrenados](#obtener-los-modelos-entrenados)).

### Reanudar entrenamiento desde checkpoint

Cada notebook de entrenamiento detecta automaticamente el ultimo checkpoint disponible en su `CKPT_DIR` y reanuda el entrenamiento desde ahi. No se requiere ninguna accion manual: basta con re-ejecutar el notebook.

---

## 13. Requisitos de hardware

Los experimentos del **experimento comparativo** (notebooks 04L, 05, 06, 08 y 09) se ejecutaron en la siguiente configuracion local:

| Componente | Especificacion                               |
| ---------- | -------------------------------------------- |
| GPU        | NVIDIA RTX 5060 Ti (Blackwell GB206, sm_120) |
| VRAM       | 16 GB GDDR7                                  |
| CUDA       | 12.8                                         |
| PyTorch    | >= 2.7.0                                     |
| SO         | Linux (Ubuntu 24.04)                         |

> Existe una version historica del entrenamiento de EfficientNet-B3 en Kaggle (`04_train_efficientnet_b3.ipynb`, 2x GPU T4). No forma parte del protocolo experimental documentado en el informe; el entrenamiento de referencia es la version local.

Los notebooks de entrenamiento usan AMP (Automatic Mixed Precision, float16), lo que reduce el consumo de VRAM aproximadamente a la mitad. Los batch sizes configurados (128 para CNN, 32 para Transformers) requieren un minimo de 8–16 GB de VRAM segun la arquitectura. Para GPUs con menos memoria, reducir `BATCH_SIZE` a la mitad.

El notebook 09 (evaluacion) carga los modelos secuencialmente y libera la VRAM entre evaluaciones, por lo que sus requisitos de memoria son equivalentes a un unico modelo de los entrenados.

---

## 14. Referencias

- Tan, M., y Le, Q. V. (2019). _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

- He, K., Zhang, X., Ren, S., y Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., y Guo, B. (2021). _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_. ICCV 2021 (Best Paper). [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)

- Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., y Li, Y. (2022). _MaxViT: Multi-Axis Vision Transformer_. ECCV 2022. [arXiv:2204.01697](https://arxiv.org/abs/2204.01697)

- Lintott, C., Schawinski, K., Bamford, S., et al. (2011). _Galaxy Zoo 1: Data Release of Morphological Classifications for nearly 900,000 Galaxies_. MNRAS, 410(1), 166-178.

- Hart, R. E., Bamford, S. P., Willett, K. W., et al. (2016). _Galaxy Zoo 2: Detailed Morphological Classifications for 304,122 Galaxies from the Sloan Digital Sky Survey_. MNRAS, 461(4), 3663-3682.

- Willett, K. W., Lintott, C. J., Bamford, S. P., et al. (2013). _Galaxy Zoo 2: Detailed Morphological Classifications for 304,122 Galaxies from the Sloan Digital Sky Survey_. MNRAS, 435(4), 2835-2860.

- Hubble, E. P. (1926). _Extra-galactic nebulae_. The Astrophysical Journal, 64, 321-369.

- de Vaucouleurs, G. (1959). _Classification and Morphology of External Galaxies_. Handbuch der Physik, 53, 275-310.

- Cheng, T., Guo, X., Shu, X., et al. (2021). _Galaxy Morphology Classification with Efficient CNNs_. ApJ. [arXiv:2105.07362](https://arxiv.org/abs/2105.07362)

- Katsaros, D., Zacharia, N., Kravariti, S.-D., y Papakostas, D. (2025). _Modern Deep Learning Approaches for Galaxy Morphology Classification_. IAU.
