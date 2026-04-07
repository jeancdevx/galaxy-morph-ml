# GalaxyMorph ML

Clasificación automática de morfología galáctica utilizando Redes Neuronales Convolucionales (CNN) basada en el esquema de Hubble-de Vaucouleurs.

---

## 🎯 Descripción General

**GalaxyMorph** es un proyecto de tesis enfocado en utilizar el aprendizaje profundo (Deep Learning) y la percepción computacional para clasificar galaxias a partir de imágenes astronómicas reales.

El modelo emula el consenso de cientos de miles de voluntarios de [Galaxy Zoo 2](https://data.galaxyzoo.org/), analizando patrones visuales complejos como brazos espirales, barras centrales y bulbos difusos, de manera completamente automática y escalable.

## 🎯 Objetivos del Proyecto

1. **Clasificación automatizada:** Construir un modelo robusto que prediga la morfología de una galaxia con alta precisión sin intervención humana.
2. **Definición taxonómica estricta:** Utilizar los datos astronómicos de Hart et al. (2016) para asegurar que la "verdad base" (Ground Truth) sea científicamente precisa.
3. **Escalabilidad en Producción:** Sentar las bases en PyTorch (`.pt`) para posteriormente montar un clasificador distribuido capaz de procesar miles de imágenes por segundo usando Apache Spark y Kafka en una arquitectura de microservicios.

---

## 🌌 ¿Qué clasificamos? (Las 5 Clases)

A diferencia de proyectos de prueba que usan solo 3 clases u omiten los detalles técnicos, esta arquitectura clasifica las galaxias en las **5 categorías principales** de la secuencia de Hubble presentes en el árbol de decisiones de Galaxy Zoo 2.

Utilizamos la cadena de consenso `gz2_class` para definir matemáticamente las etiquetas:

1. ⭕ **Elliptical (Smooth):** Galaxias de población estelar vieja, de forma suave, sin disco visible ni polvo. _(Consenso inicia con `E`)_.
2. 🌀 **Spiral:** Galaxias con un disco brillante visto de frente y brazos espirales claramente visibles. _(Consenso inicia con `S`, sin barra)_.
3. ➖ **Barred_Spiral:** Galaxias espirales que presentan una estructura elongada (barra) de estrellas atravesando su núcleo. _(Consenso inicia con `SB`)_.
4. 🛸 **Edge_on:** Galaxias de disco que estamos viendo de canto/perfil desde la Tierra. Sus brazos no son visibles, pero el polvo intergaláctico crea una línea oscura que parte el disco. _(Consenso inicia con `Se`)_.
5. 💥 **Irregular_Merger:** Galaxias con formas asimétricas o caóticas, comúnmente causadas por el choque gravitacional de dos galaxias (Mergers) o perturbaciones severas. _(Consenso marcado con `(i)`, `(m)`, o `(d)`)_.

---

## 📊 Dataset y Origen de Datos

El entrenamiento se sustenta en los datos del segundo relanzamiento del proyecto **Galaxy Zoo 2**. Toda la información de Kaggle o descargas no oficiales se descartó para mantener la rigurosidad científica.

### Archivos clave:

- **`gz2_hart16.csv`**: El catálogo de Hart et al. (2016). Aquí residen las respuestas al árbol de decisión de GZ2 y la importantísima columna `gz2_class`.
- **`gz2_filename_mapping.csv`**: Tabla relacional que une el identificador astronómico (`objid`) con el nombre del archivo `.jpg` en disco.
- **Imágenes (`images_gz2/images/`)**: ~240,000 recortes a color en formato JPG de $424 \times 424$ obtenidos originalmente del telescopio Sloan Digital Sky Survey (SDSS). Todo nuestro repositorio se conecta con Google Drive / R2 para no albergarlas en git.

---

## ⚙️ Flujo Completo del Aprendizaje (Arquitectura)

Desde que la foto del satélite entra al disco duro hasta que sale una predicción, ocurre el siguiente proceso (End-to-End Learning):

### 1. Construcción del Manifest (`build_dataset.py`)

El script cruza las tablas SQL usando Pandas, limpia errores de datos y artefactos (galaxias invisibles o corruptas). Verifica que los archivos existan en el sistema y estratifica equitativamente los datos en:

- **Train (80%)**: Donde la red ajusta sus pesos.
- **Validation (10%)**: Donde verificamos que no esté memorizando (Overfitting).
- **Test (10%)**: Muestra sagrada que usamos al final del proyecto para reportar la precisión final.

### 2. Preprocesamiento In-Memory (DataLoaders)

La red nunca ve el JPG limpio. En las transformaciones (usando `torchvision`) suceden 4 cosas:

- **Resize y CenterCrop:** La imagen de $424 \times 424$ se escala un poco y se recorta un bloque perfecto de **$224 \times 224$** enfocándose justo en la galaxia (eliminando ruido negro de los bordes).
- **Data Augmentation:** Aleatoriamente la foto se gira $\pm 15^\circ$, se invierte como en un espejo o recibe un ajuste de brillo. Esto fuerza al modelo a "aprender astronomía", dictando que una barra espiral de cabeza sigue siendo una barra espiral.
- **ToTensor & Normalize:** Se convierte a decimales `float32` y se aplican promedios estadísticos estrictos de ImageNet para balancear la luz de la imagen con lo que la arquitectura ResNet espera ver.

### 3. La Red Neuronal: ResNet50

Se eligió la profunda **Residual Network de 50 capas (ResNet50)**.

- La imagen entra como un tensor de `[3, 224, 224]` (3 canales RGB).
- Va cruzando decenas de filtros convolucionales que extraen desde simples bordes (Capas iniciales) hasta detectar las barras brillantes (Capas medias) y los difusos brazos espirales azules (Capas profundas).
- A diferencia de redes más viejas (como VGG o AlexNet), la estructura "Residual" (conexiones de salto) permite que una señal matemática atraviese 50 capas sin degradarse.

### 4. Head de Salida

Se corta la capa final original de ResNet50 que reconocía perros/gatos, y ponemos una capa _Fully Connected (Linear)_ de **frente a nuestras 5 clases**. Se aplica una función `CrossEntropyLoss` ponderada temporalmente durante el entrenamiento para compensar que el universo tiene más galaxias espirales que irregulares.

Finalmente obtenemos una distribución softmax (`[1%, 2%, 91%, 3%, 3%]`). Para inferencia, el número más grande es la ganadora absoluta.

---

## 🚀 Entornos de Ejecución

El código está estructurado para ejecutarse modularmente en dos entornos diferentes:

### 1. Desarrollo Local (Docker Cpu/Experimentación)

En la PC del desarrollador usando `docker-compose up --build`. No entrena (muy lento sin GPU local), pero sirve para correr debuggers, compilar el dataset, o graficar imágenes.

- _Comando de arranque:_ `docker compose run --rm app bash`

### 2. Entrenamiento en la Nube (Kaggle - 2 GPU T4)

Para el entrenamiento real, el proyecto usa un modelo híbrido. En caso cuentes con una grafica local potente, puedes entrenar ahí. Pero para la mayoría, el entrenamiento se realiza en Kaggle (con 2 GPU T4) haciendo uso de la notebook `galaxymorph-cnn-for-classifying-galaxy-morphology.ipynb`. El código de entrenamiento es idéntico al local, pero con rutas adaptadas a la estructura de Kaggle.

Para este caso ya dejamos una notebook con un modelo entrenado y guardado en Kaggle, puedes hacer cualquier experimento o análisis adicional partiendo de ese checkpoint.

Puedes revisar la notebook [GalaxyMorph](https://www.kaggle.com/code/jeancdevx/galaxymorph-cnn-for-classifying-galaxy-morphology) para ver el proceso de entrenamiento, validación y evaluación del modelo.

## 📦 Tecnologías y Librerías Base

- **Core DL**: `PyTorch`, `torchvision`
- **Data Handling**: `Pandas`, `Numpy`
- **Metrics**: `Scikit-learn`
- **Deployment**: `Docker`

---

_Galaxymorph - 2026. Explorando el universo con Inteligencia Artificial._
