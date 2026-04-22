# GalaxyMorph ML

Clasificación automática de morfología galáctica usando una arquitectura híbrida **CNN + Transformer Encoder** basada en el esquema de Hubble-de Vaucouleurs.

---

## 🎯 Descripción General

**GalaxyMorph** es un proyecto de tesis enfocado en utilizar el aprendizaje profundo (Deep Learning) para clasificar galaxias a partir de imágenes astronómicas reales del [Galaxy Zoo 2](https://data.galaxyzoo.org/).

El modelo emula el consenso de cientos de miles de voluntarios, analizando patrones visuales complejos como brazos espirales, barras centrales y bulbos difusos, de manera completamente automática y escalable.

## 🎯 Objetivos del Proyecto

1. **Clasificación automatizada:** Construir un modelo robusto que prediga la morfología de una galaxia con alta precisión sin intervención humana.
2. **Definición taxonómica estricta:** Utilizar los datos astronómicos de Hart et al. (2016) para asegurar que la "verdad base" (Ground Truth) sea científicamente precisa.
3. **Escalabilidad en Producción:** Sentar las bases en PyTorch (`.pt`) para posteriormente montar un clasificador distribuido capaz de procesar miles de imágenes por segundo usando Apache Spark y Kafka en una arquitectura de microservicios.

---

## 🌌 ¿Qué clasificamos? (Las 5 Clases)

Clasificamos galaxias en las **5 categorías principales** de la secuencia de Hubble presentes en el árbol de decisiones de Galaxy Zoo 2. Utilizamos la cadena de consenso `gz2_class` para definir matemáticamente las etiquetas:

| # | Clase | Descripción | Consenso `gz2_class` |
|---|-------|-------------|----------------------|
| 0 | ⭕ **Elliptical** | Galaxias de población estelar vieja, forma suave sin disco ni polvo | Inicia con `E` |
| 1 | 🌀 **Spiral** | Galaxias con disco brillante de frente y brazos espirales visibles | Inicia con `S` (sin barra) |
| 2 | ➖ **Barred_Spiral** | Espirales con una estructura elongada (barra) atravesando el núcleo | Inicia con `SB` |
| 3 | 🛸 **Edge_on** | Galaxias de disco vistas de canto/perfil, con línea de polvo visible | Inicia con `Se` |
| 4 | 💥 **Irregular_Merger** | Galaxias asimétricas o caóticas por colisiones o perturbaciones | Marcado con `(i)`, `(m)` o `(d)` |

---

## 📊 Dataset y Origen de Datos

El entrenamiento se sustenta en los datos del segundo relanzamiento del proyecto **Galaxy Zoo 2**. Toda información de Kaggle o descargas no oficiales se descartó para mantener rigurosidad científica.

### Archivos clave

- **`gz2_hart16.csv`**: Catálogo de Hart et al. (2016) con las respuestas al árbol de decisión de GZ2 y la columna `gz2_class`.
- **`gz2_filename_mapping.csv`**: Tabla relacional que une el identificador astronómico (`objid`) con el nombre del archivo `.jpg` en disco.
- **Imágenes (`images_gz2/images/`)**: ~240,000 recortes a color en formato JPG de $424 \times 424$ obtenidos del telescopio Sloan Digital Sky Survey (SDSS).

### Estadísticas del Dataset

| Split | Muestras |
|-------|----------|
| Train | ~167,321 (70%) |
| Val   | ~35,854 (15%) |
| Test  | ~35,854 (15%) |
| **Total** | **~239,029** |

---

## 🔄 Evolución del Modelo: De ResNet50 a Arquitectura Híbrida

### v1 — ResNet50 (descartado)

La primera versión del modelo usaba **ResNet50** pretrained en ImageNet como backbone único, con una capa FC de salida para 5 clases.

**Problema detectado:** Overfitting severo a partir de la epoch 25:

```
Epoch 25: Train Loss: 0.48  Val Loss: 0.72  ← gap se abre
Epoch 30: Train Loss: 0.31  Val Loss: 0.95  ← memorización
Epoch 35: Train Loss: 0.21  Val Loss: 1.24  ← colapso de generalización
```

| Causa del overfitting | Detalle |
|----------------------|----------|
| **Demasiados parámetros** | ResNet50 tiene 25.6M params para ~167K imágenes de entrenamiento |
| **Arquitectura monolítica** | Una CNN sola no modela relaciones espaciales globales |
| **Sin regularización suficiente** | No había label smoothing, mixup ni gradient clipping |

Puedes ver el experimento original en: [Kaggle v1 — ResNet50](https://www.kaggle.com/code/jeancdevx/galaxymorph-cnn-for-classifying-galaxy-morphology)

### v2 — GalaxyMorphHybrid (descartado)

La migración a una arquitectura híbrida **EfficientNet-B0 + Transformer Encoder** solucionó el overfitting, pero introdujo **Underfitting** severo:

- **Problema:** El Transformer junto con el `MixUp` regularizó en exceso el modelo. La red no podía asimilar las características geométricas borrosas (brazos espirales débiles) de las galaxias SDSS en el tiempo dado (40 épocas).
- **Resultado:** Val F1 estancado en **0.5606**. Gran confusión entre galaxias "face-on" (Espirales vs Elípticas vs Barradas).
- **Lo bueno:** El `WeightedRandomSampler` demostró ser un éxito total descubriendo las galaxias minoritarias (Mergers y Edge-on).

Puedes ver el experimento original en: [Kaggle v2 — EfficientNet-B0 + Transformer](https://www.kaggle.com/code/jeancdevx/hybrid-cnn-transformer-for-galaxy-morphology)

### v3 — Pure Vision ConvNeXt (actual)

Para romper el bloqueo del modelo híbrido, saltamos a la cúspide evolutiva de las CNNs: **ConvNeXt-Tiny**.

| Problema (v2) | Solución (v3) |
|---------------|---------------|
| Underfitting por Transformer | **Arquitectura ConvNeXt** con Inductive Bias natural para detectar formas con pocos datos. |
| CNN clásica no ve el contexto global | ConvNeXt usa **kernels gigantes (7x7)** que emulan la atención global de los Transformers. |
| Regularización destructiva (MixUp) | **MixUp APAGADO.** El modelo aprende de imágenes puras y limpias. |
| Convergencia lenta (LR=1e-4) | **Acelerador a fondo (LR=3e-4).** ConvNeXt es sumamente robusto ante LRs agresivos. |

Puedes ver el experimento actual en: [Kaggle v3 — Pure Vision ConvNeXt](https://www.kaggle.com/code/jeancdevx/pure-vision-convnext-for-galaxy-morphology)

---

## ⚙️ Arquitectura del Modelo: ConvNeXt-Tiny Pura

La arquitectura actual es un modelo puramente convolucional de última generación (**ConvNeXt-Tiny**). Este modelo combina la velocidad y el sesgo inductivo de las CNNs con las macro-arquitecturas modernas de los Vision Transformers.

### Componentes Clave

| Componente | Rol |
|------------|-----|
| **Patchify Stem** | En lugar de convoluciones iniciales complejas, usa una Conv de 4x4 (Stride 4) para cortar la imagen en parches, procesando la imagen mucho más rápido sin perder información estructural. |
| **Depthwise Convolutions 7x7** | Usa kernels masivos (7x7) que permiten observar enormes porciones de la galaxia a la vez, emulando la atención global. |
| **Inverted Bottleneck** | Expande canales x4 y los vuelve a comprimir, exactamente igual a las capas Feed-Forward de un Transformer. |
| **Classification Head** | `LayerNorm` → `Dropout(0.5)` → `Linear(768, 5)`. |

**Parámetros Totales:** ~28 Millones.

---

## 🏋️ Estrategia de Entrenamiento (Flujo "Acelerado")

### Two-Phase Fine-tuning Simplificado

El entrenamiento se divide en 2 fases veloces para adaptar los pesos pretrained de ImageNet a la astronomía:

#### Fase 1 — Backbone Congelado (Epochs 1-5)

```
Backbone ConvNeXt: 🧊 FROZEN (no se actualiza)
Classification Head: 🔥 ENTRENANDO
LR: warmup lineal → 3e-4
```

La cabecera aprende a interpretar rápidamente las representaciones puras extraídas por ConvNeXt.

#### Fase 2 — Descongelado Parcial (Epochs 6-40)

```
Backbone (Etapas 1 y 2): 🧊 FROZEN
Backbone (Etapas 3 y 4): 🔥 LR diferencial bajo = 1e-5
Classification Head:     🔥 LR normal = 3e-4
MixUp:                   APAGADO (imágenes puras)
```

### Regularización Segura (Visión Pura)

A diferencia del modelo anterior, hemos quitado la regularización extrema destructiva (MixUp, Blur, Hue) para permitir que la CNN absorba detalles geométricos limpios.

| Técnica | Valor | Propósito |
|---------|-------|-----------|
| **Label Smoothing** | 0.1 | Evita confianza excesiva en etiquetas |
| **Weight Decay** | 1e-3 | L2 regularization en AdamW |
| **Dropout** | 0.5 (head) | Apagado aleatorio de neuronas |
| **Augmentation Espacial** | Resize(256) + Crop(224) | Invarianza traslacional conservadora |
| **Rotación Completa** | Flips + Rot(180) | Emula la falta de orientación en el universo |

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Batch Size | 128 |
| Epochs máximos | 40 |
| Learning Rate (head) | 3e-4 |
| Learning Rate (backbone P2) | 1e-5 |
| Warmup Epochs | 3 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Balanceador | WeightedRandomSampler |

---

## 🚀 Entornos de Ejecución

### 1. Desarrollo Local (Docker — Experimentación sin GPU)

```bash
docker compose run --rm app bash
```

Útil para construir el dataset, depurar código y graficar imágenes. No entrena (sin GPU local).

### 2. Entrenamiento en Kaggle (2× GPU T4)

El entrenamiento real se realiza en Kaggle. Hay tres notebooks en la historia del proyecto:

| Notebook | Modelo | Estado |
|----------|--------|--------|
| [v1 — ResNet50](https://www.kaggle.com/code/jeancdevx/galaxymorph-cnn-for-classifying-galaxy-morphology) | ResNet50 tradicional | ⚠️ Descartado (overfit) |
| [v2 — Hybrid CNN+Transformer](https://www.kaggle.com/code/jeancdevx/hybrid-cnn-transformer-for-galaxy-morphology) | EfficientNet-B0 + Transformer | ⚠️ Descartado (underfit) |
| [v3 — Pure Vision ConvNeXt](https://www.kaggle.com/code/jeancdevx/pure-vision-convnext-galaxy-morphology) | ConvNeXt-Tiny | ✅ Actual |

La notebook activa es self-contained — no depende de archivos `src/` externos.

```
Configuración Kaggle:
  Accelerator: GPU T4 x2
  Runtime:     Save & Run All (Commit) — hasta 12h
  DataParallel: activado automáticamente si hay 2 GPUs
```

---

## 📁 Estructura del Proyecto

```
galaxy-morph-ml/
├── notebooks/
│   ├── galaxymorph-cnn-for-classifying-galaxy-morphology.ipynb  ← v1 ResNet50 (legacy)
│   ├── hybrid-cnn-transformer-for-galaxy-morphology.ipynb       ← v2 Hybrid (legacy)
│   └── pure-vision-convnext-galaxy-morphology.ipynb             ← v3 ConvNeXt (actual)
├── src/
│   ├── data/
│   │   └── build_dataset.py       ← Construcción del manifest CSV
│   └── models/                    ← Definiciones de modelos (legacy)
├── configs/                       ← Configuraciones YAML
├── logs/
│   └── training_history.json      ← Historial de métricas por época
├── models/                        ← Checkpoints guardados (.pt)
├── tests/                         ← Tests unitarios
├── galaxymorph_knowledge_base.md  ← Documentación técnica extendida
├── cnn-architecture.md            ← Diagrama detallado de arquitectura
└── requirements.txt
```

---

## 📦 Tecnologías y Librerías

| Categoría | Librería |
|-----------|----------|
| **Core DL** | `PyTorch`, `torchvision` |
| **Backbone** | EfficientNet-B0 (pretrained ImageNet) |
| **Data** | `Pandas`, `NumPy` |
| **Métricas** | `Scikit-learn` (F1, Accuracy, Confusion Matrix) |
| **Visualización** | `Matplotlib`, `Seaborn` |
| **Deployment** | `Docker` |

---

## 📈 Resultados

> ⚠️ Entrenando la versión v3 (ConvNeXt-Tiny). Los resultados se actualizarán al finalizar.

| Modelo | Val F1 (macro) | Val Accuracy | Params | Estado |
|--------|---------------|--------------|--------|--------|
| v1: ResNet50 | 0.6939 | ~0.71 | 25.6M | Overfit severo |
| v2: GalaxyMorphHybrid | 0.5606 | 0.5961 | 8.2M | Underfit (MixUp excesivo) |
| **v3: ConvNeXt-Tiny** | _en curso_ | _en curso_ | ~28.0M | ✅ En entrenamiento |

---

_GalaxyMorph — 2026. Explorando el universo con Inteligencia Artificial._
