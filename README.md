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

### v2 — GalaxyMorphHybrid (actual)

La migración a una arquitectura híbrida **EfficientNet-B0 + Transformer Encoder** resuelve los problemas del ResNet50:

| Problema (v1) | Solución (v2) |
|---------------|---------------|
| 25.6M params → overfit | **10.9M params** (~57% menos, menos riesgo de memorización) |
| CNN no ve el contexto global | **Transformer Encoder** relaciona las 49 regiones espaciales de la galaxia entre sí |
| Sin regularización | Label smoothing + MixUp + Dropout + Weight Decay + Early Stopping |
| Todos los pesos se entrenan desde epoch 1 | **Two-phase fine-tuning:** backbone congelado primero, descongelado gradual después |
| Augmentation básica | Augmentation agresiva aprovechando simetría rotacional galáctica |

**¿Por qué EfficientNet-B0 y no ResNet50 o EfficientNet-B1?**
- **vs ResNet50:** EfficientNet-B0 logra mayor precisión con 5.3M params (vs 23.5M del backbone ResNet50) usando compound scaling
- **vs EfficientNet-B1:** B1 tiene ~7.8M params en el backbone — más VRAM, más lento, y el Transformer ya compensa la capacidad extra

Puedes ver el experimento actual en: [Kaggle v2 — Hybrid CNN+Transformer](https://www.kaggle.com/code/jeancdevx/hybrid-cnn-transformer-for-galaxy-morphology)

---

## ⚙️ Arquitectura del Modelo: GalaxyMorphHybrid

La arquitectura actual es un **modelo híbrido CNN + Transformer** que combina la eficiencia de EfficientNet-B0 para extraer features locales con la capacidad del Transformer Encoder para capturar relaciones espaciales globales entre regiones de la galaxia.

### ¿Por qué híbrida?

| Componente | Rol | Por qué |
|------------|-----|---------|
| **EfficientNet-B0** (CNN) | Extractor de features locales | Detecta bordes, texturas, patrones de brazos espirales, barras |
| **Transformer Encoder** | Relaciones espaciales globales | Relaciona tokens de diferentes regiones: "¿el brazo izquierdo se corresponde con el derecho?" |
| **Combinación** | Lo mejor de ambos | CNN sola no ve el conjunto; Transformer solo sin CNN pierde precisión local |

**Ventaja clave sobre ResNet50 anterior:** ~11M parámetros vs ~25.6M. Menos parameters = menos riesgo de memorización (overfitting).

### Flujo de datos

```
Imagen (3, 224, 224)
        │
        ▼
┌─────────────────────────────────┐
│   EfficientNet-B0 (backbone)   │  52 capas, pretrained ImageNet
│   features: (B, 1280, 7, 7)   │  detecta features locales
└─────────────────────────────────┘
        │  flatten + transpose
        ▼
   Tokens: (B, 49, 1280)        ← 49 regiones de 7×7 de la imagen
        │  Linear projection
        ▼
   Tokens: (B, 49, 512)  + Positional Encoding (learnable)
        │
        ▼
┌─────────────────────────────────┐
│   Transformer Encoder          │  2 capas, 8 heads, dim=512
│   Self-Attention global        │  relaciona las 49 regiones entre sí
└─────────────────────────────────┘
        │  Global Average Pool (mean over 49 tokens)
        ▼
   Vector: (B, 512)
        │
        ▼
┌─────────────────────────────────┐
│   Classification Head          │  LayerNorm → Dropout(0.5) → Linear(512→5)
└─────────────────────────────────┘
        │
        ▼
   Logits: (B, 5)  →  softmax  →  clase predicha
```

### Parámetros

| Componente | Params |
|------------|--------|
| EfficientNet-B0 (backbone) | ~5.3M |
| Projection + Positional Encoding | ~0.7M |
| Transformer Encoder (2 capas) | ~4.7M |
| Classification Head | ~0.3M |
| **Total** | **~10.9M** |

---

## 🏋️ Estrategia de Entrenamiento

### Two-Phase Fine-tuning (Gradual Unfreezing)

El entrenamiento se divide en 2 fases para evitar que los gradientes ruidosos del inicio corrompan los pesos pretrained de ImageNet:

#### Fase 1 — Backbone Congelado (Epochs 1-10)

```
Backbone EfficientNet: 🧊 FROZEN (no se actualiza)
Transformer + Head:    🔥 ENTRENANDO (~6.9M params)
LR: warmup lineal 3e-5 → 1e-4 → cosine decay
```

El Transformer aprende a interpretar las features del backbone sin perturbarlo.

#### Fase 2 — Descongelado Parcial (Epochs 11-40)

```
Backbone (bloques 0-5): 🧊 FROZEN
Backbone (bloques 6-8): 🔥 LR bajo = 1e-5  (~3.1M extra params)
Transformer + Head:     🔥 LR normal = 1e-4
MixUp (alpha=0.2):      activado
```

Solo los últimos 3 bloques del backbone (los más específicos del dominio) se adaptan a galaxias.

### Regularización Anti-Overfitting

| Técnica | Valor | Propósito |
|---------|-------|-----------|
| **Label Smoothing** | 0.1 | Evita confianza excesiva en etiquetas |
| **Weight Decay** | 1e-3 | L2 regularization en AdamW |
| **Dropout** | 0.5 (head), 0.3 (transformer) | Apagado aleatorio de neuronas |
| **MixUp** | α=0.2 (solo Fase 2) | Mezcla imágenes para generalizar |
| **Grad Clipping** | max_norm=1.0 | Evita gradient explosion |
| **Early Stopping** | patience=7 | Para cuando Val F1 no mejora |

### Augmentation Agresiva (aprovecha simetría galáctica)

Las galaxias son simétricas rotacionalmente — una espiral girada 90° sigue siendo una espiral:

```python
RandomResizedCrop(224, scale=(0.5, 1.0))  # zoom variable
RandomHorizontalFlip(p=0.5)               # espejo horizontal
RandomVerticalFlip(p=0.5)                 # espejo vertical
RandomRotation(degrees=180)               # rotación completa
ColorJitter(brightness, contrast, ...)    # variación de brillo/color
GaussianBlur(kernel_size=3)              # simula diferentes telescopios
RandomErasing(p=0.25)                    # oculta regiones aleatorias
```

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Batch Size | 128 |
| Epochs máximos | 40 |
| Learning Rate (head) | 1e-4 |
| Learning Rate (backbone P2) | 1e-5 |
| Warmup Epochs | 3 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss (weighted + label smoothing) |
| Scheduler | Linear warmup → Cosine decay |

---

## 🚀 Entornos de Ejecución

### 1. Desarrollo Local (Docker — Experimentación sin GPU)

```bash
docker compose run --rm app bash
```

Útil para construir el dataset, depurar código y graficar imágenes. No entrena (sin GPU local).

### 2. Entrenamiento en Kaggle (2× GPU T4)

El entrenamiento real se realiza en Kaggle. Hay dos notebooks en la historia del proyecto:

| Notebook | Modelo | Estado |
|----------|--------|--------|
| [v1 — ResNet50](https://www.kaggle.com/code/jeancdevx/galaxymorph-cnn-for-classifying-galaxy-morphology) | ResNet50 tradicional | ⚠️ Descartado (overfit epoch 25) |
| [v2 — Hybrid CNN+Transformer](https://www.kaggle.com/code/jeancdevx/hybrid-cnn-transformer-for-galaxy-morphology) | EfficientNet-B0 + Transformer Encoder | ✅ Actual |

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
│   └── hybrid-cnn-transformer-for-galaxy-morphology.ipynb       ← v2 Hybrid (actual)
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

> ⚠️ Entrenamiento en curso con la nueva arquitectura híbrida. Los resultados se actualizarán al finalizar.

| Modelo | Val F1 (macro) | Val Accuracy | Params | Estado |
|--------|---------------|--------------|--------|--------|
| ResNet50 (baseline) | 0.6939 | ~0.71 | 25.6M | Overfit severo (epoch 25+) |
| **GalaxyMorphHybrid** | _en curso_ | _en curso_ | 10.9M | ✅ En entrenamiento |

---

_GalaxyMorph — 2026. Explorando el universo con Inteligencia Artificial._
