# 🔭 Pure Vision ConvNeXt — Diagrama de Arquitectura Completa

## Resumen

| Propiedad | Valor |
|-----------|-------|
| **Nombre** | GalaxyMorphConvNeXt |
| **Arquitectura Base** | ConvNeXt-Tiny (Pretrained ImageNet1K_V1) |
| **Tipo de Red** | CNN Pura Moderna (Inspirada en Transformers) |
| **Parámetros totales** | ~28 Millones |
| **Input** | (B, 3, 224, 224) |
| **Output** | (B, 5) logits |
| **Clases** | Elliptical, Spiral, Barred_Spiral, Edge_on, Irregular_Merger |

---


## 1. Preprocesamiento (Data Augmentation "Visión Pura")

Sin MixUp ni difuminados destructivos, permitimos que la CNN absorba detalles estructurales limpios.

```
Imagen Cruda: 424×424 RGB
        │
        ▼
┌─────────────────────────────────────────────┐
│  TRAIN AUGMENTATION (Astronomía)            │
│                                             │
│  1. Resize(256)                             │
│     → Escalar suavemente la galaxia original│
│                                             │
│  2. RandomCrop(224)                         │
│     → Evitamos el zoom destructivo. Recorte │
│       centrado conservador.                 │
│                                             │
│  3. RandomHorizontalFlip(p=0.5)             │
│  4. RandomVerticalFlip(p=0.5)               │
│  5. RandomRotation(degrees=180)             │
│     → Simetría espacial pura                │
│                                             │
│  6. ColorJitter(b=0.2, c=0.2)               │
│     → SIN hue/saturation. Preservamos el    │
│       color de la temperatura estelar.      │
│                                             │
│  7. Normalización (ImageNet)                │
└─────────────────────────────────────────────┘
        │
        ▼
  Tensor: (B, 3, 224, 224)
```

---

## 2. El "Patchify Stem" (Extracción Inicial)

En lugar de múltiples convoluciones complejas, ConvNeXt corta la imagen en parches no superpuestos, igual que un Vision Transformer.

```
Input: (B, 3, 224, 224)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  STEM (Patchify)                                     │
│  Conv2d(3, 96, kernel_size=4, stride=4)              │
│  + LayerNorm                                         │
│  → (B, 96, 56, 56)                                  │
└──────────────────────────────────────────────────────┘
```

---

## 3. Las Etapas ConvNeXt (Stages)

La red consta de 4 etapas que reducen la resolución espacial progresivamente mientras aumentan el número de canales.

```
Input: (B, 96, 56, 56)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  STAGE 1 (3 Bloques)                                 │
│  Canales: 96                                         │
│  → (B, 96, 56, 56)                                  │
├──────────────────────────────────────────────────────┤
│  DOWNSAMPLE                                          │
│  LayerNorm + Conv2d(96, 192, kernel_size=2, stride=2)│
│  → (B, 192, 28, 28)                                 │
├──────────────────────────────────────────────────────┤
│  STAGE 2 (3 Bloques)                                 │
│  Canales: 192                                        │
│  → (B, 192, 28, 28)                                 │
├──────────────────────────────────────────────────────┤
│  DOWNSAMPLE                                          │
│  LayerNorm + Conv2d(192, 384, kernel_size=2, stride=2)
│  → (B, 384, 14, 14)                                 │
├──────────────────────────────────────────────────────┤
│  STAGE 3 (9 Bloques)                                 │
│  Canales: 384                                        │
│  → (B, 384, 14, 14)                                 │
├──────────────────────────────────────────────────────┤
│  DOWNSAMPLE                                          │
│  LayerNorm + Conv2d(384, 768, kernel_size=2, stride=2)
│  → (B, 768, 7, 7)                                   │
├──────────────────────────────────────────────────────┤
│  STAGE 4 (3 Bloques)                                 │
│  Canales: 768                                        │
│  → (B, 768, 7, 7)                                   │
└──────────────────────────────────────────────────────┘
```

> [!NOTE]
> **¿Qué hay dentro de CADA Bloque ConvNeXt?**
> Es un "Inverted Bottleneck" (cuello de botella invertido) inspirado en los Transformers:
> 1. **Depthwise Conv 7x7:** Convolución masiva de 7x7 píxeles procesando un canal a la vez. (Emula Atención Global).
> 2. **LayerNorm** (Normalización por capa, no por batch).
> 3. **Pointwise Conv 1x1 (Expand):** Multiplica los canales x4 (ej. 96 → 384).
> 4. **GELU:** Función de activación matemáticamente suave.
> 5. **Pointwise Conv 1x1 (Project):** Comprime de vuelta a los canales originales (ej. 384 → 96).
> 6. **DropPath + Residual Connection.**

---

## 4. Classification Head

```
Features de Salida: (B, 768, 7, 7)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  GLOBAL AVERAGE POOLING                              │
│  AvgPool2d(7x7) + Flatten                            │
│  → (B, 768)                                         │
│  Comprime el espacio 2D en un vector por galaxia     │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  LayerNorm(768)                                      │
│  Estabiliza el vector final                          │
│  → (B, 768)                                         │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Dropout(0.5)                                        │
│  Apaga el 50% de las neuronas para evitar Overfitting│
│  → (B, 768)                                         │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Linear(768, 5)                                      │
│  Capa de decisión para las 5 clases finales          │
│  → (B, 5) logits                                    │
└──────────────────────────────────────────────────────┘
  │
  ▼
CrossEntropyLoss(label_smoothing=0.1)
```

---

## 5. Estrategia de Entrenamiento ("El Acelerado")

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING STRATEGY                                              │
│  ─────────────────                                              │
│  Fase 1 (epochs 1-5):  Backbone CONGELADO                       │
│     • Solo entrena: Cabecera (Classification Head)              │
│     • LR: 3e-4, AdamW, weight_decay=1e-3                       │
│     • Warmup: 3 epochs lineales                                 │
│                                                                 │
│  Fase 2 (epochs 6-40): Backbone PARCIALMENTE DESCONGELADO       │
│     • Descongelar: Últimos 2 bloques masivos de ConvNeXt        │
│     • LR diferencial: backbone=1e-5, head=3e-4                  │
│     • Cosine Decay                                              │
│     • Early stopping: patience=7                                │
│                                                                 │
│  MixUp: APAGADO (Queremos imágenes puras)                       │
│  Balanceo: WeightedRandomSampler (oversample de clases raras)   │
│  Batch: 128 | Workers: 2 | DataParallel: T4 x2                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparación contra el Híbrido Anterior

| Aspecto | Híbrido (Ef-B0 + Transformer) | ConvNeXt-Tiny (Nuevo) |
|---------|-----------------------------|-----------------------|
| **Backbone** | EfficientNet-B0 | ConvNeXt-Tiny |
| **Atención Espacial** | Transformer (Self-Attention) | Convoluciones Depthwise Gigantes (7x7) |
| **Params totales** | ~8.2 Millones | ~28 Millones |
| **Función Activación** | SiLU / GELU | GELU |
| **Normalización** | BatchNorm + LayerNorm | 100% LayerNorm |
| **Extracción Inicial** | Conv 3x3 progresiva | Patchify Stem (4x4, stride 4) |
| **MixUp** | Activado (alpha=0.2) | **Apagado (Fomenta aprendizaje veloz)** |
| **Learning Rate** | 1e-4 | **3e-4 (Soporta mayor velocidad)** |
| **Resolución** | 224x224 | 224x224 |
