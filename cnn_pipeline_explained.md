# Pipeline CNN — Paso a Paso

## Paso 0: La imagen original

Tienes una foto JPEG de 424×424 píxeles, 3 canales (RGB).

```
Cada píxel = [R, G, B]    →   valores de 0 a 255
Ejemplo: [142, 98, 45]    →   un tono amarillento (luz de estrellas viejas)

La imagen completa es un tensor de forma:
   (3, 424, 424)
    │   │    │
    │   │    └── ancho: 424 píxeles
    │   └─────── alto: 424 píxeles
    └─────────── canales: R, G, B
```

En esta etapa, la imagen es solo números. No hay concepto de "galaxia" ni "espiral". Es una cuadrícula de intensidades de color.

---

## Paso 1: Preprocesamiento (Transforms)

Antes de entrar a la red, transformamos la imagen.

### 1a. Resize + Crop → 224×224

```
Original: 424×424
    │
    ▼  Resize(256)  →  achica a 256×256 (mantiene proporción)
    │
    ▼  CenterCrop(224)  →  recorta el centro a 224×224
    
¿Por qué 224? Porque ResNet50 fue diseñado para ese tamaño.
Los pesos pretrained esperan imágenes de 224×224.
```

### 1b. Augmentation (solo en entrenamiento)

```
RandomHorizontalFlip(p=0.5)   →  espejo horizontal con 50% probabilidad
RandomRotation(15°)            →  rota ±15 grados
ColorJitter(0.2)               →  varía brillo/contraste/saturación

¿Por qué? Una galaxia espiral rotada 15° sigue siendo espiral.
Le enseñamos a la red que la orientación NO importa para la clase.
Esto se llama DATA AUGMENTATION — multiplicamos artificialmente
la variedad del dataset sin necesitar más fotos.
```

### 1c. Normalización

```
Cada píxel pasa de [0, 255] → [0.0, 1.0] → normalizado

Fórmula:  pixel_norm = (pixel / 255 - mean) / std

Donde mean = [0.485, 0.456, 0.406]   ← promedio de ImageNet
      std  = [0.229, 0.224, 0.225]   ← desviación estándar de ImageNet

¿Por qué estos valores? Porque ResNet50 fue entrenado con estos.
Si no normalizamos igual, los pesos pretrained no funcionan bien.

Resultado: valores centrados en ~0, rango aprox [-2.5, 2.5]
```

### Resultado del Paso 1

```
Tensor de forma: (3, 224, 224)
Tipo: float32
Valores: centrados en 0, normalizados
Listo para entrar a la red.
```

---

## Paso 2: Conv1 — Primera convolución

Aquí empieza la red neuronal.

### ¿Qué es una convolución?

```
Imagina una ventana pequeña (llamada KERNEL o FILTRO) de 7×7 píxeles.

La deslizas sobre toda la imagen, y en cada posición:
1. Multiplicas los valores del kernel × los píxeles debajo
2. Sumas todo → un solo número
3. Ese número va al mapa de salida

  Imagen de entrada          Kernel 7×7           Mapa de salida
  ┌──────────────┐       ┌───────────┐        ┌──────────────┐
  │ · · · · · · ·│       │ 0.1  0.2 .│        │              │
  │ · ■ ■ ■ ■ · ·│   ×   │ 0.3 -0.1 .│   =    │     12.4     │
  │ · ■ ■ ■ ■ · ·│       │-0.2  0.5 .│        │              │
  │ · · · · · · ·│       └───────────┘        └──────────────┘
  └──────────────┘       (aprende los         (valor alto = "aquí
                          valores)             hay algo que el
                                               filtro busca")
```

### Conv1 en ResNet50

```
Entrada:  (3, 224, 224)     ← imagen RGB
Kernel:   7×7, stride=2     ← ventana grande, salta de 2 en 2
Filtros:  64                ← 64 kernels diferentes

Salida:   (64, 112, 112)
           │    │
           │    └── la imagen se redujo a la mitad (stride=2)
           └─────── 64 "mapas de activación" diferentes

Cada uno de los 64 filtros busca algo distinto:
- Filtro #1: bordes verticales
- Filtro #2: bordes horizontales
- Filtro #12: gradientes de brillo
- Filtro #37: manchas circulares
- etc.

Estos filtros NO los programamos. La red los APRENDE durante el
entrenamiento. Los pesos pretrained ya tienen filtros útiles.
```

### BatchNorm + ReLU

```
Después de cada convolución:

BatchNorm:  normaliza las activaciones del batch
            → estabiliza el entrenamiento, permite lr más alto

ReLU:       f(x) = max(0, x)
            → si el valor es negativo, lo pone en 0
            → si es positivo, lo deja igual
            
            ¿Por qué? Introduce NO-LINEALIDAD.
            Sin ReLU, apilar convoluciones sería equivalente
            a una sola convolución (operación lineal).
            ReLU permite a la red aprender funciones complejas.
```

### MaxPool

```
Después de Conv1 + BN + ReLU:

MaxPool 3×3, stride=2:
    Toma ventanas de 3×3 y se queda con el valor MÁXIMO.
    
    ┌─────────┐
    │ 2  4  1 │
    │ 3  8  2 │ → 8 (el máximo)
    │ 1  5  3 │
    └─────────┘

Efecto: (64, 112, 112) → (64, 56, 56)
    - Reduce tamaño a la mitad
    - Mantiene las activaciones más fuertes
    - Hace la red más robusta a pequeños desplazamientos
```

**Estado actual: (64, 56, 56) — 64 mapas de 56×56 con bordes y texturas básicas.**

---

## Paso 3: Bloques Residuales (la clave de ResNet)

### ¿Qué es un bloque residual?

El problema: en redes muy profundas (muchas capas), la señal se "degrada" — la red profunda rinde PEOR que una superficial. Esto no tiene sentido.

La solución de ResNet: **conexiones de atajo (skip connections)**.

```
                    ┌─────────────────────────────┐
                    │         ATAJO (skip)         │
                    │                              │
  Entrada ─────────┼──► Conv1×1 → Conv3×3 → Conv1×1 ──► (+) ──► Salida
     x              │        bloque de 3 capas          │    x + F(x)
                    │                                   │
                    └───────────────────────────────────┘
                    
  Salida = x + F(x)
  
  En vez de aprender la transformación completa F(x),
  la red aprende el RESIDUO: lo que falta sumar a x
  para llegar al resultado correcto.
  
  Si el bloque no aporta nada útil → aprende F(x) ≈ 0
  → la señal original x pasa intacta.
  → la red NUNCA puede ser peor que una más superficial.
```

### Los 4 bloques de ResNet50

```
Block 1 (Layer1) — 3 bloques residuales
    Entrada: (64, 56, 56)
    Salida:  (256, 56, 56)
    
    ¿Qué aprende? TEXTURAS SIMPLES
    - Gradientes de brillo (centro brillante → borde oscuro)
    - Bordes suaves vs bordes afilados
    - Manchas difusas vs puntuales
    
    Para galaxias: distingue "luz suave y homogénea" (elliptical)
    de "luz con estructura" (tiene features/disco)

─────────────────────────────────────────

Block 2 (Layer2) — 4 bloques residuales
    Entrada: (256, 56, 56)
    Salida:  (512, 28, 28)    ← reduce tamaño a la mitad
    
    ¿Qué aprende? PATRONES LOCALES
    - Curvas y líneas
    - Regiones de contraste
    - Patrones repetitivos
    
    Para galaxias: empieza a detectar "hay algo curvo" (brazo?)
    vs "hay algo recto" (barra?) vs "línea oscura" (dust lane)

─────────────────────────────────────────

Block 3 (Layer3) — 6 bloques residuales  ← el más profundo
    Entrada: (512, 28, 28)
    Salida:  (1024, 14, 14)
    
    ¿Qué aprende? PARTES DE OBJETOS
    - Combinaciones de patrones del Block 2
    - Estructuras espaciales complejas
    
    Para galaxias:
    - "Brazo espiral" = curva + nudos brillantes
    - "Barra" = línea recta + gradiente de brillo
    - "Bulbo" = mancha brillante central + caída suave
    - "Disco edge-on" = elipse + dark lane horizontal
    - "Tidal tail" = extensión asimétrica (merger)

─────────────────────────────────────────

Block 4 (Layer4) — 3 bloques residuales
    Entrada: (1024, 14, 14)
    Salida:  (2048, 7, 7)
    
    ¿Qué aprende? CONCEPTOS COMPLETOS
    - Composición global de la imagen
    - Relaciones entre partes
    
    Para galaxias:
    - "Galaxia espiral" = bulbo + brazos curvos + sin barra
    - "Galaxia barrada" = bulbo + barra + brazos desde extremos
    - "Edge-on" = disco + quizá dark lane + quizá bulbo
    - "Elliptical" = solo gradiente suave, nada más
    - "Irregular" = asimetría + múltiples centros + caos
```

**Estado actual: (2048, 7, 7) — 2048 mapas de activación de 7×7. Cada mapa representa un "concepto" aprendido.**

---

## Paso 4: Global Average Pooling

```
Entrada: (2048, 7, 7)

Para cada uno de los 2048 mapas de 7×7:
    → calcula el PROMEDIO de los 49 valores (7×7)
    → un solo número

    Mapa #1:                  Promedio:
    ┌─────────────┐
    │ 2.1  0.3 ...│
    │ 1.5  3.2 ...│  → mean = 1.87
    │ ...      ...│
    └─────────────┘

Salida: vector de 2048 números (2048,)

¿Qué es este vector?
Es el RESUMEN COMPLETO de lo que la red "ve" en la imagen.
Es como un "embedding" — una representación numérica de la galaxia.

- Si mapa #47 = valor alto → "hay brazos espirales"
- Si mapa #183 = valor alto → "hay una barra central"  
- Si mapa #901 = valor alto → "es simétrica/suave"
- etc.

Dos galaxias espirales similares tendrán vectores similares.
Una espiral y una elliptical tendrán vectores muy diferentes.
```

**Estado actual: vector de (2048,) — la "identidad visual" de la galaxia.**

---

## Paso 5: Head de clasificación (lo que reemplazamos)

```
El backbone original de ResNet50 termina con:
    Linear(2048 → 1000)  ← 1000 clases de ImageNet (perro, gato, auto...)

Nosotros lo reemplazamos con:

    Dropout(p=0.5)
        │
        │  ¿Qué hace? Durante entrenamiento, APAGA aleatoriamente
        │  el 50% de las 2048 neuronas (las pone en 0).
        │  
        │  ¿Por qué? REGULARIZACIÓN — fuerza a la red a no depender
        │  de un solo feature. Si el feature #47 ("brazos") se apaga,
        │  la red tiene que usar otros features para decidir.
        │  Esto previene OVERFITTING (memorizar en vez de generalizar).
        │  
        │  En inferencia (cuando el modelo ya está entrenado),
        │  Dropout se desactiva — se usan todas las neuronas.
        │
        ▼
    Linear(2048 → 5)
        │
        │  Multiplica el vector por una matriz de pesos:
        │  
        │  [2048 valores] × [matriz 2048×5] + [5 biases] = [5 valores]
        │  
        │  Cada columna de la matriz aprende a "detectar" una clase.
        │  Columna 0 aprende: "¿qué combinación de features = Elliptical?"
        │  Columna 1 aprende: "¿qué combinación de features = Spiral?"
        │  etc.
        │
        ▼
    5 logits (valores crudos, sin escala definida)
    
    Ejemplo: [3.2, 0.8, -1.1, 0.3, -2.5]
              │    │     │     │     │
              E    S    SB    Se   I/M
```

---

## Paso 6: De logits a predicción

### Durante entrenamiento — CrossEntropyLoss

```
CrossEntropyLoss hace internamente:

1. Softmax: convierte logits → probabilidades (suman 1.0)
   
   softmax([3.2, 0.8, -1.1, 0.3, -2.5])
   = [exp(3.2), exp(0.8), exp(-1.1), exp(0.3), exp(-2.5)] / sum
   = [0.771, 0.070, 0.010, 0.042, 0.003]
   
   Interpretación: 77.1% Elliptical, 7% Spiral, 1% Barred, 4.2% Edge-on, 0.3% Irregular
   
2. Cross-Entropy: compara probabilidades vs etiqueta real
   
   Si la etiqueta real = 0 (Elliptical):
   loss = -log(0.771) = 0.26   ← bajo (la red acertó, poco castigo)
   
   Si la etiqueta real = 1 (Spiral):
   loss = -log(0.070) = 2.66   ← alto (la red falló, mucho castigo)
   
   El loss CASTIGA cuando la probabilidad asignada a la clase
   correcta es baja. La red ajusta sus pesos para minimizar esto.

3. Class weights: multiplican el loss por clase

   Como hay 97k Ellipticals pero solo 8k Irregulars,
   sin weights la red aprende a decir "Elliptical" para todo
   (acertaría ~40% sin esfuerzo).
   
   Con weights: el error en Irregular "pesa" ~12× más que en Elliptical.
   → la red se ve FORZADA a aprender también las clases minoritarias.
```

### Durante inferencia (predicción)

```
Logits → argmax → clase predicha

[3.2, 0.8, -1.1, 0.3, -2.5]
  ↑
  máximo en posición 0

Predicción: clase 0 = Elliptical

Si quieres probabilidades: aplicas softmax manualmente.
```

---

## Paso 7: Backpropagation (actualización de pesos)

```
Esto ocurre DURANTE ENTRENAMIENTO, después de calcular el loss.

1. La red calculó: loss = 2.66 (se equivocó)

2. Backpropagation: calcula ∂loss/∂peso para CADA peso de la red
   (~25 millones de pesos en ResNet50)
   
   Vas "hacia atrás" por la red:
   Loss → Head → Block4 → Block3 → Block2 → Block1 → Conv1
   
   Para cada peso calculamos:
   "¿cuánto contribuyó este peso al error?"
   "¿en qué dirección debería cambiar para reducir el error?"

3. Optimizer (Adam): actualiza cada peso
   
   peso_nuevo = peso - lr × gradiente
   
   lr = learning rate = 1e-4 = 0.0001
   → pasos MUY pequeños (la red ya está pretrained, no queremos
     destruir lo que ya sabe)

4. Repite con el siguiente batch de imágenes.
   Un epoch = pasar por TODAS las imágenes del train set.
   50 epochs = 50 pasadas completas.
```

---

## Resumen visual completo

```
IMAGEN ORIGINAL (424×424×3, JPEG)
    │
    ▼ Resize + Crop + Normalize
TENSOR (3×224×224, float32, normalizado)
    │
    ▼ Conv1 (7×7, 64 filtros, stride 2) + BN + ReLU + MaxPool
    │   "detecto bordes y gradientes básicos"
(64×56×56)
    │
    ▼ Block 1 (3 bloques residuales)
    │   "detecto texturas: luz suave vs estructura"
(256×56×56)
    │
    ▼ Block 2 (4 bloques residuales)
    │   "detecto patrones: curvas, líneas, contraste"
(512×28×28)
    │
    ▼ Block 3 (6 bloques residuales)
    │   "detecto partes: brazos, barras, bulbos, dark lanes"
(1024×14×14)
    │
    ▼ Block 4 (3 bloques residuales)
    │   "detecto conceptos: tipo de galaxia completo"
(2048×7×7)
    │
    ▼ Global Average Pooling
    │   "resumo todo en un vector de identidad"
(2048)
    │
    ▼ Dropout(0.5) → regularización
    ▼ Linear(2048→5) → clasificación
    │
(5 logits)
    │
    ▼ Softmax (en loss) o Argmax (en predicción)
    │
CLASE PREDICHA: Elliptical / Spiral / Barred / Edge-on / Irregular
```
