Estoy desarrollando un proyecto llamado GalaxyMorph, cuyo objetivo principal es construir una plataforma capaz de clasificar automáticamente la morfología de galaxias a partir de imágenes astronómicas.

Objetivo del proyecto

Queremos que un usuario pueda subir una o varias imágenes de galaxias desde una aplicación web, y que el sistema procese cada imagen para predecir su clase morfológica. En esta primera versión, las clases que vamos a usar son:

Espiral

Elíptica

Lenticular

Irregular

El objetivo no es hacer segmentación ni detección de objetos, sino clasificación de imágenes.

Dataset que vamos a usar

Trabajaremos con el dataset de Galaxy Zoo 2. Para el entrenamiento necesitamos combinar tres fuentes:

Images dataset: contiene las imágenes de galaxias.

gz2_filename_mapping.csv: sirve para relacionar cada archivo de imagen con su identificador astronómico.

gz2_hart16.csv.gz o Table 1 - Normal-depth sample with new debiasing method: contiene la información morfológica que usaremos para construir las etiquetas de entrenamiento.

La idea es unir estos archivos para formar un dataset final con:

ruta de la imagen

identificador del objeto

clase final asignada

y, si hace falta, metadatos útiles adicionales

Enfoque de percepción computacional

Vamos a resolver el problema con una CNN preentrenada para clasificación de imágenes.

No vamos a usar:

encoder-decoder

segmentación

decoder

transformer como primera opción

Sí vamos a usar:

una CNN backbone preentrenada

una capa final nueva de 4 clases

La arquitectura conceptual es:

Imagen -> preprocesamiento -> CNN backbone -> Global Average Pooling -> capa fully connected -> softmax -> 4 clases

Modelo inicial

Vamos a empezar con una ResNet50 preentrenada como baseline.

Más adelante podremos comparar con modelos como:

DenseNet121

ConvNeXt-Tiny

Pero el primer modelo a implementar debe ser ResNet50.

Flujo de entrada al modelo

Las imágenes originales del dataset son de aproximadamente 424x424x3.

El flujo de preprocesamiento para el modelo será:

cargar imagen

aplicar resize/crop al tamaño esperado por la CNN

convertir a tensor

normalizar con los parámetros estándar del modelo preentrenado

formar batches

entrenar la red

Para entrenamiento, queremos usar augmentations razonables como:

random crop

horizontal flip

rotaciones suaves

Para validación e inferencia:

resize estable

center crop

normalización

Qué queremos construir primero

La prioridad actual es toda la parte de entrenamiento del modelo, no el backend ni el frontend todavía.

Necesitamos construir un pipeline completo que incluya:

carga y unión de los archivos del dataset

limpieza y preparación de etiquetas

dataset final para entrenamiento

dataloaders de train / validation / test

implementación del modelo con ResNet50

reemplazo de la última capa para 4 clases

entrenamiento

evaluación

métricas

matriz de confusión

guardado de checkpoints

script de inferencia

Tecnologías previstas para la fase de CNN

Queremos trabajar con:

Python

PyTorch

Torchvision

pandas

scikit-learn

Pillow

matplotlib