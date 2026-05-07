# Galaxy Morph

Análisis y clasificación de la morfología galáctica basado en el esquema de Hubble de Vaucouleurs, utilizando técnicas de aprendizaje profundo basadas en redes neuronales convolucionales

![Dataset Cover](./dataset-cover.png)
![Hubble de Vaucouleurs Diagram](./hubble-de-vaucouleurs.png)

## Introducción

La clasificación morfológica de galaxias representa la piedra angular sobre la cual se erige la astrofísica extragaláctica y la cosmología observacional moderna. Lejos de constituir un mero ejercicio taxonómico de categorización visual, la morfología de una estructura galáctica codifica de manera intrínseca información física vital sobre su historia de formación, la dinámica orbital de sus poblaciones estelares, su contenido de gas interestelar y la evolución térmica y estructural del universo a lo largo del inmenso tiempo cósmico. La apariencia visual de una galaxia no es aleatoria; es el resultado directo de las intrincadas relaciones causales entre la materia oscura invisible, la materia bariónica visible y los violentos procesos termodinámicos que operan a escala galáctica.

El estudio detallado de la morfología permite a los investigadores reconstruir el historial de ensamblaje de las galaxias. A lo largo del tiempo cósmico, el incremento de masa resultante de colisiones y fusiones galácticas, así como la disminución progresiva en la tasa de formación de nuevas estrellas, se acompañan de una acumulación paulatina de estrellas en el centro geométrico de las galaxias. Este proceso transforma dinámicamente las estructuras observadas, dictando de manera determinista si una galaxia exhibirá majestuosos brazos espirales ricos en gas frío donde nacen estrellas jóvenes y masivas, o si, por el contrario, se convertirá en un elipsoide dominado por poblaciones estelares antiguas, rojas y carentes de gas. Por lo tanto, establecer un sistema robusto de clasificación es la traducción directa de observables fotométricos en leyes fundamentales de la física evolutiva.

## Conceptos básicos de galaxias

Definimos una galaxia como un sistema ligado gravitacionalmente formado por estrellas, remanentes de estrellas, gas, polvo, y materia oscura. Se estima que existen unos 2 000 000 000 000 billones de galaxias en el Universo observable, que se encuentran principalmente en cúmulos o grupos. Cada elemento que las forma tiene sus características que permite distinguirlos visualmente

### Estrellas y cúmulos de estrellas.

Las estrellas son uno de los componentes más distinguibles de una galaxia, responsables de la emisión de luz visible y otras formas de radiación electromagnética, y son fundamentales para la estructura y dinámica de una galaxia. Se forman a partir de nubes de gas y polvo que colapsan gravitacionalmente. Pueden formarse de manera individual, en sistemas binarios, o en agrupaciones compactas de estrellas, que se forman a partir de la misma nube, y se denominan cúmulos estelares.

### Medio interestelar.

El espacio entre las estrellas no está vacío, sino que está compuesto por gas (un 70 % de hidrogeno, un 28 % de helio y un 2 % de materiales pesados), polvo, campos magnéticos, radiación electromagnética, y rayos cósmicos. Todas estas componentes desempeñan un papel importante en la estructura y evolución de la galaxia.

### Polvo interestelar.

Alrededor de un 1 % del medio interestelar está en forma de polvo y contiene el 50 % de los elementos pesados, como átomos de carbono, fragmentos de hielo, y otros elementos que se encuentran a temperaturas muy bajas (del orden de 200º bajo cero). Se encuentra distribuido por toda la galaxia y tiene dos efectos importantes: la extinción y enrojecimiento de la luz estelar, y la emisión de radiación térmica en el infrarrojo. El polvo que se encuentra delante de las estrellas cuando realizamos observaciones, absorbe la radiación electromagnética que emiten las estrellas, haciendo que la luz que nos llega tenga menor energía y, por tanto, parezca más roja y tenue. Cuando absorbe esta energía, se calienta.

### Nubes moleculares.

Son nubes de gran masa, densas y frías a unos 100 K compuestas por hidrógeno molecular H2. Estas nubes se encuentran en equilibrio gravitacional, pero cualquier perturbación puede hacer que colapsen sobre si mismas y tenga lugar formación estelar. No emiten luz en el espectro visible, pero pueden verse en longitudes de onda más largas (luz más roja) gracias a las transiciones entre los niveles rotacionales y vibracionales de la molécula.

### Regiones HII.

Las estrellas que se forman en las nubes moleculares emiten fotones ultravioletas muy energéticos que ionizan los átomos de hidrógeno de las regiones circundantes, dejando núcleos de hidrógeno y electrones, que después vuelven a recombinarse en un átomo de hidrógeno excitado en el que el electrón va decayendo desde los niveles de energía más altos hasta el nivel fundamental. En estas regiones se alcanza un equilibrio entre el ritmo de ionización y el de recombinación. Se observa un espectro de emisión que solamente emite luz en las longitudes de onda especificas correspondientes a la energía liberada por la desexcitación del electrón. En la siguiente figura, se muestra el espectro de visión de la nebulosa de Orión, que es una región en la que hay gas y polvo. Como en medio de todo ese material hay un cúmulo de estrellas jóvenes, observamos un espectro de emisión, no solo del hidrógeno, sino de otros materiales más pesados que también se encuentran en la zona.

### Hidrógeno atómico

Gran parte del medio interestelar se encuentra en forma de hidrógeno atómico H. Está distribuido en todo el disco galáctico y se observa exactamente a una longitud de onda de 21cm.

### Núcleo de galaxias.

En el centro de las galaxias se sitúan agujeros negros supermasivos con un orden de masa de millones de veces la masa de nuestro sol. El centro de nuestra galaxia está oculto en el rango visible por la presencia de polvo, pero en otras longitudes de onda se observa una alta densidad de estrellas, una fuente intensa de radio, y grandes arcos de gas producidos por electrones moviéndose a velocidades relativistas. Se han observado durante 15 años los movimientos de las estrellas en esta zona y se ha visto que orbitan en torno a un punto concreto a velocidades de 1500 km/s.

## Clasificación de galaxias

La sistematización de las formas galácticas es un campo académico con una rica historia de más de un siglo de desarrollo iterativo y debates científicos. Hace casi cien años, en 1926, el astrónomo estadounidense Edwin Hubble introdujo un modelo seminal basado en sus meticulosas observaciones a través de placas fotográficas, un modelo que hoy se conoce universalmente y de manera coloquial como el "Diapasón de Hubble" debido a la forma gráfica en la que se representa tradicionalmente. Este esquema pionero dividió inicialmente a las galaxias en tres grandes clases elípticas, espirales e irregulares, basándose estrictamente en su apariencia visual, prestando especial atención al tamaño de la región central (conocida como el bulbo) y al grado de compresión o enrollamiento de los brazos espirales.

Originalmente, Hubble teorizó, basándose en la tecnología óptica limitada de su época, que las galaxias con bulbos centrales más grandes y masivos tendían inexorablemente a poseer brazos espirales mucho más apretados. Esta correlación visual brindó durante décadas un soporte empírico fundamental al modelo teórico de "ondas de densidad" estáticas, el cual postulaba que los brazos espirales eran perturbaciones de densidad estacionarias a través de las cuales orbitaban las estrellas y el gas. Sin embargo, la evolución de este esquema taxonómico ha sido una constante en la astronomía moderna, refinándose a medida que la capacidad instrumental mejoraba.

Astrónomos posteriores, destacando principalmente Gérard de Vaucouleurs y Allan Sandage, expandieron significativamente el modelo original y algo rígido de Hubble para reflejar un continuo morfológico mucho más matizado y representativo de la realidad física del universo. El sistema de Vaucouleurs representó un salto conceptual masivo al introducir el concepto de la "Etapa de Hubble" (denotada por la letra T), asignando valores numéricos continuos a las clases morfológicas. Este índice numérico abarca desde T = -6 para galaxias elípticas extremadamente compactas (cE), pasando por el cero para las galaxias lenticulares, hasta alcanzar valores positivos progresivos para las espirales de tipo temprano, tardío y finalmente las irregulares.

Además de la indexación numérica, de Vaucouleurs integró formalmente la continuidad morfológica entre las galaxias espirales barradas y las no barradas (creando clases mixtas como SAB), así como la presencia o ausencia de anillos interiores y estructuras en forma de lente, mitigando en gran medida la subjetividad inherente a una clasificación basada puramente en estimaciones de resolución visual. Allan Sandage, por su parte, reconoció explícitamente a las galaxias de transición S0 (lenticulares) e introdujo subtipos que dependían directamente de la cantidad de polvo interestelar presente en el plano del disco, un factor crítico para entender la opacidad galáctica.

La forma de una galaxia es, en esencia, un registro fósil de su historia. Las investigaciones modernas han demostrado que las galaxias cambian y crecen a través de procesos de fusión y colisión. Hace unos 10,000 millones de años, las galaxias eran mucho más caóticas y pequeñas, con tasas de formación estelar diez veces superiores a las actuales. Casi todas las galaxias masivas han experimentado al menos una fusión mayor desde que el universo tenía 6,000 millones de años.

### Galaxias elípticas.

Las galaxias elípticas son concentraciones esferoidales de estrellas que parecen cúmulos globulares, aunque a gran escala. Tienen un perfil de luz brillante en el centro que decrece continuamente hacia el exterior con distribución regular de luz. Como su nombre indica, tienen forma elíptica, y no tienen ningún tipo de estructura interna que destaque a parte del núcleo. Típicamente contienen muy poco gas y polvo interestelar, y tampoco presentan formación estelar reciente, solo tienen poblaciones estelares viejas que le dan a la galaxia un aspecto más rojizo.

Las galaxias elípticas tienen unos perfiles de brillo que son muy tenues en las zonas externas, y puede ser un poco difícil delimitar el borde de la galaxia. Las isofotas son líneas que conectan zonas de la imagen con el mismo brillo superficial y las usamos para definir distintos radios y para ayudarnos a calcular la excentricidad de la elipse. El radio más utilizado es el radio efectivo y se define como el radio de la isofota que contiene la mitad de la luminosidad total de la galaxia.

Hubble las situó en su diagrama como galaxias tempranas, ya que, al observar su simplicidad y ausencia de estructuras más complejas, pensó que aún no se habían formado del todo. Sin embargo, los astrónomos en realidad piensan que el caso es al revés, que son galaxias más evolucionadas y que en la mayoría de los casos son el producto final de la unión de dos galaxias espirales. Aun así, se siguen llamando galaxias de tipo temprano por la nomenclatura que les dio Hubble.

### Galaxias lenticulares.

Son un punto intermedio entre las elípticas y las espirales. Son planas y tienen un disco un bulbo en el centro, pero no tienen brazos espirales.

### Galaxias espirales.

Las galaxias espirales son objetos muy dinámicos, cunas de formación estelar y con muchas estrellas jóvenes que se sitúan en un disco plano. Tienen bulbos centrales que sobresalen del plano del disco y tienden a estar hechos de estrellas más viejas con tonos más rojizos. En las regiones más externas presentan halos más difusos formados por las estrellas más antiguas del Universo. Todas las componentes del disco se estructuran en zonas más densas formando brazos espirales que están continua en rotación. Se dividen en dos tipos en función de si presentan una barra central o no. Todas estas galaxias tienen polvo, regiones HII nubes de gas y estrellas de todas las edades, así que cambia bastante su aspecto dependiendo de la longitud de onda a la que las observamos.

Las galaxias espirales se subdividen en distintos grupos en función de lo prominente que es el bulbo de estrellas, del brillo superficial total, y de lo abiertos o enrollados que estén los brazos. Estas tres características están relacionadas entre sí de modo que una galaxia espiral que muestra un abultamiento central mayor tendrá un brillo superficial mayor y los brazos espirales estarán pegados al centro y más enrollados. Mientras que una galaxia con un abultamiento central menor tendrá un disco más débil, y los brazos menos enrollados. A pesar de esta subdivisión y de la distinción de con barra y sin barra, se puede hacer una clasificación única de galaxias espirales siempre que tengan disco plano que presente brazos espirales.

Hubble las denomino galaxias de tipo tardío por la gran cantidad de componentes que tienen, y por su estructura con mayor complejidad que las elípticas. Sin embargo, se sabe que son galaxias más jóvenes por la intensa formación estelar que hay en los brazos del disco y por la gran cantidad de material que contiene, que aún no ha sido expulsado de la galaxia por la muerte de las estrellas en forma de supernova. Aun así, se siguen denominando con la nomenclatura que les dio Hubble.

Al contrario de lo que pasa en las elípticas, la orientación de las galaxias espirales cuando las observamos es crucial para entender su estructura y propiedades. Si una galaxia espiral se observa de frente, podemos ver con facilidad la forma de sus brazos espirales, la distribución de estrellas en ellos y el núcleo galáctico con mayor claridad. En cambio, si se observa de canto, no veremos bien ninguna de estas estructuras, pero obtendremos información sobre el grosor del disco galáctico, la presencia de un bulbo central y la distribución del polvo interestelar. La orientación también afecta a la interpretación de sus características físicas y dinámicas como la medición de la velocidad de rotación, o la determinación de la dinámica interna de la galaxia.

### Galaxias irregulares.

No tienen ni un disco dominante, ni estructura espiral, ni simetría aparente. Simplemente son un conjunto caótico de estrellas, gas y polvo.

## Efecto Doppler y Redshift

El efecto Doppler es el cambio en la frecuencia de una onda debido al movimiento relativo entre la fuente emisora y el observador. En el caso de la astronomía, el cambio lo vemos en las ondas electromagnéticas. Si una fuente de luz se aleja del observador, las longitudes de onda de la luz percibida se estiran, lo que resulta en un corrimiento al rojo. Por el contrario, si la fuente de luz se acerca al observador, las longitudes de onda se¸ comprimen”, lo que da lugar a un corrimiento al azul.

Aunque la gravedad hace que los objetos sean atraídos unos por otros, el Universo está en expansión acelerada, lo que hace que los objetos se alejen de nosotros y que experimenten un corrimiento al rojo denominado reddshift. Este corrimiento al rojo se cuantifica mediante la fracción del cambio en la longitud de onda observada en relación con la longitud de onda emitida.

El corrimiento al rojo se relaciona directamente con la velocidad radial del objeto a través de la ley de Hubble, que establece que la velocidad de recesión de un objeto es proporcional a su distancia, es decir, cuanto más lejos está un objeto, más rápido se aleja, luego la expansión es acelerada. Gracias al redshift y a la ley de Hubble, podemos calcular la distancia a la que se encuentran los objetos.

## Demografía, Capital Humano y el Cuello de Botella Clasificatorio

La astronomía profesional es una disciplina académica altamente especializada y demográficamente minúscula en comparación con la escala del cosmos que pretende analizar. La Unión Astronómica Internacional (IAU), la máxima autoridad mundial no gubernamental responsable de establecer definiciones y estándares astronómicos registra en su base de datos global un total de apenas 256,749 miembros individuales y juveniles activos. Estos profesionales están distribuidos a través de 92 países. Aproximadamente un 21% de estos expertos acreditados (unos 53,917 individuos) están radicados en los Estados Unidos. De hecho, estimaciones más amplias calculan que la comunidad mundial entera de investigadores astronómicos activos ronda los 200,000 profesionales, una población laboral que puede compararse con el tamaño de los habitantes de un pueblo mediano.

Este reducido pool de talento subraya la imposibilidad matemática y logística de clasificar visualmente el universo. Si el proyecto LSST arrojará 20 mil millones de galaxias, requeriría que cada uno de los 200,000 astrónomos del planeta evaluará manualmente 100 mil imágenes individuales. Incluso dedicando 24 horas al día sin descanso, el tiempo requerido superaría con creces la duración de la carrera académica de un individuo.

## Los Tiempos de Clasificación Históricos

Históricamente, la clasificación morfológica dependió de la inspección visual prolongada de grupos de expertos, un proceso plagado de sesgos subjetivos (donde incluso expertos renombrados no siempre coinciden plenamente en la topología de un objeto difuso) e inherentemente no escalable. En condiciones normales, clasificar manualmente apenas decenas de miles de galaxias es una tarea que le toma meses de dedicación exclusiva a un equipo de científicos capacitados.

Para intentar solventar este abismo operativo, la comunidad recurrió a la ciencia ciudadana. El proyecto Galaxy Zoo, nacido en 2007 al amparo de los datos del Sloan Digital Sky Survey, invitó al público general a observar imágenes y responder árboles de decisión morfológica. Aunque fue un éxito rotundo en la democratización de la ciencia, movilizando a cientos de miles de voluntarios globales, reveló el límite temporal de la fuerza bruta humana: el proyecto Galaxy Zoo original requirió aproximadamente 3 años de esfuerzo colaborativo masivo y continuo para obtener clasificaciones morfológicas redundantes y estadísticamente confiables de tan solo ~300,000 galaxias. Al ritmo de Galaxy Zoo, clasificar los catálogos del LSST tomaría decenas de miles de años.

## La Economía del Capital Humano: Sueldos y Costos

El costo financiero asociado a la clasificación manual a nivel profesional es otra variable prohibitiva. Los astrónomos e investigadores especializados en galaxias requieren típicamente un doctorado (Ph.D.) en física o astrofísica, seguido de un arduo camino formativo a través de múltiples posiciones postdoctorales temporales que suelen durar de 2 a 3 años.

## Problema a resolver

La incapacidad de los métodos tradicionales de clasificación manual para manejar el volumen masivo de datos generado por la astronomía moderna. Proyectos observacionales actuales producen cantidades de información que superan ampliamente la capacidad de análisis humano disponible. Esta limitación no solo genera retrasos significativos en el procesamiento de datos, sino que también incrementa los costos operativos. Además, la clasificación manual introduce inconsistencias y sesgos que afectan la calidad de los resultados científicos. En consecuencia, existe una brecha crítica entre la generación de datos y su análisis efectivo.

## Objetivos

### Objetivo General

Diseñar, evaluar e implementar un sistema integral de análisis y clasificación automática de la morfología galáctica, aprovechando el estado del arte en técnicas de Deep Learning y redes neuronales convolucionales CNN, con el propósito de erradicar el cuello de botella clasificatorio, eliminando el sesgo humano y posibilitando el procesamiento masivo, determinista y económicamente sostenible de los catálogos astronómicos modernos.

### Objetivos Específicos

- Desarrollo y Optimización Arquitectónica: Lograr la construcción matemática de una arquitectura de red neuronal profunda que posea la capacidad intrínseca de identificar de manera autónoma características físicas y patrones morfológicos sutiles en imágenes del espacio profundo.
- Ingeniería de Datos y Preprocesamiento: Alcanzar la consolidación de un flujo algorítmico de normalización que depure eficazmente imágenes borrosas, afectadas por ruido lumínico o artefactos instrumentales, garantizando la curación de un conjunto de datos (dataset) astronómico balanceado y estadísticamente representativo.
- Mapeo Taxonómico Determinista: Lograr la implementación de metodologías de lógica relacional que permitan traducir árboles de decisión probabilísticos en etiquetas categóricas rígidas, asegurando que el entrenamiento supervisado se alimente exclusivamente de datos con certeza empírica.
- Validación de Eficiencia Operativa y Precisión: Demostrar analíticamente que la inferencia algorítmica desarrollada reduce los tiempos históricos de clasificación a fracciones de segundo por imagen, ofreciendo una exactitud y un F1-Score comparables a los consensos de expertos humanos, validando así la sostenibilidad del modelo ante arquitecturas de macrodatos.

## Pre-Procesamiento y Normalización

### Medidas, Datos, Bases de Datos y Elaboración del Data-Set

Para desarrollar un modelo de clasificación utilizando visión por ordenador, es esencial tener con un conjunto de imágenes clasificadas que sirvan como base para el entrenamiento. Estas imágenes etiquetadas proporcionan los ejemplos necesarios para que el algoritmo aprenda a identificar los patrones y características que distinguen cada clase. Etiquetar individualmente las suficientes imágenes para tener un conjunto de entrenamiento de un tamaño decente, llevaría una cantidad de tiempo desmesurada y la clasificación de una sola persona podría no ser del todo acertada.

#### Galaxy Zoo Proyect

El Galaxy Zoo es un proyecto online que nace para clasificar morfológicamente galaxias a partir de imágenes espaciales. Los miembros voluntarios que participan responden preguntas sobre el aspecto de la galaxia, y en función de las respuestas, se lleva a cabo una ponderación de los resultados para poder clasificar las galaxias en dos tipos: elípticas y espirales. El 2 de agosto de 2007, Galaxy Zoo público su primer reporte, que explicaba que 80.000 voluntarios habían clasificado ya más de 10 millones de imágenes de galaxias. En este primer proyecto de clasificación se trató de minimizar el error humano mostrando imágenes en blanco y negro o al revés. Con esto, se obtuvieron resultados positivos ya que resulta que la aparente predominancia de las espirales en el sentido contrario a las agujas del reloj era realmente un error de percepción del ojo humano, algo que se evitó con este método.

En este primer proyecto, se trataban de clasificar las galaxias en elípticas y espirales principalmente en función del color, la presencia de formación estelar, y la forma. Después del éxito del Galaxy Zoo 1, se ha tratado de dar una clasificación visual fijándose en una mayor cantidad de detalles, como la presencia de barras, el número de brazos, o la apertura de estos. Esto se ha realizado mediante un árbol de decisión en el que, para pasar a cierta pregunta, tienes que haber dado una respuesta específica en la anterior. Con este método, cada galaxia es clasificada por varios miembros y se saca una fracción de votos de los voluntarios que han observado cierta característica en la imagen.

#### Selección del Dataset: Galaxy Zoo 2 (GZ2)

Galaxy Zoo 2 (GZ2) fue el proyecto sucesor de Galaxy Zoo. GZ2 amplía las clasificaciones originales de Galaxy Zoo para una submuestra de las galaxias más brillantes y grandes del lanzamiento Legacy, midiendo características morfológicas más detalladas. Esto incluye barras galácticas, brazos espirales y ángulo de paso, bulbos, galaxias vistas de canto, elipticidades relativas y muchas otras.

Lanzado tras el rotundo éxito de su predecesor, GZ2 abandonó la idea de clasificar todo el universo visible y se centró estratégicamente en una submuestra de aproximadamente 300.000 de las galaxias más grandes y brillantes del catálogo del Sloan Digital Sky Survey (SDSS). Más de 83.000 voluntarios participaron en esta fase, realizando alrededor de 16 millones de clasificaciones, lo que dio lugar al catálogo morfológico más detallado de su época.

Por un lado, están las imágenes del Galaxy Zoo 2 que provienen del Sloan Digital Sky Survey o SDSS que utiliza un telescopio terrestre de 2,5 metros de diámetro situado en el Observatorio Apache Point para obtener imágenes. Al ser un telescopio diseñado específicamente para el cartografiado digital de gran campo, la calidad de las imágenes permite cubrir una vasta área del cielo de forma sistemática. Las imágenes están procesadas para reducir el ruido térmico, lo que nos permite observar y clasificar cientos de miles de galaxias en el universo cercano con una claridad sin precedentes. Además, como el telescopio opera con un sistema de cinco filtros fotométricos (u, g, r, i, z), tenemos imágenes compuestas que revelan detalles morfológicos precisos como brazos espirales, barras y bulbos. Esto nos devuelve un dataset muy profundo y con mucha información estructural. El telescopio del SDSS está compuesto por un espejo primario de 2,5 metros que alimenta una cámara digital de 120 megapíxeles, lo que lo convierte en un instrumento con una eficiencia enorme que permite obtener un mapa tridimensional del cosmos con gran precisión estadística.

En el dataset de GZ2, una galaxia no tiene una etiqueta absoluta de "espiral" o "elíptica". Cada imagen fue evaluada de forma independiente por un promedio de 40 a 45 personas diferentes. El resultado de este esfuerzo colectivo se expresa en fracciones de votos.

Por ejemplo, si 40 personas evaluaron una imagen y 30 respondieron que veían una barra central, la fracción estadística para la característica "barra" es de 0.75 (75%). Este enfoque es revolucionario porque cuantifica el nivel de certidumbre: las características físicas obvias alcanzan fracciones cercanas a 1.0, mientras que las galaxias borrosas o de morfología dudosa tienden a tener votos divididos alrededor del 0.5.

La Tabla 1 introduce el método de Hart et al. (2016), una corrección matemática exhaustiva que logra "limpiar" los datos de casi 240.000 galaxias. En esta base de datos, cada galaxia está identificada por un código único del telescopio SDSS llamado objid. A este código le siguen decenas de columnas que contienen las fracciones matemáticas exactas para cada respuesta del árbol de decisión. La brillantez de esta tabla radica en que estas fracciones ya han sido ajustadas matemáticamente para contrarrestar el sesgo óptico, simulando cómo habrían votado los humanos si todas las galaxias estuvieran a una distancia y claridad ideales.
