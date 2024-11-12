Stochastic Neighbor Embedding (SNE) es un método de reducción de dimensionalidad que proyecta datos de alta dimensión en un espacio de baja dimensión (normalmente 2D o 3D) para facilitar la visualización. Su principal objetivo es preservar las relaciones de proximidad entre puntos en alta dimensión en la representación de baja dimensión.
Pasos clave del algoritmo:

    Cálculo de probabilidades en alta dimensión: SNE define la similitud entre puntos en el espacio de alta dimensión usando probabilidades. Para cada par de puntos, calcula la probabilidad de que uno sea un "vecino" del otro basado en una distribución Gaussiana centrada en cada punto. Este cálculo depende de un parámetro llamado perplejidad, que controla la cantidad de vecinos efectivos.

    Cálculo de probabilidades en baja dimensión: Se asignan posiciones iniciales aleatorias a los puntos en baja dimensión y se define la similitud entre estos puntos usando una distribución de probabilidad diferente, normalmente la distribución t-distribución en t-SNE (una variante de SNE). Esto ayuda a reducir el problema de la acumulación de puntos en baja dimensión.

    Minimización de la divergencia KL: SNE ajusta las posiciones de los puntos en baja dimensión minimizando la divergencia de Kullback-Leibler (KL) entre las distribuciones de probabilidad en alta y baja dimensión. Esto se realiza a través de un proceso iterativo, moviendo los puntos de baja dimensión para que sus relaciones de similitud reflejen mejor las relaciones en alta dimensión.

Variantes de SNE:

    t-SNE (t-Distributed SNE) es una variante que emplea una distribución t en lugar de Gaussiana en baja dimensión, lo que mejora la separación de grupos densos y es más popular en aplicaciones de visualización.

SNE y sus variantes, como t-SNE, son populares en análisis de datos de alta dimensión, ya que destacan patrones, grupos y estructuras internas en los datos que pueden ser difíciles de detectar en espacios de mayor dimensión.
