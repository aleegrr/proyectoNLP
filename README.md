# Clasificador de Noticias Falsas con Análisis de Sentimiento

Este proyecto implementa un modelo de aprendizaje automático para clasificar titulares de noticias como **Falsos** o **Reales**. Además, incorpora un análisis de sentimiento para proporcionar información adicional sobre el tono emocional del titular.

## Descripción del Proyecto

El objetivo principal de este proyecto es construir un sistema capaz de analizar un titular de noticia y determinar su probabilidad de ser falso o real. Para lograr esto, se utiliza un modelo de clasificación de texto entrenado con técnicas de Procesamiento del Lenguaje Natural (PLN).

El proyecto consta de dos partes principales:

1.  **`model.ipynb`:** Este script se encarga de la carga, preprocesamiento y entrenamiento del modelo de clasificación. Utiliza un conjunto de datos de entrenamiento para aprender a distinguir entre titulares falsos y reales. También realiza un análisis de sentimiento sobre los datos de entrenamiento. Finalmente, guarda el modelo entrenado y el vectorizador TF-IDF para su posterior uso.

2.  **`app.py`:** Este script utiliza la librería Streamlit para crear una interfaz web interactiva. Los usuarios pueden ingresar un titular de noticia a través de esta interfaz, y la aplicación mostrará la predicción del modelo (Falso o Real), la probabilidad asociada a cada clase y un análisis del sentimiento del titular.

## Estructura del Repositorio

```
├── data/
│   ├── training_data.csv      # Archivo con los datos de entrenamiento (etiquetados).
│   └── testing_data.csv       # Archivo con los datos de prueba (para evaluar el modelo).
├── fake_news_model.joblib     # Modelo de clasificación entrenado (guardado por training_script.py).
├── tfidf_vectorizer.joblib    # Vectorizador TF-IDF entrenado (guardado por training_script.py).
├── app.py                     # Script de la aplicación web de Streamlit.
├── model.ipynb                # Script de preprocesamiento y enetrenamiento del modelo.
└── README.md                  # Este archivo, que proporciona información general del proyecto.
```

## Cómo Ejecutar el Proyecto

Sigue estos pasos para ejecutar el proyecto en tu entorno local:

1.  **Clonar el repositorio:**

    ```bash
    git clone [URL_DEL_REPOSITORIO]
    cd [NOMBRE_DEL_REPOSITORIO]
    ```

2.  **Crear un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    venv\Scripts\activate   # En Windows
    ```

3.  **Instalar las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el script de entrenamiento (opcional, si no tienes los archivos `.joblib`):**

    ```bash
    python training_script.py
    ```

    Este script descargará los recursos de NLTK necesarios, cargará y preprocesará los datos de entrenamiento, entrenará el modelo de Regresión Logística, realizará un análisis de sentimiento y guardará el modelo y el vectorizador.

5.  **Ejecutar la aplicación web de Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Esto abrirá automáticamente una nueva pestaña en tu navegador con la interfaz de la aplicación.

## Uso de la Aplicación Web

1.  Una vez que la aplicación se esté ejecutando en tu navegador, verás un título que dice "Clasificador de Noticias Falsas".
2.  Habrá un cuadro de texto donde podrás ingresar cualquier titular de noticia que desees analizar.
3.  Después de ingresar el titular (y presionar Enter o hacer clic fuera del cuadro), la aplicación mostrará:
    - **Resultado de la Predicción:** Indicando si el titular se clasifica como "Falso" o "Real".
    - **Probabilidad:** Mostrando la probabilidad de que el titular pertenezca a cada clase (Falso y Real).
    - **Análisis de Sentimiento:** Indicando si el sentimiento del titular es Positivo, Negativo o Neutral, junto con una puntuación numérica.

## Detalles Técnicos

### `training_script.py`

- **Librerías Utilizadas:** `pandas`, `sklearn` (para `train_test_split`, `TfidfVectorizer`, `LogisticRegression`, `accuracy_score`, `classification_report`), `nltk` (para `stopwords`, `WordNetLemmatizer`, `SentimentIntensityAnalyzer`), `re`, `joblib`.
- **Preprocesamiento de Texto:** Se aplican las siguientes técnicas:
  - Conversión a minúsculas.
  - Eliminación de caracteres especiales y números.
  - Tokenización.
  - Eliminación de stopwords (palabras comunes en inglés).
  - Lematización (reducción de las palabras a su forma base).
- **Vectorización:** Se utiliza la técnica TF-IDF (Term Frequency-Inverse Document Frequency) para convertir el texto preprocesado en vectores numéricos que el modelo puede entender.
- **Modelado:** Se entrena un modelo de Regresión Logística para la clasificación.
- **Evaluación:** El modelo se evalúa utilizando precisión y un informe de clasificación en un conjunto de prueba separado.
- **Análisis de Sentimiento:** Se utiliza VADER para calcular una puntuación de sentimiento compuesto para cada titular.
- **Persistencia del Modelo:** El modelo entrenado y el vectorizador TF-IDF se guardan en archivos `.joblib` para su posterior uso en la aplicación web.

### `app.py`

- **Librerías Utilizadas:** `streamlit`, `joblib`, `pandas`, `sklearn` (para las clases de los objetos cargados), `nltk` (para los recursos), `re`.
- **Interfaz de Usuario:** Streamlit se utiliza para crear una interfaz web simple con un cuadro de entrada de texto y áreas para mostrar la predicción, la probabilidad y el análisis de sentimiento.
- **Carga de Modelo y Vectorizador:** El modelo y el vectorizador guardados por el script de entrenamiento se cargan utilizando `joblib`.
- **Predicción en Tiempo Real:** Cuando el usuario ingresa un titular, se aplica el mismo preprocesamiento y vectorización utilizados durante el entrenamiento, y luego el modelo cargado realiza la predicción.
- **Visualización de Resultados:** Streamlit se utiliza para mostrar los resultados de la predicción y el análisis de sentimiento de manera clara al usuario.

## Posibles Mejoras Futuras

- **Experimentar con otros modelos de clasificación:** Probar diferentes algoritmos como Naive Bayes, Support Vector Machines o modelos más avanzados como Transformers.
- **Optimización de hiperparámetros:** Ajustar los parámetros del modelo y del vectorizador para mejorar el rendimiento.
- **Incorporar más características:** Considerar la inclusión de otras características como información del autor, fuente de la noticia, o metadatos.
- **Mejorar el análisis de sentimiento:** Explorar modelos de análisis de sentimiento más sofisticados o entrenar uno específico para el dominio de las noticias.
- **Implementar retroalimentación del usuario:** Permitir a los usuarios indicar si la predicción fue correcta o incorrecta para refinar el modelo con el tiempo.
- **Visualizaciones más avanzadas:** Utilizar gráficos para mostrar las probabilidades o la distribución del sentimiento.

## Contribuciones

¡Las contribuciones a este proyecto son bienvenidas! Si tienes ideas para mejorar el código, agregar nuevas funcionalidades o corregir errores, no dudes en crear un "pull request".
