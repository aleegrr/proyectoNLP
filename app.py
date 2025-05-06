import streamlit as st # Importa la librería Streamlit, que facilita la creación de aplicaciones web interactivas para el aprendizaje automático y la ciencia de datos. Se utiliza el alias 'st' para simplificar el acceso a sus funciones.
import joblib # Importa la librería joblib, utilizada para cargar los modelos y objetos de Python guardados de manera eficiente.
import pandas as pd # Aunque no se usa directamente en este script, pandas es una librería fundamental para la manipulación de datos y a menudo se utiliza en proyectos de aprendizaje automático, por lo que su importación podría ser una práctica común o anticipación de uso futuro.
from sklearn.feature_extraction.text import TfidfVectorizer # Aunque no se instancia directamente aquí, se importa porque el objeto vectorizador guardado fue creado con esta clase.
from sklearn.linear_model import LogisticRegression # Similar al vectorizador, se importa porque el modelo guardado fue entrenado con esta clase.
import nltk # Importa la librería Natural Language Toolkit (NLTK), esencial para tareas de procesamiento de lenguaje natural como la descarga de recursos y el acceso a stopwords y lematización.
import re # Importa la librería de expresiones regulares 're', utilizada para realizar búsquedas y manipulaciones de patrones en cadenas de texto (como la limpieza del texto).
from nltk.corpus import stopwords # Importa el módulo 'stopwords' de NLTK, que proporciona una lista de palabras comunes en varios idiomas que a menudo se eliminan del texto durante el preprocesamiento.
from nltk.stem import WordNetLemmatizer # Importa el lematizador WordNet de NLTK, una herramienta para reducir las palabras a su forma base o lema.
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Importa el analizador de sentimiento VADER (Valence Aware Dictionary and sEntiment Reasoner) de NLTK, especializado en el análisis de sentimiento en textos de redes sociales.

# --- Descargar recursos de NLTK ---
# Esta sección se asegura de que los recursos necesarios de NLTK (stopwords, WordNet y el léxico VADER) estén descargados.
# Se envuelve en bloques try-except para manejar el caso en que ya estén descargados, evitando descargas innecesarias.
try:
    stopwords.words('english') # Intenta acceder a la lista de stopwords en inglés para verificar si está descargada.
    WordNetLemmatizer().lemmatize('running') # Intenta usar la función de lematización para verificar si WordNet está descargado.
except LookupError:
    st.warning("Descargando recursos de NLTK (stopwords, wordnet)... Esto puede tardar un momento.") # Muestra una advertencia en la interfaz de Streamlit mientras se descargan los recursos.
    nltk.download('stopwords') # Descarga la lista de stopwords si no se encuentra.
    nltk.download('wordnet') # Descarga el léxico WordNet si no se encuentra.
    st.success("Recursos de NLTK descargados exitosamente.") # Muestra un mensaje de éxito una vez que los recursos se han descargado.
try:
    SentimentIntensityAnalyzer() # Intenta inicializar el analizador de sentimiento VADER para verificar si el léxico está descargado.
except LookupError:
    st.warning("Descargando léxico VADER de NLTK... Esto puede tardar un momento.") # Muestra una advertencia si el léxico VADER no está descargado.
    nltk.download('vader_lexicon') # Descarga el léxico VADER si no se encuentra.
    st.success("Léxico VADER de NLTK descargado exitosamente.") # Muestra un mensaje de éxito tras la descarga del léxico VADER.

# --- Definición de la función clean_text ---
# Esta función es idéntica a la utilizada en el script de entrenamiento y se encarga de preprocesar el texto ingresado por el usuario.
stop_words = set(stopwords.words('english')) # Crea un conjunto de stopwords en inglés para una búsqueda eficiente.
lemmatizer = WordNetLemmatizer() # Inicializa el lematizador WordNet.

def clean_text(text):
    text = text.lower() # Convierte el texto a minúsculas para asegurar la uniformidad.
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Elimina cualquier carácter que no sea una letra o un espacio en blanco, limpiando el texto de números y signos de puntuación.
    tokens = text.split() # Divide el texto en una lista de palabras (tokens) utilizando los espacios como delimitadores.
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Lematiza cada palabra que no sea una stopword, reduciéndola a su forma base.
    return " ".join(tokens) # Une los tokens limpios y lematizados de nuevo en una cadena de texto.

@st.cache_resource # Este decorador de Streamlit permite que la función load_model_and_vectorizer se ejecute solo una vez y almacene en caché los resultados. Esto es crucial para la eficiencia, ya que cargar modelos grandes puede ser costoso en términos de tiempo y recursos.
def load_model_and_vectorizer():
    loaded_model = joblib.load('fake_news_model.joblib') # Carga el modelo de clasificación de noticias falsas guardado previamente.
    loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib') # Carga el vectorizador TF-IDF guardado previamente, que se utilizó para transformar el texto durante el entrenamiento del modelo.
    sentiment_analyzer = SentimentIntensityAnalyzer() # Inicializa el analizador de sentimiento VADER, que se utilizará para analizar el sentimiento del titular ingresado.
    return loaded_model, loaded_vectorizer, sentiment_analyzer # Devuelve el modelo cargado, el vectorizador cargado y el analizador de sentimiento inicializado.

model, vectorizer, sentiment_analyzer = load_model_and_vectorizer() # Llama a la función para cargar el modelo, el vectorizador y el analizador de sentimiento, y los asigna a las variables correspondientes. Esta carga se realizará solo una vez gracias al decorador @st.cache_resource.

st.title("Clasificador de Noticias Falsas") # Establece el título principal de la aplicación web.
news_headline = st.text_input("Ingresa un titular de noticia:") # Crea un cuadro de texto donde el usuario puede ingresar un titular de noticia. El texto dentro de las comillas es la etiqueta que se muestra al usuario.

if news_headline: # Este bloque de código se ejecuta solo si el usuario ha ingresado texto en el cuadro de texto y ha interactuado (por ejemplo, presionando Enter o haciendo clic fuera del cuadro).
    cleaned_headline = clean_text(news_headline) # Aplica la función de limpieza de texto al titular ingresado por el usuario.
    vectorized_headline = vectorizer.transform([cleaned_headline]) # Utiliza el vectorizador TF-IDF *cargado* para transformar el titular limpio en un vector numérico, en el mismo formato que los datos de entrenamiento. La entrada a 'transform' debe ser una lista de textos.
    prediction = model.predict(vectorized_headline)[0] # Utiliza el modelo de clasificación *cargado* para predecir la clase del titular vectorizado. '[0]' se utiliza para extraer la predicción individual de la matriz resultante.
    probability = model.predict_proba(vectorized_headline)[0] # Obtiene las probabilidades de pertenencia a cada clase (en este caso, falso y real) para el titular vectorizado. '[0]' extrae las probabilidades para la única muestra.
    sentiment = sentiment_analyzer.polarity_scores(news_headline)['compound'] # Realiza un análisis de sentimiento en el titular original utilizando VADER y extrae la puntuación compuesta, que representa el sentimiento general.

    st.subheader("Resultado de la Predicción:") # Muestra un subencabezado para el resultado de la predicción.
    if prediction == 0: # Si la predicción es 0 (que se asumió que representa 'Falso' en el script de entrenamiento).
        st.write(f"El titular se clasifica como **Falso** (0).") # Muestra el resultado indicando que el titular es falso.
    else: # Si la predicción no es 0 (se asume que 1 representa 'Real').
        st.write(f"El titular se clasifica como **Real** (1).") # Muestra el resultado indicando que el titular es real.

    st.subheader("Probabilidad:") # Muestra un subencabezado para las probabilidades.
    st.write(f"Probabilidad de ser Falso: {probability[0]:.4f}") # Muestra la probabilidad de que el titular sea clasificado como falso, formateada a cuatro decimales.
    st.write(f"Probabilidad de ser Real: {probability[1]:.4f}") # Muestra la probabilidad de que el titular sea clasificado como real, formateada a cuatro decimales.

    st.subheader("Análisis de Sentimiento:") # Muestra un subencabezado para el análisis de sentimiento.
    if sentiment >= 0.05: # Si la puntuación de sentimiento compuesto es mayor o igual a 0.05, se considera positivo.
        st.write(f"Sentimiento: **Positivo** ({sentiment:.2f})") # Muestra que el sentimiento es positivo junto con la puntuación.
    elif sentiment <= -0.05: # Si la puntuación de sentimiento compuesto es menor o igual a -0.05, se considera negativo.
        st.write(f"Sentimiento: **Negativo** ({sentiment:.2f})") # Muestra que el sentimiento es negativo junto con la puntuación.
    else: # Si la puntuación de sentimiento compuesto está entre -0.05 y 0.05, se considera neutral.
        st.write(f"Sentimiento: **Neutral** ({sentiment:.2f})") # Muestra que el sentimiento es neutral junto con la puntuación.