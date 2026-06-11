# ==========================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ==========================================================
import os
import string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Descargas inicial NLTK (descomentar si es la primera ejecución)
# nltk.download('punkt')
# nltk.download('stopwords')

# Inicialización del modelo de spaCy procesamiento español
try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
except ImportError:
    nlp = None
    print("Nota: spaCy o el modelo 'es_core_news_sm' no están disponibles. Se omitirá la lematización.")


# ==============================
# 2. GESTIÓN Y CARGA DEL CORPUS 
# ==============================
def cargar_corpus(ruta_archivo):
    """
    Lee un archivo de texto externo de forma dinámica.
    Devuelve una lista de documentos eliminando líneas vacías y espacios extra.
    Nota: Se utiliza 'latin-1' para corregir error de caracteres en español
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en la ruta esperada: {ruta_archivo}")
        
    with open(ruta_archivo, "r", encoding="latin-1") as f:
        lineas = [linea.strip() for linea in f.readlines() if linea.strip()]
    return lineas


# ==============================================================================
# BLOQUE 3: PIPELINE DE PREPROCESAMIENTO DE TEXTO (NLP) - CORREGIDO
# ==============================================================================
def preprocesar_texto_spa(texto_puro):
    """
    Aplica tokenización, remoción de stop words/puntuación y lematización e
    Incluye un bypass explícito para corregir el sesgo de lematización en 'enseñanza'.
    """
    # 1. Tokenización inicial
    palabras = word_tokenize(texto_puro.lower())
    
    # 2. Definición de Stop Words y caracteres de puntuación a eliminar
    stop_words_es = set(stopwords.words('spanish'))
    puntuacion_personalizada = set(string.punctuation + '¿' + '?' + '¡' + '!' + '...' + '`' + '´')
    
    # Filtrar palabras vacías y signos
    palabras_limpias = [p for p in palabras if p not in stop_words_es and p not in puntuacion_personalizada]
    
    # Reconstruir texto temporal para spaCy
    texto_puente = " ".join(palabras_limpias)
    
    # 3. Lematización con spaCy con control de excepciones 
    documento = nlp(texto_puente)
    
    tokens_lematizados = []
    for token in documento:
        # Control del sesgo: si la palabra original es "enseñanza", forzamos su lema correcto 
        if token.text == "enseñanza":
            tokens_lematizados.append("enseñanza")
        else:
            # Para el resto del corpus, se aplica la lematización estadística normal
            tokens_lematizados.append(token.lemma_)
    
    # Retornar el texto normalizado listo para el vectorizador
    return " ".join(tokens_lematizados)

# ====================================================
# 4. ANÁLISIS ESTADÍSTICO Y VISUALIZACIÓN DE N-GRAMAS
# ====================================================
def analizar_y_graficar_ngramas(corpus_procesado):
    """
    Construye la matriz de términos utilizando CountVectorizer 
    bajo restricciones.
    Extrae las frecuencias de las combinaciones y 
    genera un gráfico comparativo.
    """
    print("\n--- Analizando N-gramas ---")
    
    # Configuración del modelo probabilístico: Bigramas y 
    # Trigramas con umbral mínimo de 2 apariciones
    vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=2)
    
    # Ajuste del modelo estadístico y transformación de la matriz dispersa
    X = vectorizer.fit_transform(corpus_procesado)
    
    # Mapeo de términos y consolidación de frecuencias absolutas
    n_gramas_nombres = vectorizer.get_feature_names_out()
    frecuencias = X.sum(axis=0).A1
    
    # Estructuración de datos para análisis tabular
    df_resultados = pd.DataFrame({
        'N-grama': n_gramas_nombres,
        'Frecuencia': frecuencias
    })
    
    # Segmentación de los 20 términos más significativos 
    # para visualizar las métricas
    df_top = df_resultados.sort_values(by='Frecuencia', ascending=False).head(20)
    print("\nTop 20 N-gramas más frecuentes encontrados (Frecuencia >= 2):")
    print(df_top.to_string(index=False))
    
    # Construcción de la interfaz gráfica del reporte (Matplotlib)
    plt.figure(figsize=(12, 8))
    plt.barh(df_top['N-grama'], df_top['Frecuencia'], color='teal', edgecolor='black')
    plt.xlabel('Cantidad de Apariciones (Frecuencia)', fontsize=12)
    plt.ylabel('2-gramas y 3-gramas Detectados', fontsize=12)
    plt.title('Comparación de Modelos de N-gramas (Corpus Educación 2025)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # El término más frecuente se posiciona en el extremo superior
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# =======================================================
# 5. DIRECCIONAMIENTO DEL PROCESO Y CONTROL DE EJECUCIÓN
# =======================================================
def ejecucion_principal():
    """
    Controlador central del script. Coordina el flujo 
    secuencial de datos desde la lectura inicial hasta la 
    presentación analítica final.
    """
    # Determinamos dinámicamente la carpeta exacta donde 
    # está guardado este script `.py`
    carpeta_del_script = os.path.dirname(os.path.abspath(__file__))
    
    # Construimos la ruta absoluta hacia el archivo de texto dentro 
    # de esa misma carpeta
    ruta_completa_corpus = os.path.join(carpeta_del_script, "CorpusEducacion.txt")
    
    try:
        # Etapa 1: Captura de datos de entrada empleando la ruta 
        # absoluta corregida
        documentos_crudos = cargar_corpus(ruta_completa_corpus)
        print(f"Se cargaron exitosamente {len(documentos_crudos)} registros del archivo.")
        
        # Etapa 2: Transformación en el pipeline 
        corpus_preparado = [preprocesar_texto_spa(doc) for doc in documentos_crudos]
        
        # Etapa 3: Modelado probabilístico reporte visual
        analizar_y_graficar_ngramas(corpus_preparado)
        
    except Exception as e:
        print(f"Ocurrió un error crítico durante la ejecución: {e}")

if __name__ == "__main__":
    ejecucion_principal()