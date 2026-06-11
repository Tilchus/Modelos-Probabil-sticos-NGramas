# Modelos-Probabil-sticos-NGramas
Análisis Estadístico y Preprocesamiento Lingüístico de un Corpus Educativo
# Pipeline de N-gramas: Modelos Probabilísticos en NLP

Este repositorio contiene el desarrollo técnico y los artefactos analíticos diseñados para el procesamiento, saneamiento lingüístico y análisis estadístico de expectativas estudiantiles en la educación superior.

## 🏫 Información Académica
* **Institución:** Instituto Tecnológico Beltrán (ISFT 197)
* **Carrera:** Ciencia de Datos e Inteligencia Artificial (2° Año)
* **Materia:** Técnicas de Procesamiento del Habla
* **Profesora:** Yanina Escudero
* **Alumna:** Silvana Alejandra Gerez
* **Año Lectivo:** 2026

---

## 🛠️ Arquitectura del Pipeline (Estructura en Bloques)

El script principal está diseñado bajo una arquitectura modular organizada en **5 bloques lógicos** secuenciales:

1. **Configuración del Entorno:** Importación de librerías esenciales (`spaCy`, `NLTK`, `Scikit-Learn`, `Pandas`, `Matplotlib`).
2. **Gestión del Corpus:** Carga optimizada del archivo externo `CorpusEducacion.txt` utilizando codificación `latin-1` para resolver conflictos de caracteres en español.
3. **Preprocesamiento Avanzado (NLP):** Tokenización, remoción estricta de *stop words* y signos de puntuación, y lematización automatizada mediante el modelo `es_core_news_sm` de spaCy.
4. **Análisis Estadístico:** Extracción de Bigramas y Trigramas mediante `CountVectorizer`, aplicando un umbral de frecuencia absoluta mínima (`min_df=2`).
5. **Visualización y Orquestación:** Resolución dinámica de rutas mediante `os.path` y generación de un gráfico de barras horizontal con las combinaciones de palabras más representativas.

---

## 📂 Archivos del Repositorio

* 📄 `pipeline_ngramas.py`: Código fuente modular en Python con la orquestación completa del proceso.
* 📝 `CorpusEducacion.txt`: Dataset de origen con los registros textuales procesados.
* 📊 `Presentacion_Modelos_Probabilistas.pdf`: Diapositivas utilizadas para la defensa oral del proyecto.

---

## 🚀 Requisitos e Instalación

Para ejecutar este pipeline en tu entorno local, asegurate de contar con Python y las dependencias necesarias:

```bash
pip install pandas matplotlib scikit-learn nltk spacy
python -m spacy download es_core_news_sm
