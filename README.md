# 📘 **Proyecto de Análisis Sísmico**
<img width="1918" height="1042" alt="image" src="https://github.com/user-attachments/assets/dbc460fc-3644-49ae-8c7e-c051684494d1" />

## 🎯 Objetivo
Este proyecto contiene un dashboard interactivo para explorar datos sísmicos del USGS y scripts auxiliares.

## 📂 Estructura clave
- `app_clean.py` – Dashboard principal (Streamlit + Plotly).
- `final.ipynb` – Notebook que muestra cómo cargar y pre‑procesar el dataset USGS.

## 🚀 Cómo usar `app_clean.py`
1. **Instalar dependencias**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install streamlit pandas numpy plotly tqdm
   ```
2. **Ejecutar el dashboard**
   ```bash
   streamlit run app_clean.py
   ```
   El dashboard se abrirá en `http://localhost:8501`.

## 📓 Cómo usar `final.ipynb`
1. **Instalar Jupyter** (si no lo tienes)
   ```bash
   pip install notebook
   ```
2. **Abrir el notebook**
   ```bash
   jupyter notebook final.ipynb
   ```
3. **Ejecutar todas las celdas** para cargar el CSV `usgs_data/earthquakes_unificado.csv`, convertir fechas y visualizar una vista previa del DataFrame.

## 👥 Autores
- **Juan Gelviz**
- **William Felipe Rodríguez**
- **Tatiana Castaño Morales**
- **Victoria Bayona**

## 📚 Clase
Matemáticas para Big Data – Diplomado

> **Nota:** Mantén los datos en la carpeta `usgs_data/` y asegúrate de que el archivo CSV esté presente antes de ejecutar el dashboard o el notebook.
