# 📘 **Proyecto de Análisis Sísmico**
<img width="1918" height="1042" alt="image" src="https://github.com/user-attachments/assets/5db37498-685a-4253-b367-d3d2580cca4b" />

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

## 📸 Capturas de pantalla
<img width="1684" height="284" alt="image" src="https://github.com/user-attachments/assets/1cf36952-4552-4f91-bde9-e937a4306744" />
<img width="1688" height="758" alt="image" src="https://github.com/user-attachments/assets/03734f29-b1f7-45d2-97e2-2a3bc84ddabe" />
<img width="975" height="475" alt="image" src="https://github.com/user-attachments/assets/b075845f-cf01-4284-b2a4-50db2d961531" />
<img width="975" height="376" alt="image" src="https://github.com/user-attachments/assets/6ad530c3-c076-4927-988a-09350ce177f9" />
<img width="975" height="477" alt="image" src="https://github.com/user-attachments/assets/dabfe1b1-26cf-4330-a07a-f2c231e32080" />
<img width="975" height="408" alt="image" src="https://github.com/user-attachments/assets/803a442c-2d45-4841-9cb9-136d73ada889" />
<img width="1918" height="1042" alt="image" src="https://github.com/user-attachments/assets/962e06b7-85d2-400a-b3fd-8c55409d6259" />
<img width="975" height="395" alt="image" src="https://github.com/user-attachments/assets/5900c302-5968-4f5c-a62b-48295fa9b046" />
<img width="975" height="542" alt="image" src="https://github.com/user-attachments/assets/6d638eeb-be57-4063-b5a4-727b1bd5dc57" />
<img width="975" height="421" alt="image" src="https://github.com/user-attachments/assets/1a71b2f4-e8ea-4c74-adac-ada15407cb2e" />
<img width="1681" height="790" alt="image" src="https://github.com/user-attachments/assets/b4119e2f-92bc-4f2b-84d3-59e825124b79" />


## 👥 Autores
- **Juan Gelviz**
- **William Felipe Rodríguez**
- **Tatiana Castaño Morales**
- **Victoria Bayona**

## 📚 Clase
Matemáticas para Big Data – Diplomado

> **Nota:** Mantén los datos en la carpeta `usgs_data/` y asegúrate de que el archivo CSV esté presente antes de ejecutar el dashboard o el notebook.
