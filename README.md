# Inicialización del proyecto

## Creación del entorno virtual en windows

1. Instalar virtualenv
```bash
pip install virtualenv
```

2. Crear el entorno virtual
```bash
virtualenv streamlitEnv
```

3. Activar el entorno virtual
```bash
streamlitEnv\Scripts\activate
```

## Instalamos requirements.txt
```bash
pip install -r requirements.txt
```

## ❗️ Especificar la ruta hacia los datos por medio de una variable de entorno
```bash
setx DIR_DATA "C:\Users\Usuario\Documents\Proyectos\streamlit\datos"
```

## Ejecutar el proyecto
```bash
streamlit run streamlit_app.py
```