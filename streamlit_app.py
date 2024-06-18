import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Toma de tiempo de ejecución y coloreado de texto
import time
from termcolor import colored

import sqlite3

import os

# Importamos la librería para cargar las variables de entorno
from dotenv import load_dotenv

# keywords, atmosphere, awards, price_range, features y special_diets no se encuentran en el dataset


# Carga las variables de entorno del archivo .env en la raíz del proyecto
load_dotenv()

DB_PATH = os.getenv("DB_PATH")

@st.cache_resource
def get_db_connection():
    # Establecer la conexión con la base de datos SQLite
    # Evita el error de "OperationalError: SQLite objects created in a thread can only be used in that same thread." 
    conn = sqlite3.connect(DB_PATH, check_same_thread=False) 
    return conn

# MEJORA Se puede dividir los datos de la BBDD en diferentes tablas para mejorar el rendimiento
@st.cache_data
def df_filtered_by_country_and_city(country, city):
    """
    Retrieves a filtered DataFrame of reviews based on the specified country and city.

    Parameters:
    - country (str): The country to filter by.
    - city (str): The city to filter by.

    Returns:
    - df_filtered (pandas.DataFrame): The filtered DataFrame of reviews.
    """
    start_time = time.time()

    # Establecer la conexión con la base de datos SQLite
    conn = get_db_connection()

    # Crear la consulta SQL
    sql_query = f"""
    SELECT *
    FROM resenas
    WHERE country = '{country}' AND city = '{city}'
    """

    # Ejecutar la consulta y leer los resultados en un DataFrame de pandas
    df_filtered = pd.read_sql_query(sql_query, conn)

    end_time = time.time()

    # Imprimimos el tiempo de ejecución en milisegundos
    print(colored('Tiempo de ejecución de df_filtered_by_country_and_city: {} milisegundos'.format(round((end_time - start_time) * 1000, 2)), 'yellow'))

    return df_filtered

@st.cache_data
def countries():
    """
    Retrieves a DataFrame containing distinct countries from the 'resenas' table.

    Returns:
        df_countries (pandas.DataFrame): DataFrame containing distinct countries.
    """
    start_time = time.time()

    conn = get_db_connection()

    # Crear la consulta SQL
    sql_query = """
    SELECT DISTINCT country
    FROM resenas
    """

    # Ejecutar la consulta y leer los resultados en un DataFrame de pandas
    df_countries = pd.read_sql_query(sql_query, conn)

    end_time = time.time()

    # Imprimimos el tiempo de ejecución en milisegundos
    print(colored('Tiempo de ejecución de countries: {} milisegundos'.format(round((end_time - start_time) * 1000, 2)), 'yellow'))

    return df_countries

@st.cache_data
def cities_of_a_country(df, country):
    """
    Returns a list of unique cities in a given country from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    country (str): The name of the country.

    Returns:
    list: A list of unique cities in the given country.
    """
    start_time = time.time()

    df_cities = df[df['country'] == country]['city'].unique()
    
    end_time = time.time()
    print(colored('Tiempo de ejecución de cities_of_a_country: {} milisegundos'.format(round((end_time - start_time) * 1000, 2)), 'yellow'))
    return df_cities

@st.cache_data
def cities_of_a_country(country):
    """
    Retrieves the distinct cities of a given country from the 'resenas' table in the 'resenas.db' SQLite database.

    Parameters:
    - country (str): The name of the country.

    Returns:
    - cities (list): A list of strings representing the distinct cities of the given country.

    Example Usage:
    >>> cities = cities_of_a_country('Spain')
    >>> print(cities)
    ['Madrid', 'Barcelona', 'Valencia', ...]
    """
    print(colored('Consultando las ciudades del país: {}'.format(country), 'yellow'))

    # Medimos el tiempo de inicio
    start_time = time.time()

    # Conectar a la base de datos SQLite
    conn = sqlite3.connect('resenas.db')

    # Crear un cursor
    cur = conn.cursor()

    # Ejecutar la consulta SQL
    cur.execute("SELECT DISTINCT city FROM resenas WHERE country=?", (country,))

    # Obtener todos los resultados
    cities = cur.fetchall()

    # Cerrar la conexión a la base de datos
    conn.close()

    # Convertir la lista de tuplas a una lista de strings
    cities = [city[0] for city in cities]

    # Medimos el tiempo de finalización
    end_time = time.time()

    print(colored('Tiempo de ejecución de cities_of_a_country: {} milisegundos'.format(round((end_time - start_time) * 1000, 2)), 'yellow'))

    return cities

# Método que cuenta los valores únicos de una columna del df
@st.cache_data
def unique_values_of_a_column(df, column):
    start_time = time.time()

    df_unique_values = df[column].value_counts()  # Aquí cambiamos unique() por value_counts().
    df_unique_values = pd.DataFrame(df_unique_values).reset_index()  # Aquí transformamos la serie en un DataFrame.

    end_time = time.time()
    print(colored('Tiempo de ejecución de unique_values_of_a_column: {} milisegundos'.format(round((end_time - start_time) * 1000, 2)), 'yellow'))
    return df_unique_values

@st.cache_data
def get_review_data(cuisine):
    conn = sqlite3.connect('resenas.db')
    cur = conn.cursor()

    cur.execute("""
        SELECT city, COUNT(*) as num_reviews 
        FROM resenas 
        WHERE cuisines LIKE ? 
        GROUP BY city
        ORDER BY num_reviews DESC
        LIMIT 10
    """, ('%' + cuisine + '%',))

    data = cur.fetchall()

    conn.close()

    return data

# No operar con las siguientes columnas
# "open_hours_per_week", "open_days_per_week", "original_open_hours", "special_diets", "features", "price_range", "atmosphere", "awards", "keywords"

# lista de roles o páginas
#roles = ["Inicio", "Turistas", "Propietarios de Restaurantes", "Inversores",
#         "Operadores turísticos / Agencias de Viajes", "Gobiernos locales / Oficinas de turismo",
#         "Investigadores / Académicos", "Aplicaciones de Entrega de Comida"]

# crea el menú superior usando st.tabs
inicio, turistas, propietarios, inversores = st.tabs(["Inicio", "Turistas", "Propietarios", "Inversores"] )

# muestra información basada en la selección del usuario
with inicio:
    # Presentación
    st.title("Visualizaciones interactivas con Streamlit")
    st.write("El presente análisis de datos contine registros de 1.083.397 restaurantes europes del popular sitio web TripAdvisor.")

    st.divider()

    # Visualización de datos generales de nuestro dataset
    ## Importamos nuestro df de restaurantes con los siguientes argumentos: header=True, inferSchema=True, sep=",", escape="\""
    df = pd.read_csv('../tripadvisor_european_restaurants.csv')

    ## Mostramos el número de registros
    ### Creamos una fila con 3 columnas
    col1, col2, col3 = st.columns(3)

    #### ¿Cuantos registros tenemos?
    with col1:
        st.metric(label = "Número de restaurantes", value = df.shape[0])

    #### ¿Cual es el país con más restaurantes?
    with col2:
        st.metric(label = "País con más restaurantes", value = df['country'].value_counts().idxmax())

    #### ¿Cual es la ciudad con más restaurantes?
    with col3:
        st.metric(label = "Ciudad con más restaurantes", value = df['city'].value_counts().idxmax())

    ### Creamos una segunda fila con 3 columnas
    col4, col5, col6 = st.columns(3)

    #### ¿Que porcentaje de restaurantes han sido claimed?
    with col4:
        st.metric(label = "Porcentaje de restaurantes claimed", value = round(df['claimed'].value_counts(normalize=True)[1]*100,2))

    #### ¿Cual es la calificación promedio de la columna 'food'?
    with col5:
        st.metric(label = "Calificación promedio de la comida", value = round(df['food'].mean(),2))

    #### ¿Cual es la calificación promedio de la columna 'service'?
    with col6:
        st.metric(label = "Calificación promedio del servicio", value = round(df['service'].mean(),2))

    ## mostramos solo 10 registros
    st.write(df.head(10))

    country = "Spain"
    city = "Madrid"

    ## Creamos un selectbox para seleccionar el país
    country = st.selectbox('Selecciona un país', countries() )

    ## Creamos un selectbox para seleccionar la ciudad
    city = st.selectbox('Selecciona una ciudad', cities_of_a_country(country))

    ## Creamos un dataframe con los restaurantes del país y ciudad seleccionados
    df_filtered = df_filtered_by_country_and_city(country, city)

    ## De df_filtered descartamos aquellos registros que carezcan de datos nulos en la columna 'working_shifts_per_week'
    # df_filtered = df_filtered[df_filtered['working_shifts_per_week'].notna()]

    ## Mostramos una distribución de turnos trabajados usando un histograma en base
    ## a la columna 'working_shifts_per_week'
    # Crea un histograma de la columna working_shifts_per_week
    if df_filtered.empty:
        st.write('No hay datos para el país y la ciudad seleccionados.')
    else:
        # Crea el histograma
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df_filtered['working_shifts_per_week'], 
                    nbinsx=10,
                    marker=dict(color='blue', line=dict(color='black', width=1))
                )
            ]
        )

        # Configura los títulos de los ejes y el título del gráfico
        fig.update_layout(
            title_text='Distribución de turnos trabajados por semana', 
            xaxis_title_text='Turnos trabajados por semana',
            yaxis_title_text='Frecuencia', 
            bargap=0.2, 
            bargroupgap=0.1
        )

        # Muestra la figura en Streamlit
        st.plotly_chart(fig)

        # Subtítulo para outliers
        st.subheader('Outliers')

        fig = go.Figure()

        columns = ['avg_rating', 'food', 'service', 'value']

        for col in columns:
            fig.add_trace(go.Violin(y=df_filtered[col], name=col,
                                    box_visible=True, line_color='black',
                                    meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                                    x0=col))

        fig.update_layout(yaxis_zeroline=False, violinmode='overlay')
        st.plotly_chart(fig)

    import json

    # Cargar los datos de cocina únicos
    with open('unique_cuisines.json', 'r') as f:
        unique_cuisines = json.load(f)

    # Ahora puedes usar 'unique_cuisines' en tu selectbox
    cuisine = st.selectbox('Selecciona un tipo de cocina:', unique_cuisines)

    # Obtén los datos de la base de datos
    data = get_review_data(cuisine)

    # Conviértelos a un DataFrame de pandas para un manejo más fácil
    df = pd.DataFrame(data, columns=['City', 'Number of Reviews'])

    # Crea el gráfico con Plotly
    fig = px.bar(df, x='City', y='Number of Reviews', title=f'Reseñas por cocina {cuisine}')

    # Muestra el gráfico en Streamlit
    st.plotly_chart(fig)

with turistas:
    # Mostrar la página de Turistas
    st.header("Página de Turistas")
    st.write("Contenido de la página de Turistas")

with propietarios:
    # Mostrar la página de Propietarios de Restaurantes
    st.header("Página de Propietarios de Restaurantes")
    st.write("Contenido de la página de Propietarios de Restaurantes")

with inversores:
    st.header("Página de Inversores")
    st.write("Seleccione un país y la ciudad que desee analizar.")

    # Cargamos df_con_resenas
    # df_con_resenas = pd.read_csv('df_con_resenas.csv')

    ## Creamos un selectbox para seleccionar el país
    countryInv = st.selectbox('Selecciona un país:', countries())

    ## Creamos un selectbox para seleccionar la ciudad
    cityInv = st.selectbox('Selecciona una ciudad:', cities_of_a_country(countryInv))

    if st.button('Mostrar datos'):
        print("El botón ha sido pulsado")

        st.write("Número de restaurantes por nivel de precio")

        ## Mostramos el número de registros
        ### Creamos una fila con 3 columnas
        col1, col2, col3 = st.columns(3)

        #### ¿Cuantos restaurantes con el price_level '€' hay?
        with col1:
            st.metric(label = "'€'", value = df[df['price_level']=='€'].shape[0])

        #### ¿Cuantos restaurantes con el price_level '€€€€' hay?
        with col2:
            st.metric(label = "'€€€€'", value = df[df['price_level']=='€€€€'].shape[0])

        #### ¿Cuantos restaurantes con el price_level '€€-€€€' hay?
        with col3:
            st.metric(label = "'€€-€€€'", value = df[df['price_level']=='€€-€€€'].shape[0])

        
        ## Si se ha seleccionado un país y una ciudad
        if countryInv and cityInv:
            ## Filtramos el df por país y ciudad
            df_filtered = df_filtered_by_country_and_city(countryInv, cityInv)

            # Crear un nuevo dataframe a partir de value_counts()
            price_level_counts = unique_values_of_a_column(df_filtered, 'price_level')

            # Renombrar las columnas para claridad
            price_level_counts.columns = ['Price Level', 'Count']

            # Generamos una gráfica de barras para mostrar la cantidad de restaurantes por cada nivel de precio
            fig = px.bar(price_level_counts, x='Price Level', y='Count', color='Price Level', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(title_text='Cantidad de restaurantes por nivel de precio', title_x=0.5)

            # Mostramos la gráfica en Streamlit
            st.plotly_chart(fig)
