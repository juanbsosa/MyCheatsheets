

import pandas as pd

​

​

def cantidad_ventas_por_barrio(anio):

    """

    Carga el dataset de propiedades en venta del año que recibe en un DataFrame y

    devuelve una Serie con la cantidad de propiedades en venta por cada barrio

    """

    # url del dataset aplicado en el año del parametro

    url = f"https://cdn.buenosaires.gob.ar/datosabiertos/datasets/departamentos-en-venta/departamentos-en-venta-{anio}.csv"

    # se carga y crea el DataFrame

    deptos = pd.read_csv(

        url, sep=";", error_bad_lines=False, warn_bad_lines=False

    )

    # se selecciona el barrio y se llama a value_counts para obtener total

    # de cada uno que finalmente se devuelve como salida de la función

    return deptos["BARRIO"].value_counts()

​

​

# imprime ventas por cada barrio

print(cantidad_ventas_por_barrio(2012))

​

# se define un diccionario vacío donde vamos a guardar los totales de cada año

# para luego usar como entrada para generar el dataframe final

totales_por_anio = {}

​

# se cargan los datos de cada año

for anio in range(2001, 2015):

    # se llama a la función de arriba de ventas para cada año

    totales_por_anio[str(anio)] = cantidad_ventas_por_barrio(anio)

​

    print(f"Datos del {anio} cargados")

​

# se crea el dataframe a partir del diccionario

ventas = pd.DataFrame(totales_por_anio)

​

# imprime el Dataframe

print(ventas)

