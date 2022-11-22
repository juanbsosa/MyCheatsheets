#PYTHON

###------- VISUAL STUDIO CODE -------###

# KEYBOARD SHORTCUTS

# Guardar la imagen del workspace para poder volver a abrir los mismos archivos que en tu ultima sesion>
file -> Add folder to workspace -> Save Workspace As

# Habilitar la opcion de buscar y reemplazar en una seleccion (find and replace in selection) en Visual Studio Code:
desgargar extension Quick Replace In Selection

# Seleccionar varias lineas al mismo tiempo:
alt+shift

# Wrap text (que el texto se ajuste al tamaño de la ventana):
alt+z

# Seleccionar una palabra y luego seleccionar todas las repeticiones de esa palabra
ctrl + d

# Mostrar/visualizar la base de datos completa con todas las columnas (para ver en Visual Studio Code u otros):
pd.set_option('display.max_columns', None)
pd.set_option('max_row', None) #esta otra no sé qué hace


###----------------------------------###

###------- USEFUL COMMANDS -------###

## METODOS DE STRINGS
# Reemplazar una parte de un string por otra
saludo = 'hola'
print(saludo.replace('la', 'mbre'))

# Cambiar todo a mayuscula
place = "poolhouse"
print(place.upper())

# Cambiar la primera letra a mayuscula
print('hola'.capitalize())

## METODOS DE LISTAS
# Contar la cantidad de veces que aparece un elemento en una lista
lista1 = [1,2,3,4,1,1]
print(lista1.count(1))

# Obtener el indice de un objeto dentro de una lista / obtener la posicion de un elemento en una lista
lista1 = [1,2,3,4]
print(lista1.index(3))

# The ; sign is used to place commands on the same line. The following two code chunks are equivalent:
# Same line
print('Hello'); print('Bye')

# Separate lines
print('Hello')
print('Bye')

# Copiar objetos (listas)
lista1 = [1,2,3,4]
lista2 = lista1 # aca estas copiando la referencia a la lista1, no los objetos
del[lista2[2]]
print(lista1)
    # Para copiar los elementos, y no solo la referencia, hay que escribir
y = list[lista1]
# o
y = x[:]

# Eliminar un elemento de una lista
lista1 = [1,2,3,4]
del[lista1[2]]
print(lista1)

# Appendear / Agregar filas de otro data frame que no esten en el data frame actual
df_diff = df2[~df2.col1.isin(A.col1)]
df_full = pd.concat([df1, df_diff], ignore_index=True)

# Crear un archivo csv / Exportar un data frame a csv
df.to_csv("path")

# Cambiar el nombre de muchas columnas a la vez
cambio_cols = {'v1old': 'v1new', 'v2old': 'v2new'}
df.rename(columns=cambio_cols,
          inplace=True)

# Crear una variable fecha a partir de variables para anio mes y dia:
df['fecha'] = pd.to_datetime(dict(year=df.anio, month=df.mes, day=1)) # aca no tengo dia y le pongo 1

# Crear una columna / crear una variable en un data frame que tome diferentes valores de acuerdo una condicion
db['var'] = np.where(db['var2']=='valor', valorsitrue, valorsifalse)

# Crear una columna / crear una variable en un data frame que tome diferentes valores de acuerdo mas de una condicion
df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
conditions = [
    (df['Set'] == 'Z') & (df['Type'] == 'A'),
    (df['Set'] == 'Z') & (df['Type'] == 'B'),
    (df['Type'] == 'B')]
choices = ['yellow', 'blue', 'purple']
df['color'] = np.select(conditions, choices, default='black')
print(df)

# Crear una columna variable con un promedio ponderado por grupo:
func_prom_pond = lambda x: np.average(x, weights=db.loc[x.index, "pesos"])
db['prom_pond'] = db4.groupby('grupo')['variable'].transform(func_prom_pond)

# Crear una muestra aleatorio de una secuencia
random.sample(range(1, 11), k=5)

# Generar una muestra aleatoria de un data frame
db.sample(n=1000, replace=False)

# Seleccionar las filas de un data frame que tengan missing value en una columna
df[df['var'].isnull()]

# Eliminar duplicados segun una variable:
df.drop_duplicates(subset=['var'])

# Merge / unir dos data frames
db1 = db1.merge(db2[['var_a_mergear']] how='left', on=[nombre de variable llave], indicator=True) #indicator te dice si agregar una columma que te diga el resultado del merge, ademas de true se le puede poner el strign que quieras

# Intertar una columna / variable al principio de un data frame
df.insert(0, 'nombrevar', var)

# Cambiar la posicion de una columna en un data frame (aca se pone en la posicion 0, al principio)
col = df.pop('Name')
df.insert(0, 'Name', col)

# Eliminar todos los objetos creados por el usuario:
for element in dir():
    if element[0:2] != "__":
        del globals()[element] # VERR PORQUE GENERA PROBLEMAS

# Seleccionar los elementos de una lsita que empiecen con determinado string:
result = [i for i in some_list if i.startswith('string')]

# Quedarse con / seleccionar las columnas de un tipo / clase determinado:
df.select_dtypes(np.number) # u otra clase, no probe con otra

# Convertir todas las columnas & variables a una clase determinada
def f(x):
    try:
        return x.astype(float) # o cualquier otra clase
    except:
        return x
df2 = df.apply(f)

# Quedarse con la primera fila de cada grupo
db.grouby('var').first()

# Ordenar un data frame segun multiples columnas:
df.sort_values(['var1', 'var2'], ascending=[False, True])

# Exportar un grafico a png o pdf o jpg:
import matplotlib as plt

# Calcular media / promedio por grupo:
db.groupby('variable', as_index=False).mean()
# y agregarlo como columna>
db['nueava_var'] = db0.groupby('var_grupo')['var_a_promediar'].transform('mean')

### Hacer graficos de carrera de barras:
# Primero hay que bajarse un programa para manejar videos que ese llama ffmpeg, unzipearlo en una carpeta tipo C o Program files, y agregarlo al path. Usar este tutorial:
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# Despues instalar el paquete de python para ese progrma en el cmd:
pip install ffmpeg-python
# Se puede chequear si se hizo bien poniendo en el cmd:
ffmpeg -version
# Instalar el paquete para hacer graficos de carreras en el cmd:
pip install bar_chart_race
# Ejemplo de como se usa:
# es importante que la base este en formato wide. Ver tutorial: https://www.dunderdata.com/blog/official-release-of-bar_chart_race-a-python-package-for-creating-animated-bar-chart-races
bcr.bar_chart_race(
    df = db2,
    filename = path_figures +  'carrera_1.mp4')

# Actualizar un paquete con pip. En cmd escribir:
pip install paquete --upgrade

# Contar los missing values de todas las columnas de un data frame:
print(df.isnull().sum())

# Eliminar la ultima fila o la primer fila de un data frame:
df.drop(df.tail(n).index,inplace=True) # drop last n rows
df.drop(df.head(n).index,inplace=True) # drop first n rows

# Eliminar la ultima columna de un data frame:
df.drop(df.columns[[-1,]], axis=1, inplace=True)

# Separar una cadena por un character:
texto.split(',') #en este caso es una coma
objeto.text.split('\n') #acá es si el objeto no es texto, primero lo pasás a texto

# Separar una cadena por comas, ignorando las comas entre comillas "":
funcion = re.compile(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")
funcion.split(objeto)

# Equivalente al comando "fillin" de Stata
import itertools
cols_a_combinar = ["var1", "var2", "var3"]
combinaciones = []
for var in cols_a_combinar:
    combinaciones.append(db[var].unique().tolist())
df1 = pd.DataFrame(columns = cols_a_combinar, data=list(itertools.product(*combinaciones)))
#df1


# Convertir un array (series) a una lista:
array.tolist()

# Obtener los valores únicos de una columna/variable:
db["variable"].unique()
#Si querés que sea una lista:
db["variable"].unique().tolist()

# Factorizar/Encodear una variable (asignarle a cada valor único de string un número):
pd.factorize(db['variable'])

# Cread dummies para diferentes valores de una misma variable (en realidad se dice "one-hot" encoding, dummies seria si dejas una categoria afuera, como en una regresión):           
pd.get_dummies(df, columns=["variable"])

# Eliminar todas las variables que empiezan con determinado string:
db = db.drop(db.filter(like='stringinicial').columns, axis=1)

# Eliminar una lista de columnas / variables a la vez:
eliminar = ['var1', 'var2']
db.drop(eliminar, axis=1, inplace=True)

# Tabular una variable:
db.groupby(['variable']).size()
#mejor:
db.variable.value_counts()

# Tabular una variable y quedarse con los procentajes de cada grupo:
db.groupby(['var1', 'var2'])['var3'].agg('count') / db['var3'].agg('count')

# Traductor de Stata a Python
http://www.danielmsullivan.com/pages/tutorial_stata_to_python.html

# Seleccionar / ver determinadas columnas de un data frame segun su posicion: (ej. ultimas 10 columnas)
df.iloc[:,-10:]

# Unir elementos/string/cadenas de una lista (o cualquier otro string/cadena) con un string:
" ".join(item for item in lista)

# Seleccionar / ver determinadas columnas de un data frame según su nombre:
db[['var1', 'var2']]

# Seleccionar filas de acuerdo al valor de una columa
db.loc[db['variable']==valor]

# Seleccionar filas de acuerdo a varios valores de una misma columna:
db[ db['variable'].isin(['valor1', 'valor2', 'valor3']) ]

#Abrir un CSV como dataframe con Pandas
import pandas as pd
df = pd.read_csv('archivo.csv')

#Cambiar un número de entero a no entero:
variable = float(variableentera)

#Convertir de no entero a entero
variable = int(variablefloat)

#Floored division: hace una división y te devuelve un núm entero si los dos son enteros y sino n float:
5//2

#OJO CON LA TOLERANCIA. A veces le preguntas si x==2.5 pero en realidad es 2.50000, o una movida así con el tema de los floats y pedirle enteros

#Hacer una lista:
nombrelista = [ ]
#(si le pones parentesis es un tupple, es inmutable)

#Llamar a un elemento de la lista (ojo que empieza desde la posición cero)
print(nombrelista[0])
#Para pedir el último valor ponés -1

#Llamar un subgrupo de la lista:
print(nombrelista[2:4])
#Ojo que el último (posición 4) no lo agarra

#Agregar un elemento a la lista: 
nombrelista.append(29)

# Loopear según el nombre de las variables (ej con un sufijo de tiempo) (aca las puse en una lista:
var1=1
var2=13
var3=-5
lista = []
for i in range(1,4):
    lista.append( eval("var"+str(i)) )
lista

#Un loop: agarrar elemento por elemento y mostrarlo:
for x in mylist:
    print(x)

#Un loop: agregar elementos a la lista, acá agrega el 4 y el 5
a = [1,2,3]
a += [4,5] 
#(o sin corchetes)

# Loopear por numeros:
for i in range(2,10,2): #inicio, fin, salto
    ...

#Quedarte con los strings de una lista:
    
# Imaginemos que tenemos una lista de nombres no ordenados que de alguna manera se incluyeron algunos números al azar.
# Para este ejercicio, queremos imprimir la lista alfabética de nombres sin los números.
# Esta no es la mejor manera de hacer el ejercicio, pero ilustrará un montón de técnicas.
names = ["John", 3234, 2342, 3323, "Eric", 234, "Jessica", 734978234, "Lois", 2384]
print("Number of names in list: {}".format(len(names)))
# Primero eliminamos esos números
new_names = []
for n in names:
    if isinstance(n, str):
        # Si n es string, agregar a la lista
        # Notar la doble sangría
        new_names.append(n)

#Eliminar un (dos) elemento de la lista:
lista[0:2]=[ ]



#######################################
############# CURSO UNSAM #############
#######################################

#OPERACIONES BÁSICAS:

#La potencia es con **

#Sumar y asignar: += . Lo que hace es sumarle tres a la variable y sobreescribirla con el resultado.



#Definir una funcion:
def say_hello()
	print(‘Hello, World’)
#ejemplo:
def calificar(sujeto, adjetivo):
        print("{} es {}".format(sujeto, adjetivo))
calificar ("Juan", "capo")

#Testear si funciona la función:
print(callable(say_hello) 	y tendría que salir true


return 
#es como una función que devuelve un valor. Este valor a menudo no es visto por el usuario humano, pero puede ser usado por la computadora en otras funciones.
#Ej:
def add_three(num):
    return num + 3

 
#%%CURSO UNSAM
#Usá el guión bajo (underscore, _) para referirte al resultado del último cálculo.

#Ejecutar en una terminal de Windows:
C:\SomeFolder>hello.py
hello world

C:\SomeFolder>c:\python36\python hello.py
hello world

#A veces es conveniente especificar un bloque de código que no haga nada. El comando pass se usa para eso.
if a > b:
    pass
else:
    print('No ganó a')

x + y      #Suma
x - y      #Resta
x * y      #Multiplicación
x / y      #División (da un float, no un int)
x // y     #División entera (da un int)
x % y      #Módulo (resto)
x ** y     #Potencia
abs(x)     #Valor absoluto


x << n     #Desplazamiento de los bits a la izquierda
x >> n     #Desplazamiento de los bits a la derecha
x & y      #AND bit a bit.
x | y      #OR bit a bit.
x ^ y      #XOR bit a bit.
~x         #NOT bit a bit.

import math
a = math.sqrt(x)
b = math.sin(x)
c = math.cos(x)
d = math.tan(x)
e = math.log(x)

x < y      #Menor que
x <= y     #Menor o igual que
x > y      #Mayor que
x >= y     #Mayor o igual que
x == y     #Igual a
x != y     #No igual a

#Con esto en mente, ¿podrías explicar el siguiente comportamiento?
>>> bool("False")
True
>>>

#Normalmente las cadenas de caracteres solo ocupan una linea. Las comillas triples nos permiten capturar todo el texto encerrado a lo largo de múltiples lineas:
# Comillas triples
c = '''
Yo no tengo en el amor
Quien me venga con querellas;
Como esas aves tan bellas
Que saltan de rama en rama
Yo hago en el trébol mi cama
Y me cubren las estrellas.
'''

#Código de escape
#Los códigos de escape (escape codes) son expresiones que comienzan con una barra invertida, \ y se usan para representar caracteres que no pueden ser fácilmente tipeados directamente con el teclado. Estos son algunos códigos de escape usuales:
'\n'      #Avanzar una línea
'\r'      #Retorno de carro El retorno de carro (código '\r') mueve el cursor al comienzo de la línea pero sin avanzar una línea. El origen de su nombre está relacionado con las máquinas de escribir.
'\t'      #Tabulador
'\''      #Comilla literal
'\"'      #Comilla doble literal
'\\'      #Barra invertida literal

#Indexación de cadenas
#Las cadenas funcionan como los vectores multidimensionales en matemática, permitiendo el acceso a los caracteres individuales. El índice comienza a contar en cero. Los índices negativos se usan para especificar una posición respecto al final de la cadena.
a = 'Hello world'
b = a[0]          # 'H'
c = a[4]          # 'o'
d = a[-1]         # 'd' (fin de cadena)
También se puede rebanar (slice) o seleccionar subcadenas especificando un range de índices con :.
d = a[:5]     # 'Hello'
e = a[6:]     # 'world'
f = a[3:8]    # 'lo wo'
g = a[-5:]    # 'world'

Operaciones con cadenas
Concatenación, longitud, pertenencia y replicación.
# Concatenación (+)
a = 'Hello' + 'World'   # 'HelloWorld'
b = 'Say ' + a          # 'Say HelloWorld'

# Longitud (len)
s = 'Hello'
len(s)                  # 5

# Test de pertenencia (in, not in)
t = 'e' in s            # True
f = 'x' in s            # False
g = 'hi' not in s       # True

# Replicación (s * n)
rep = s * 5             # 'HelloHelloHelloHelloHello'

#Métodos de las cadenas
#Las cadenas en Python tienen métodos que realizan diversas operaciones con este tipo de datos.
#Ejemplo: sacar (strip) los espacios en blanco sobrantes al inicio o al final de una cadena.
s = '  Hello '
t = s.strip()     # 'Hello'
#Ejemplo: Conversión entre mayúsculas y minúsculas.
s = 'Hello'
l = s.lower()     # 'hello'
u = s.upper()     # 'HELLO'
#Ejemplo: Reemplazo de texto.
s = 'Hello world'
t = s.replace('Hello' , 'Hallo')   # 'Hallo world'

#Más métodos de cadenas:
#Los strings (cadenas) ofrecen una amplia variedad de métodos para testear y manipular textos. Estos son algunos de los métodos:
s.endswith(suffix)     # Verifica si termina con el sufijo
s.find(t)              # Primera aparición de t en s (o -1 si no está)
s.index(t)             # Primera aparición de t en s (error si no está)
s.isalpha()            # Verifica si los caracteres son alfabéticos
s.isdigit()            # Verifica si los caracteres son numéricos
s.islower()            # Verifica si los caracteres son minúsculas
s.isupper()            # Verifica si los caracteres son mayúsculas
s.join(slist)          # Une una lista de cadenas usando s como delimitador
s.lower()              # Convertir a minúsculas
s.replace(old,new)     # Reemplaza texto
s.split([delim])       # Parte la cadena en subcadenas
s.startswith(prefix)   # Verifica si comienza con un sufijo
s.strip()              # Elimina espacios en blanco al inicio o al final
s.upper()              # Convierte a mayúsculas

#Los strings son "inmutables" o de sólo lectura. Una vez creados, su valor no puede ser cambiado. Esto implica que las operaciones y métodos que manipulan cadenas deben crear nuevas cadenas para almacenar su resultado.

#Ejercicio 1.16: Testeo de pertenencia (test de subcadena)¶
#Experimentá con el operador in para buscar subcadenas. En el intérprete interactivo probá estas operaciones:
>>> 'Naranja' in frutas
?
>>> 'nana' in frutas
True
>>> 'Lima' in frutas
?
>>>
#Ejercicio 1.21: Expresiones regulares
#Una limitación de las operaciones básicas de cadenas es que no ofrecen ningún tipo de transformación usando patrones más sofisticados. Para eso vas a tener que usar el módulo re de Python y aprender a usar expresiones regulares. El manejo de estas expresiones es un tema en sí mismo. A continuación presentamos un corto ejemplo:
>>> texto = 'Hoy es 6/8/2020. Mañana será 7/8/2020.'
>>> # Encontrar las apariciones de una fecha en el texto
>>> import re
>>> re.findall(r'\d+/\d+/\d+', texto)
['6/8/2020', '7/8/2020']
>>> # Reemplazá esas apariciones, cambiando el formato
>>> re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\2-\1', texto)
'Hoy es 2020-8-6. Mañana será 2020-8-7.'
>>>
#Para más información sobre el módulo re, mirá la documentación oficial en inglés o algún tutorial en castellano. Es un tema que escapa al contenido del curso pero te recomendamos que mires en detalle en algún momento. Aunque no justo ahora. Sigamos...
#Comentario
#A medida que empezás a usar Python es usual que quieras saber qué otras operaciones admiten los objetos con los que estás trabajando. Por ejemplo. ¿cómo podés averiguar qué operaciones se pueden hacer con una cadena?
#Dependiendo de tu entorno de Python, podrás ver una lista de métodos disponibles apretando la tecla tab. Por ejemplo, intentá esto:
>>> s = 'hello world'
>>> s.<tecla tab>
>>>
#Si al presionar tab no pasa nada, podés volver al viejo uso de la función dir(). Por ejemplo:
>>> s = 'hello'
>>> dir(s)
['__add__', '__class__', '__contains__', ..., 'find', 'format',
'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
'title', 'translate', 'upper', 'zfill']
>>>
#dir() produce una lista con todas las operaciones que pueden aparecer luego del parámetro que le pasaste, en este caso s. También podés usar el comando help() para obtener más información sobre una operación específica:
>>> help(s.upper)
#Help on built-in function upper:

#upper(...)
    S.upper() -> string

    #Return a copy of the string S converted to uppercase.


#Los elementos de una cadena pueden ser separados en una lista usando el método split():
line = 'Pera,100,490.10'
row = line.split(',') #la coma indica el elemento que separa
row
['Pera', '100', '490.10']

#Para encontrar rápidamente la posición de un elemento en una lista, usá index().
nombres = ['Rosita','Manuel','Luciana']
nombres.index('Luciana')   # 2
#Si el elemento está presente en más de una posición, index() te va a devolver el índice de la primera aparición. Si el elemento no está en la lista se va a generar una excepción de tipo ValueError.
#rdenar una lista
#Las listas pueden ser ordenadas "in-place", es decir, sin usar nuevas variables.
s = [10, 1, 7, 3]
s.sort()                    # [1, 3, 7, 10]

# Orden inverso
s = [10, 1, 7, 3]
s.sort(reverse=True)        # [10, 7, 3, 1]

# Funciona con cualquier tipo de datos que tengan orden
s = ['foo', 'bar', 'spam']
s.sort()                    # ['bar', 'foo', 'spam']
#Usá sorted() si querés generar una nueva lista ordenada en lugar de ordenar la misma:
t = sorted(s)               # s queda igual, t guarda los valores ordenados

#Podés acceder a los elementos de las listas anidadas usando múltiples operaciones de acceso por índice.
>>> items[0]
'spam'
>>> items[0][0]
's'
>>> items[1]
['Banana', 'Mango', 'Frambuesa', 'Pera', 'Granada', 'Manzana', 'Lima']
>>> items[1][1]
'Mango'
>>> items[1][1][2]
'n'
>>> items[2]
[101, 102, 103]
>>> items[2][1]
102
>>>
#MANERA DE VER LO QUE ESTÁS ITERANDO (con un ejemplo de la clase):
for i,c in enumerate(cadena):
        capadepenapa=capadepenapa+c
        if c in ("aeiou"):
            capadepenapa=capadepenapa+"p"+c #es lo mismo que poner capadepenapa += "p"+c
        print(i,c,capadepenapa)
print(capadepenapa)


#PARA HACER UN BLOQUE/SECCIÓN:
####   #%% Sección 1

Cómo chequear la versión de Python:
import sys
print(sys.version)


# Cuantiles ponderados
 
def comando_cuantiles(df, variable, cuantiles, ponderador=None):

    import matplotlib.pyplot as plt
    #!pip install weightedcalcs
    import weightedcalcs as wc

    if ponderador!=None:
        calc = wc.Calculator(ponderador)
        percentiles = []
        #Computo los percentiles
        for x in range(1,cuantiles+1) :
            p = calc.quantile(df[df[variable]>0], variable,x/100)
            percentiles = percentiles + [p]

        data=df[df[variable]>0]
        lista_df = []
        link = []
        
        for index, row in data.iterrows():
            t = False
            per=0
            for i in percentiles:
                if t==False:   
                    if row[variable]>=i:
                        t=False
                    else:
                        t=True
                    per += 1
            lista_df = lista_df  + [per]
            link = link + [row['link']]
        dict_df = {'link':link,'percentil':lista_df}
        out = pd.DataFrame.from_dict(dict_df)
        out.percentil = out.percentil.astype(int)

        return out
    # else:
    #     df.quantile(q=cuantiles)
        
#     bar_df = gdf.groupby('percentil').agg({ponderador:'sum'})
#     bar_df = bar_df.reset_index()
#     plt.bar(bar_df['percentil'], bar_df[ponderador])


###################################
###### DATOS ESPACIALES ###########
###################################

import geopandas as gpd

# Leer un archivo shape / geoespacial
mapa = gpd.read_file('path')

# Graficar el mapa
mapa.plot

# Explorar el mapa / graficar el mapa sobre un mapa real:
mapa.explore()

# Generar una matriz de distancias entre los puntos de un geo data frame:
matriz_distancias = mapa.geometry.apply(lambda g: mapa.distance(g))

# Covertir un data frame en geo data frame
db_gdf = gpd.GeoDataFrame(data=db, geometry=gpd.points_from_xy(db.longitud, db.latitud), crs='epsg:4326')

# Uniones espaciales / Seleccionar puntos dentro de un poligono
db_gdf = gpd.sjoin(db_gdf, polydf, op = 'within')

# Cambiar el CRS de un geo data frame
db_gdf = db_gdf.to_crs('epsg:4326') # creo que hay otro que se llama set_crs # y otra es db_gdf.crs = {'init': 'epsg:4326'}

# Crear un shapefile / convertir un geo data frame a shapefile
gdf.to_file("algo.shp")

# Diferencia entre estos dos CRSs 3857 y 4326: https://gist.github.com/Rub21/49ed3e8fea3ae5527ea913bf80fbb8d7

# Graficar dos capas juntas
base_plot = poligono.plot()
puntos.plot(ax=base_plot, color='blue');

# Ver ipynb "arreglo_espacioal_estaciones_servicio"



# %% BASIC COMMANDS

# Look for the position of an element in a list/dictionary/etc according to its value
x=["a","b"]
x.index("b")

# Create a list of numbers from 1 to n
list(range(1,5))

# Create a dictionary from two lists
list_a=["a", "b", "c"]
list_b=[1,2,3]
dict1 = dict(zip(list_a, list_b))

# %% DICTIONARIES
a= {"saludo":"hola"}

# Get keys
a.keys()

# Add/change a key-value pair
a["saludar"] = "holar"

# Delete key-value pairs
del(a["saludar"])

# Check if value is part of keys
"saludar" in a

# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
    # Use chained square brackets
print(europe["france"]["capital"])

# %% LOOPS

# for lop

# pair index with value
fam =[1,3,5,1]
for index, value in enumerate(fam):
    print("index " + str(index) + ": " + str(value))
    
# Iterate through key-value pairs in a dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
for country, capital in europe.items():
    print(country, " ", capital)
    
# Iterate over elements of a 2D numpy array
for x in np.nditer(my_array) :
    ...

# Continue code in the next line
df.groupby("col") \
    .mean()

# %% NUMPY - ARRAYS

# Logical equivalents of AND and OR and NOT
np.logical_and()
np.logical_or()
np.logical_not()

# %% NUMPY - RANDOM NUMBERS
from numpy import random

# Set seed
random.seed(123)

# %% PANDAS - DATA FRAMES
import pandas as pd

# Build a DataFrame from a dictionary of lists (column by column)
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)

# Build a DataFrame from a list of dictionaries (row by row)
list_of_dicts[{"country":"United States", 'drives_right':"True", 'cars_per_cap':809},
{"country":"Australia", 'drives_right':"False", 'cars_per_cap':731}]
cars=pd.DataFrame(list_of_dicts)

# Access/change row index
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels

# Read a csv file
pd.read_csv("file.path") # index_col to say which column corresponds to row index

# Write a csv file / convert data frame to csv
df.to_csv("file.csv")

# Select a column of a data frame (several ways)
cars["cars_per_cap"] # returns a pandas series
type(cars["cars_per_cap"])

cars[["cars_per_cap"]] # returns a pandas (sub) data frame
type(cars[["cars_per_cap"]])

# Select multiple columns of data frame
df[["col1, col2"]]
#or
cols = ["col1, col2"]
df[cols]

# Select rows by index
cars[0:5]

# Slice by rows/columns

# Select rows by row names
cars.loc["US"]
cars.loc[["US", "RU"]]

# Select rows by row names and column names
cars.loc[["RU"], ["country"]]

# Select all rows from columns
cars.loc[:, ["country", "drives_right"]]

# Subset data frame based on positions of rows/columns
cars.iloc[[1, 4]]
cars.iloc[[1, 4], [0, 1]]
cars.iloc[:, [0, 1]]

# Iterate over column names of data frame
for col in cars:
    print(col)

# Iterate over rows of data frame (row labes and row data)
for lab, row in cars.iterrows():
    print(lab)
    print(row)
    
# Create new columns by using apply on a function
    # eg the length of the name of the country
cars["length_country_name"] = cars["country"].apply(len)
cars

# Display column names, data types and missing values
df.info()

# Data frame dimension attribute
df.shape

# Overview of descriptive statistics of data frame
df.describe()

# Values of data frame (attribute) / get data frame values as array
df.values

# Columns of data frame (attribute)
df.columns

# Row names / row indexes (attribute)
df.index

# Sort rows of data frame
df.sort_values("column_name")
df.sort_values("column_name", ascending=False)
# Sort data frame accoridn to multiple columns
df.sort_values(["column_name", "col_name2"], ascending=[True, False])

# Subsetting according to logicla condition
df[df["col"]>10]
df[df["col"]=="value"]
df[(df["col"]>10) & (df["col"]=="value")] # !!! add parenthesis

# Subsetting using .isin()
df[df["col"].isin(["v1", "v2"])]

# Create a new column
df["col"] = df["col2"]*2

## SUMMARY STATISTICS
df["col"].mean() # mean of column
df["col"].median() # median of column
df["col"].mode() # mode of column
df["col"].min() # minimum of column
df["col"].max() # maximum of column
df["col"].var() # variance of column
df["col"].std() # standard deviation of column
df["col"].sum() # sum of column
df["col"].quantile() # quantile of column
df["col"].cumsum() # cumulative sum of column
# NOTE: this are all calculated based on the "index" axis of the data frame, which means "by rows", because of the default value
df.mean(axis="index")
# But you can also calculate summary statistics for each row (across columns)
df.mean(axis="columns")
# If you dont specify any column, the operation is computed over all columns
df.mean()

# Custom summary statistics using ".agg()" method (example: get 30th percentile)
def pct30(column):
    return column.quantile(0.3)
df["col"].agg(pct30) # there can be more than one function as arguments
df["col"].agg([pct30, median])

## COUNTING
# Count unique values of a column
df["col"].value_counts()
# count unique values and sort by frequency in descending order
df["col"].value_counts(sort=True)
# turn the counts into proportions of the total
df["col"].value_counts(normalize=True)

# Drop duplicates
df.drop_duplicates(subset=["column1", "column2"])

# Group by column
df.groupby(["col1", "col3"])["col2", "col4"].agg([np.min, np.max, np.mean(), np.median()])
df.groupby("col").agg({'col2':'count'})
    # Group by index (when you have a multiindex)
df.groupby(level=0).agg({'col2':'count'}) # first index

# Pivot tables
df.pivot_table(values="col1", index="col_group") # mean by default
df.pivot_table(values="col1", index="col_group", aggfunc=[np.mean, np.median])
df.pivot_table(values="col1", index="col_group1", columns="col_group2") # group by two columns
df.pivot_table(values="col1", index="col_group1", columns="col_group2", fill_value=0) # group by two columns, fill missing values
df.pivot_table(values="col1", index="col_group1", columns="col_group2", fill_value=0, margins=True) # group by two columns, add row and column totals

## ROW INDEXES

# Set column as row INDEX
df = df.set_index("col")
# Reset data frame index
df.reset_index() # turns index as column
df.reset_index(drop=True) # discard index column
# Row indexes are useful because they allow you to subset much more easily. Instead of writing
df[df["col"].isin(["val1", "val2"])]
# you can write
df.loc[["val1", "val2"]]
# You can use multiple columns as index / multi-level/hierarchichal index
df = df.set_index(["col1", "col2"])
# Subsetting is done slightly differently
df.loc[[("val1_col1", "val1_col2"), ("val2_col1", "val2_col2")]]
# Sort by index
df.sort_index(level=["ind1", "ind2"], ascending=True)

# Slicing a data frame by index
    # first sort index
df.loc["val0":"val11"] # last value is included (this is different from list subsetting)
    # with multiple indexes
df.loc[("val1_ind1","val1_ind2"):("val2_ind1","val11_ind2")]
# This is particularly useful when indexes are dates. For example, you can pass a year as an argument without specifying month or day
df.loc["2013":"2015"]
df.loc["2013-01":"2015-12"]

# Slicing a data frame by column/s
df.loc[:, "col1":"col3"]

# Subsetting a data frame by row/column number / position
df.iloc[1:3, 6:8]

# Access the components of a date
df["col1"].dt.month
df["col1"].dt.year
# For example, if you have a column with a date, you can create a new column with the year
df["year"] = df["date"].dt.year

# Get all missing values of df == True
df.isna()

# Check if there are any missing values in each column
df.isna().any()

# Count missing values of columns
df.isna().sum()
df.isna().sum().plot(kind="bar") # plot nº of NaNs in a bar chart

# Remove rows with missing values
df.dropna()

# Replace missing values with another value
df.fillna("MISSING")


# %% JOINING DATA - MUTATING JOINS - PANDAS

## INNER JOIN: return rows with matching values in both tables
df.merge(df2, on="col", suffixes=("_df1", "_df2"))

## LEFT JOIN
df.merge(df2, on='col', how='left') # default is 'inner'

## RIGHT JOIN
df.merge(df2, on='col', how='right') # default is 'inner'

## OUTER JOIN
df.merge(df2, on='col', how='outer') # default is 'inner'
# it can be used to find rows that do not have a match

# Merge according to columns with differnt names
df.merge(df2, left_on='col1', right_on='col1_right')

# Add a column that specifies the result of the merge for each row
df1.merge(df2, on="id", indicator=True)

## MERGE A TABLE TO ITSELF: you can do this with any type of merge

# Merge on data frame indexes> the sintax is the same as before (on='id') except when keys do not have the same name, where you have to add
df.merge(df2, on='id', left_on='col_df', right_on='right_col', left_index=True, right_index=True) # you need to set this last two to True

# Read csv and set column as index
pd.read_csv('file.csv',  index_col='col')

# Verifying merges
df.merge(df2, on='id', validate='one_to_one') # default is "none"
df.merge(df2, on='id', validate='one_to_many')
df.merge(df2, on='id', validate='many_to_many')
df.merge(df2, on='id', validate='many_to_one')

# Merge ORDERED data or TIME SERIES data

# MERGE ORDERED
pd.merge_ordered(df1, df1, on='date') # default of how argument is 'outer'
# You can INTERPOLATE missing data 
pd.merge_ordered(df1, df1, on='date', fill_method='ffill') # forward fill: with the last value
# When using merge_ordered() to merge on multiple columns, the order is important when you combine it with the forward fill feature. The function sorts the merge on columns in the order provided.

# MERGE AS OF (also very useful for TIME SERIES data)
# similar to a left merge_ordered, but matches on the nearest key column and not on exact matches
pd.merge_asof(df1, df2, on='date') # default 'direction' argument is 'backwards': assigns the last row where the key column value in the right table is less than or equal to the key column value in the left table
pd.merge_asof(df1, df2, on='date', direction='forward') # assings the last row in the right table where the key is equal or greater than the one in the left
pd.merge_asof(df1, df2, on='date', direction='nearest')


# %% JOINING DATA - FILTERING JOINS - PANDAS

# Filter observations from one table based on whether or not they match an observation in another table

# SEMI JOIN:
# - Returns observations in the left table that have a match in the right table.
# Only the columns from the left table are shown. 
# No duplicate rows are returned, even if there is a one-to-many relationship
df_3 = df1.merge(df2, on="id") # first do an inner join
df1["id"].isin(df_3["id"])

# ANTI JOIN:
# - Returns observations in the left table that DO NOT have a match in the right table.
# - Only the columns from the left table are shown. 
df_3 = df1.merge(df2, on="id", how="left", indicator=True) # first do an inner join
df_3.loc[df_3["_merge"] == 'left_only', 'gid']


# %% JOINING DATA - CONCATENATION - PANDAS

# Vertical bind / row bind (default)
pd.concat([df1, df2, df3])
    # You can ignore the index
pd.concat([df1, df2, df3], ignore_index=True)
    # Add keys labels to identify which row came from which data frame
pd.concat([df1, df2, df3], ignore_index=False, keys=["1", "2", "3"])
    # You can bind two tables where one has more rows than the other. The method will keep all columns. You can sort columns alphabetically
pd.concat([df1, df2, df3], sort=True)
    # Only keep matching columns
pd.concat([df1, df2, df3], join='inner') # default is 'outer'

# Append method on data frames: simplified version of concat
df.append([df2, df3], ignore_index=True, sort=True)

# Horizontal bind / row bind (default)
pd.concat([df1, df2, df3])

# Verifying concatenations
df.concat(verify_integrity=True) # deafult is False. True verifies if there are duplicated indexes

# %% SELECTING DATA - QUERY 
# Similar to the WHERE clause of a SQL statement
df.query('col > 10') # returns all rows where col is grater than 10
df.query('col > 10 and col < 20')
df.query('col > 10 or col2 == "value"') # use double quotes inside the statement

# %% RESHAPING DATA - melt
# Reshape data from wide to long format
df.melt(id_vars=['col1', 'col2'])
# chose which variables will remain unpivoted
df.melt(id_vars=['col1', 'col2'], value_vars=['2019', '2020'])
# Set names for the new variable column and the values column
df.melt(id_vars=['col1', 'col2'], var_name='years', value_name='dollars')


# %% GRAPHS - MATPLOTLIB

import matplotlib.pyplot as plt

x=[1,2,3,4]

y=[4,65,12,4]

z=[3,3,2,1]


# To visualize graph, show it
plt.show()

# To close visualization
plt.clf()

# Line plot
plt.plot(x, y, kind="line") # here "line" is default, so not necessary

# Scater plot
plt.scatter(x, y) # or df.plot(x,y,kind="scatter")
    # Set dot size according to a variable
plt.scatter(x, y, s=z)

# Histogram
plt.hist(x)
# control bin number
plt.hist(bins=29)

# Title
plt.title("Title")

# X and Y labels
plt.xlabel()
plt.ylabel()

# Change to logarithmic scale
plt.scale("log")

# Bar plot
plt.plot(kind="bar")
df.plot(kind="bar")

# Place two or more graphs in the same plot
df.plt.hist(alpha=0.7)
df2.plt.hist(alpha=0.7)
plt.show()

# Plot multiplie histogramas at a time, but not on the same plot
df.[["col1", "col2"]].hist()

# Add legend to plot
plt.legen(["A", "B"])