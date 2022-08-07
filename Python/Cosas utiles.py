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
db['var'] = np.where(db['var2']=='valor', valorsitrue, valorsifal
    se)

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