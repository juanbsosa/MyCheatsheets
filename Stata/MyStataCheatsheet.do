* Eliminar la primera fila
drop if _n == 1

* Eliminar la ultima fila
drop if _n==_N

* Exportar a un archivo csv
export delimited using "C:\archivo.csv", replace

* Extraer parte de un string de acuerdo a la posicion de un caracter
gen var2 = substr(var1, strpos(var1, "caracter"), .)

* Extraer la primera palabra de un string o la ultima
egen first = ends(calle), head
egen last = ends(name), tail

* Timer / Calcular el tiempo de procesamiento (ver mejor):
timer clear
timer on
timer off 1
timer list 1
local tiempo=r(t1)

* Bajar datos de INDEC:
cap copy http://www.indec.gov.ar/nuevaweb/cuadros/4/t213_dta.zip t213_dta.zip, replace
cap unzipfile t213_dta.zip

* Mostrar todas las varibles con sus respectivos tipos

foreach var of varlist _all {
	display " `var' "  _col(20) "`: type `var''"
}

* Ordenar en orden descendiente
gsort -variable

* Que te aparezca el error justo abajo de la linea que tiene el error:
set trace on

* Ver cantidad de missings de una variable en un tabulate:
tab var, m

* Desactivar la abreviacion de variables.
*Esta activo por default, y te toma, por ejemplo "varia" como abreviacion de "varbiable" sin que vos se lo pidas, lo mismo si pones "vari", hasta que encuentre una ambiguedad
set varabbrev off

* Instalar un paquete
ssc install nombredelpaquete

* Crear variables con varios criterios (no lo probé quizá está mal la sintaxis)
gen aux = .
replace aux if inlist(var, 1, 3, 6)

* Ver el paquete gtools

*MACHETE STATA
*Abrir un base de datos:
use "elpath pero con esta barra /"

*Ver los datos (pero no editarlos)
browse

*Ver los datos y editarlos
edit

*Nombrar el directorio para no tener que copiarlo entero si hay que llamarlo de nuevo (MUY ÚTIL, sobre todo si hacés un tp grupal)
global DATA = "/Users/magibbons/Google Drive/Seminario de Tesis/clase 8/data" 
cd "$DATA"

*Borrar todo
clear

*Salir de STATA
exit

*Fijar la ruta de trabajo/ el directorio:
cd " "

*Ver qué archivos hay en la carpeta
dir

*Ver las frecuencias de una variable
tabulate variable

*Crear variables
generate (o creo que gen sirve, después hay que ver cómo crear los diferentes tipos de variables)

*Bajar un excel
import excel

*GRAAAANN COMMAANDOO: /*TE TIRA LO QUE STATA TIENE GUARDADO DESPUES DE QUE HACES UN COMANDO*/
ereturn list



**********************
** BUENAS PRÁCTICAS **
**********************

*Nunca dropear datos de las encuestas, sino generar dummies para identificar y correr los cálculos con un if para esa dummy

*Para excluir missings poner "<." en vez de "==1" porque Stata tiene varios tipos de missings. "." es igual a infinito.

* Cortar los comandos con un string que no sea comando como "--" o "stop"

*Al loopear por variables ya definidas, usar "foreach x of var1 var var3" en vez de "for x in var1 var2 var3" asi si te equivocas en el nombre de la variable te tira ese error y no que no pudo correrlo por otra cosa.


************************
** KEYBOARD SHORTCUTS **
************************

* Historial de comandos: Ctrl+3