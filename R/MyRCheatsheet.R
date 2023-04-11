#____________________________________________
#              COSAS ÚTILES 
#____________________________________________


# BUENAS PRACTICAS --------------------------------------------------------

# No usar "df" como el nombre de un data frame (sobre todo en una funcion), porque es un nombre definido de R
#https://stackoverflow.com/questions/69849609/error-argument-df1-is-missing-with-no-default

# COMBINACIONES DE TECLADO / KEYBOARD SHORTCUTS ---------------------------

### !!! OJO. Cada usuario puede cambiar esto a su gusto

# Posicionar dos cursores, uno atras y otro adelante de un texto seleccionado:
CTRL + ALT + A

# Abrir el "document outline" que es como un resumen de las secciones de tu codigo
Ctr + Shift + O

# Colapsar todos los "folds" (son los pedazos de codigo que se agrupan con {} )
Alt + O
# Volver a expandirlos
Alt + Shift + O

# Cambiar la direccion de las barras / en una linea de codigo (lo agregué yo, es parte de un addin)
Ctrl + Shift + F

# Activar que cuando tocás TAB se inserte una indentación (a veces se desactiva) / "Toggle Tab Key Moves Focus" / Insert spaces with TAB:
Alt + Shift + [

# Sourcear un archivo (correr el archivo como si fuese en la consola)
ctrl + shift + s

# Correr el script entero
ctrl + alt + r

# Correr el script desde el inicio hasta la linea
Ctrl + Alt + B

# Hacer la flecha para asignar <-:
alt + -

# Ejecutar la seccion actual / run code section
ctrl + alt + t

# Indentear codigo
ctrl + }

# Unindentear codigo
ctrl + {
    
#__________________________________________________________________________



# PAQUETES ÚTILES ---------------------------------------------------------

# ]Momentos estadísticos
library(moments)
  
## ARGENTINA (https://github.com/PoliticaArgentina/polArverse)
# Datos espaciales Argentina
library(geoAr)
# Datos legislatura Argentina
library(lesgislAr)
# Datos legislatura Argentina
library(lesgislAr)
# Datos elecciones Argentina
library(electorAr)

# Conectarse a bases relacionales en SQL
library(RSQLite)
  
# Hacer operaciones con bases en SQL como si fuesen data frames
library(dbplyr)
  
  #_________ Tema: Data Tidying

# Template de directorios y carpetas
library(ProjectTemplate)

# Organizacion de archivos y directorios
library(here)

# Leer CSVs
library(readr)

# Acceder a planillas en Google Sheets
library(googlesheets4){
  gs4_auth() #habilitar API
  gs4_find() # ver planillas disponibles
  read_sheet("2cdw-678dSPLfdID__LIt8eEFZPasdebgIGwHk") #leer / descargar planillas
  survey_sheet <- read_sheet("https://docs.google.com/spreadsheets/d/1FN7VVKzJJyifZFY5POdz_LalGTBYaC4SLB-X9vyDnbY/", sheet = 2) # otra opcion
}

# Leer archivos Excel
library(readxl){
  example <- readxl_example("datasets.xlsx")
  df <- read_excel(example)
  read_excel(example, col_names = LETTERS[1:5])
}
  
# Acceder a archivos en Google Drive
library(googledrive)

# Leer o escribir archivos .dta de Stata, SPSS o SAS
library(haven)

# Leer archivos en formato JSON (APIs) y XML (HTML)
library(jsonlite)
  read_json("json_file.json", simplifyVector = TRUE) # simplifies nested lists into vectors and data frames
  
library(xml2)
  read_xml("xml_file.xml")

# Web scraping
library(rvest){
  #When rvest is given a webpage (URL) as input, an rvest function reads in the HTML code from the webpage.
  # Usar la extension de Chrome "SelectorGadget"
  packages <- xml2::read_html("https://datatrail-jhu.github.io/stable_website/webscrape.html")
  packages %>% 
    html_nodes("strong") %>%
    html_text() 
}
  
# Ver la version actual de R
R.version
    
# Actualizar R
library(installr)
updateR()
    
# Comunicarse con APIs
library(httr){
    # HTTP is based on a number of important verbs : GET(), HEAD(), PATCH(), PUT(), DELETE(), and POST().
    # Blog "Using R to extract data from web APIs": https://www.tylerclavelle.com/code/2017/randapis/
    ## Save GitHub username as variable
    username <- 'janeeverydaydoe'
    ## Save base endpoint as variable
    url_git <- 'https://api.github.com/'
    ## Construct API request
    api_response <- httr::GET(url = paste0(url_git, 'users/', username, '/repos'))
    ## Check Status Code of request (200 is ok)
    api_response$status_code
    names(api_response)
    repo_content <- content(api_response)
    ## function to get name and URL for each repo
    lapply(repo_content, function(x) {
      df <- data_frame(repo = x$name,
                       address = x$html_url)}) %>% 
      bind_rows()
    # Otro ejemplo: ## Make API request
    api_response <- GET(url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/steak-survey/steak-risk-survey.csv")
    ## Extract content from API response
    df_steak <- content(api_response, type="text/csv")
    # Para Twitter:
    myapp = oauth_app("twitter",
                      key = "yourConsumerKeyHere",
                      secret = "yourConsumerSecretHere")
    sig = sign_oauth1.0(myapp,
                        token = "yourTokenHere",
                        token_secret = "yourTokenSecretHere")
    homeTL = GET("https://api.twitter.com/1.1/statuses/home_timeline.json", sig)
  }

# Data wrangling / limpiar bases de datos:
library(dplyr) #(mutate, select, filter, summarize, and arrange)
library(tidyr)
library(janitor)

# Lidiar con variables categoricas (factor variables)
library(forcats)

# Procesar strings / cadenas de texto
library(stringr)

# Procesar fechas y tiempos
library(lubridate)

# Equivalente a f strings en Python
library(glue) # tambien usar sprintf de R base

# Hacer resumenes / summary de los datos, estadisticas descriptivas
library(skimr)

# Analizar textos largos
library(tidytext)

# Trabajar con funciones y vectores
library(purrr)

#_________ Tema: visualización de datos

# Graficos en general
library(ggplot2)
library(ggrepel)
library(cowplot)

# Combinar graficos
library(patchwork)

# Hacer tablas lindas / graficos de tablas
library(kableExtra)

# Crear animaciones
library(gganimate)

# Lidiar con imagenes
library(magick){
  img1 <- image_read("https://ggplot2.tidyverse.org/logo.png")
  print(img1)
  cat(image_ocr(img1))
}


#__________ Tema: Modelos estadísticos

# Inferencia estadistica
library(broom)
library(infer)

# Trabajar con datos de panel
library(tsibble)
library(feasts)
library(fable)

#__________ Tema: Optimizacion de procesamiento

# Paralelizar procesos:
library(parallel)
    

#__________ Tema: paquete googledrive
    
library(googledrive){
  # Listar hasa n_max archivos en mi unidad
  drive_find(n_max = 30)
  
  # Encontrar archivos de acuerdo a un patron o a un tipo de archivo / extension, por visibilida,d si esta en favoritos
  drive_find(pattern = "chicken")
  drive_find(type = "spreadsheet")
  files <- drive_find(q = c("starred = true", "visibility = 'anyoneWithLink'"))
  
  # Bajarse un archivo
  (x <- drive_get("~/abc/def/googledrive-NEWS.md"))
  # let's retrieve same file by id (also a great way to force-refresh metadata)
  drive_get(x$id)
  drive_get(as_id(x))
  
  # Subir un archivo
  drive_upload('C:/Users/juanb/bases_amba_2022_Alquiler.csv', path = 'Data/Meli/0. Raw data/bases_amba_2022_Alquiler.csv', type = "text/csv", overwrite = TRUE) # type hace que no lo convierta automaticamente en un google sheets
  
  # Especificar el tipo de archivo
  chicken_sheet <- drive_example_local("chicken.csv") %>% 
    drive_upload(
      name = "index-chicken-sheet",
      type = "spreadsheet"
    )
  
  # Ver los permisos del archivo y compartir:
  chicken_sheet %>% 
    drive_reveal("permissions")
  (chicken_sheet <- chicken_sheet %>%
      drive_share(role = "reader", type = "anyone"))
  
  # Descargar archivos
  drive_download("index-chicken-sheet", type = "csv")
  
  # Eliminar archivos del Drive
  drive_trash("tidyverse.txt")
  drive_untrash("tidyverse.txt")
  drive_rm("tidyverse.txt")
  
  # Compartir un archivo
  drive_share(file = "tidyverse", 
              role = "commenter", 
              type = "user", 
              emailAddress = "someone@example.com",
              emailMessage = "Would greatly appreciate your feedback.")
  drive_share_anyone(file = "tidyverse", verbose = TRUE) # anyone with link can read
  drive_reveal(file = "tidyverse", what = "permissions") # ver quien puede
}
    

#__________________________________________________________________________


# Addins / Snippets ------------------------------------------------------------------

# Addin para invertir las barras de un directorio, insertar un pipe %>%, y otra cosa más
devtools::install_github("sfr/RStudio-Addin-Snippets", type = "source") # Instalar
    # Reiniciar R
    # Puse el de "flip slash" como keyboard shortcut en CTRL+Shift+F (Tools->Modify Keyboard Shortcuts)
    
# Abrir el archivo con los snippets y agregar snippets o modificar snippets
usethis::edit_rstudio_snippets()



# ERRORES / PROBLEMAS HABITUALES ----------------------------------------------------

# Si estas en browser mode en el debugger y la consola no te printea los outputs, corre
sink()
#https://github.com/rstudio/shiny/issues/509

# Error: "Error in locale(encoding = "UTF-8") : could not find function "locale" "
library(readr) #solo hay que correr eso y se define la variable



# CODIGO ÚTIL / COMANDOS UTILES -------------------------------------------------------------
#(se lee de abajo para arriba)

# Recodear una variable / Mapear los valores de una columna a otros valroes /
# Cambiar los valores de una variabl segun una correspondencia
mapping <- c("III" = "03", "IV" = "04", "V"="05", "VI"="06") # viejo valor=nuevo valor
df <- df %>% mutate(var = recode(var, !!!mapping))

# Leer archivos Excel especificando los tipos de columnas solo para algunas columnas
path <- "file.xlsx"
# Cargo solo una fila para usar nombres de columnas
df <- read_excel(path, sheet = "Sheet1", n_max = 1)
col_names <- names(df)
# Crear un vector con el string "text" para dos cols, y "guess" para el resto
col_types <- rep("guess", length(col_names))
col_types[col_names %in% c("col1", "col2")] <- "text"
# Leer la planilla especificando el tipo de columnas de esas dos columnas
df <- read_excel(path=path, sheet="Sheet1", col_types=col_types)
rm(col_names, col_types)

# Escribir archivos Excel con multiples hojas
sheets <- list("sheet_name_1" = df1, "sheet_name_2" = df2)
writexl::write_xlsx(x=sheets, path=path)

# Eliminar todas las filas de un data frame que tienen missing values en todas
#   las columnas
df <- df[rowSums(is.na(df)) != ncol(df), ]

# Crear un data frame a partir de una lista con nombres
lista <- list(
    variable1 = c(1,345,14,732,13,52,76,12,5667,12),
    variable2 = c("asda", "kopk", "asdas")
)
df <- stack(lista)

# 
url <- "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/comunas/comunas.zip"
download.file(url, destfile = "Comunas-poligonos-shp.zip")
# Extraer numeros de una cadena de texto
db$var <- readr::parse_number(db$var)

# Calcular la tasa de cambio de todas las columnas de un data frame (o la primera diferencia si sacamos la division)
pct_chage <- function(vector){
    
    diff <- diff(vector, lag=1)
    pct_diff <- diff/vector[-length(vector)]
    
    return(pct_diff)
    
}
as.data.frame(lapply(df, pct_chage))

# Transponer un data frame y hacer que la primera columna sea los nombres de las columnas
df_t = setNames(data.frame(t(df[,-1])), df[,1])

# Citar un paquete en formato bibtex (sino citarlo solo con citation("base"))
toBibtex(citation("base")) # ojo que te devuelve el @ con mayusculas, y en latex va con minusculas

# Create a copy of an R object with different memory address/ create a deep copy of an object // Crear una copia de un objeto en R que no este vinculada al objecto original
mtcars1 <- data.table::copy(mtcars)
tracemem(mtcars1) == tracemem(mtcars) # esto es para ver si los objetos tienen la misma direccion en la memoria

# Fast table for summary statistics / Tabla rapida de estadisticas descriptivas
psych::describe(mtcars)

# FUNCIONES: R automaticamente te devuelve el ultimo elemento definidio en una funcion, sin tener que poner return. Estas funciones dan todas lo mismo:
f1 <- function(x){2*x}
a <- f1(3)
a
f2 <- function(x){
    res <- 2*x
    res
    }
a <- f2(3)
a
f3 <- function(x){
    res <- 2*x
}
a <- f3(3)
a



# Hacer que R espere determinado tiempo / que tarde X segundos
Sys.sleep(5)

# Actualizar un paquete
update.packages("splm")

# Calcular los valores propios de una matrix
eigen(matriz)

# Ver la ubicación de los paquetes
.libPaths()

# Obtener mas de un elemento definidio en el ambiente
a <- 1
b <- 2
c <- mget(c("a", "b"))

# Obtener el directrio del script actual abvierto en RStudio
dirname(rstudioapi::getSourceEditorContext()$path)

# Quedarse con los valores numericos de un objeto con nombres
unname(x)

# Evaluar una instruccion que puede dar error sin frenar el procesamiento
a <- NA
try(a <- "f" + 2, silent=T)
a
try(a <- 3 + 2, silent=T)
a

# Ver cuántos núcleos tiene tu computadora (count pc's cores)
parallel::detectCores()

# Usar un string para nombrar una variable en las funciones del paquete dplyr
col <- "var2"
df <- df %>% group_by(var1) %>% mutate(var3 = !!as.symbol(col)) # a veces con !!sym(col) basta

# Identificar columnas con valores negativos (crea un data frame)
sum <- df %>%
        select_if(is.numeric) %>%
        gather(var, val) %>%
        group_by(var) %>%
        summarise(res = any(val[!is.na(val)] < 0)) 

# Agregar un termino a una formula
formula <- y ~ 1 + x*y
formula <- update(formula, ~ . + z)

# Convertir un string en una formula
as.formula("y ~ x1 + x2")

# Hacer un merge entre dos data frames y actualizar una columna en comun entre los dos (la columna no es parte de la llave utilizada para unir los dfs). Es decir, si encuentra columnas duplicadas entre los dfs, en vez de agregar una columna nueva para cada una utilizando un sufijo, que actualice los valores de la columna existente en el df de la izquierda.
df <- powerjoin::power_left_join(df1, df2, by = c("var1", "var2", "var3"), conflict = powerjoin::coalesce_yx)

# Crear un objeto en el ambiente padre / asignar un objeto en el ambiente por fuera de una funcion
assign("le_pongo_este_nombre", este_es_el_objeto, envir=parent.frame())


# Crear un indice por grupo
db <- db %>%
    group_by(vargrupo1) %>%
    mutate(id = cumsum(!duplicated(vargrupo2))) #revisar

# Crear una lista de data frames repetidos / replicar el mismo data frame en una lista n veces
df_list <- rep(list(df), 3)

# Obtener el nombre de un objeto:
deparse(substitute(location))
# ejemplo:
objeto_con_nombre <- cars
nombre <- deparse(substitute(objeto_con_nombre))
print(nombre)

# Eliminar objetos de la memoria que no se estan usando
gc()

# Hacer referencia a una variable con un string (ej. en dplyr):
var <- "cyl"
resultado <- mtcars %>% filter((!!sym(var)) == 4)

# Ver todos los comandos / todas las funciones disponibles en un paquete
require(ggplot2)
ls("package:ggplot2")

# Visualizar una matrix con colores
m <- matrix(rnorm(100), ncol = 10)
image(t(m[nrow(m):1,] ), axes=FALSE, zlim=c(-4,4), col=gray.colors(100))

# Normalizar una variable entre dos valores (a y b) / Re-escalar un vector
#   entre dos valores
# https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
normalize_ab <- function(vec, a, b){
    vec_rescaled <- (b-a) * (( (vec - min(vec)) / (max(vec) - min(vec)) )) + a
    return(vec_rescaled)
}
var <- runif(100, min=-27, max=3)
var_rs <- normalize_ab(var, -1, 1)

# Normalizar una variable entre 0 y un valor / Re-escalar
a <- c(1,4,3,5,67,8123,2342,23465,123,12)
norm_minmax <- function(var, k){(var - min(var)) / (max(var)-min(var)) * k}
norm_minmax(a,10)

# Crear una variable usando un string / crear un variables de un data frame con nombres dinamicos dentro de un loop
for (i in 1:3){
    db[[ glue::glue("ejemplo_{i}")[1] ]] <- 1:nrow(db) # mas facil con paste que con glue, arreglar
}


# Crear una lista de objetos poniendo definiendo los nombres de los objetos de la lista como sus nombres originales
lista <- tibble::lst(db1, db2, vector1)
names(lista) #aca verias los nombres

# Convertir el indice de filas de un data frame a una columna / row index as column
db <- tibble::rownames_to_column(db, var = "nueva_var")


# Cambiar la posicion de una columna en un data frame / cambiar el orden de una columna
db <- db %>% relocate(var2, .before = var1)
db <- db %>% relocate(var2) # mover una columna al principio del data frame, en la primera posicion

# Para cada fila encontrar un string de una columna en el string de otra columna
db$str_match <- FALSE
for (i in 1:nrow(db)){db$neighborhood_match[i] <- stringr::str_detect(db$var_aca_buscar[i], db$var_esto_buscar[i])}# !!! es muy ineficiente, se tiene que poder hacer con otro comando

# Cambiar un string dentro de un string por otro / cambiar un patrón dentro de una cadena de texto por otro patrón
db$var <- stringr::str_replace_all(db$var, "encontrar_esto", "reemplazarlo_por_esto")
# si no le pones "_all" solo cambia la primera ocurrencia

# Sacar las tildes de un string
db$var <- stringi::stri_trans_general(db$var,"Latin-ASCII")

# Convertir un string a mayúsculas
db$var <- toupper(db$var)

# Duplicar filas / observaciones en base a una variable
db <- db %>% tidyr::uncount(variable_de_frecuencia_o_conteo)

# Agregar sufijos a los valores duplicados de una variable
db$var <- make.unique(db$var, sep = '-')
# entonces si tenes que el valor de var es "1234" para dos filas, a la primera la deja igual y a la segunda le pone "1234-1"

# Calcular el largo de un texto / string/ la longitud de un string:
nchar("hola")

# Guardar el resultado / output de la consola en un archivo .txt
sink(file="./output_funcion.txt", append=TRUE) #desde aca se imprime todo
cat("\n") #esto es un espacio/ enter en la consola
print("Hola")
sink() # aca deja de escribir el archivo

# Sacar un paquete de la memoria
detach("package:tidyverse", unload=TRUE)
# hay una funcion mas compleja en: https://stackoverflow.com/questions/6979917/how-to-unload-a-package-without-restarting-r

# Crear un data frame con estadisticas descriptivas de multiples variables
library(tidyverse)
db.sum2 <- db %>% 
  select(var1, var2, var3) %>% 
  map_df(.f = ~ broom::tidy(summary(.x)), .id = "variable")

# Crear cuantiles por grupo
db <- db %>% dplyr::group_by(group_var) %>% mutate(q99_var = quantile(var, probs=0.99)) # aca percentil 99

# Contar filas por grupo
db <- db %>% 
    group_by(var1, var2) %>% 
    mutate(conteo = n()) %>%
    ungroup() #opcional

# Agrupar usando un vector de strings con los nombres de las columnas
cols <- c("cyl", "disp")
mtcars %>% 
    group_by_at(cols) %>% 
    summarise(n = n())

# Sumar por grupo
df %>%
    group_by(col_to_group_by) %>%
    summarise(Freq = sum(col_to_aggregate))

# Extraer los números de un string / texto /cadena:
string <- c("20 years old", "1 years old")
resultado <- as.numeric(gsub("([0-9]+).*$", "\\1", string))
resultado
# otra forma
tidyr::extract_numeric(string)

# Leer un archivo csv tomando solo las filas que cumplan con determinada condicion
# https://stackoverflow.com/questions/23197243/how-to-read-only-lines-that-fulfil-a-condition-from-a-csv-into-r
library(sqldf)
write.csv(iris, "iris.csv", quote = FALSE, row.names = FALSE)
iris2 <- read.csv.sql("iris.csv", 
                      sql = "select * from file where `Sepal.Length` > 5", eol = "\n")
#!!! (esto no lo probé todavía)

# Verificar si un string / un patrón de texto se encuentra dentro de otro string

# Evitar la notación científica
options(scipen=999)

# Apagar y prender los warnings / advertencias
options(warn=-1)
options(warn=0)

# Allow more prints in console
options(max.print=10000000)

# Allow full width of console
options("width"=200)

# OCR / Leer palabras de una imagen
img1 <- magick::image_read("https://ggplot2.tidyverse.org/logo.png")
print(img1)
cat(image_ocr(img1))

# Merge / Mutating joins
inner <- inner_join(artists, albums, by = "ArtistId")
left <- left_join(artists, albums, by = "ArtistId") 
right <- right_join(as_tibble(artists), as_tibble(albums), by = "ArtistId")
full <- full_join(as_tibble(artists), as_tibble(albums), by = "ArtistId")

# Hacer un merge de dos data frames usando keys distintas para cada uno
df_merged <- df1 %>% left_join(df2, by=c('var_en_df1y2', 'var_en_df1'  = 'var_en_df2'))

# Filtering joins: filtrar todas las filas que tengan un elemento en común con otro data frame
#semi_join(x, y) : keeps all observations in x with a match in y.
#anti_join(x, y) : keeps observations in x that do NOT have a match in y.

# Conectarse a bases de datos relacionales en SQL
library(RSQLite)
## Specify driver
sqlite <- dbDriver("SQLite")
## Connect to Database
db <- dbConnect(sqlite, "company.db")
## List tables in database
dbListTables(db)
# Leer tables de una base en SQL
albums <- dbplyr::tbl(db, "albums")

#o ejemplo con MySQL
## This code is an example only
con <- DBI::dbConnect(RMySQL::MySQL(), 
                      host = "database.host.com",
                      user = "janeeverydaydoe",
                      password = rstudioapi::askForPassword("database_password")
)

# Convertir un data frame a JSON
json <- jsonlite::toJSON(mydf)

# Convertir un archivo JSON a un data frame
mydf <- jsonlite::fromJSON(json)

# Leer un archivo .csv
df <- readr::read_csv("sample_data - Sheet1.csv", colnames=FALSE, skip=0, n_max=100)
# Leer una cantidad determinada de filas de un csv: argumento "n_max"

# Leer un archivo .txt o delimitado por algo (se puede usar para csvs)
df_txt <- read_delim("sample_data.txt", delim = "\t", n_max=100) #"\t" es que el delimitador es un TAB
    
# Pipeline para instalar paquetes no instalados y luego cargarlos todos
required_packages <- c("glue", "tictoc")
not_installed_packages <- required_packages[!required_packages %in% installed.packages()]
for(lib in not_inFstalled_packages) install.packages(lib,dependencies=TRUE)
sapply(required_packages,require,character=TRUE)

# Crear una secuencia de fechas con intervalos por dias o meses o anios
meses <- seq(as.Date('2017-01-01'), as.Date('2017-08-01'), by='1 month')

# Convertir una fecha en formato numerico a formato fecha (es importante desde que dia empieza a contar R, usualmente es el 1 de enero del 1970)
as.Date(db$numdate, origin='1970-01-01')

# Crear variables de anio/trimestre/mes/semestre a partir de una fecha
df$year = lubridate::year(df$fecha)
df$semester = lubridate::semester(df$fecha)
df$quarter = lubridate::quarter(df$fecha)
df$month = lubridate::month(df$fecha)

# Reemplazar los valores de una variable en base a una condicion
db$var <- ifelse(db$var=="valor_a_cambiar", "valor_nuevo", db$var)

# Eliminar todos los objetos en el entorno que tengan determinado patron en su nombre     
rm(list = ls(pattern = "patron"))
    
# Eliminar todos los objetos en el entorno que tengan determinado patron en su nombre / eliminar todos los objetos que empiecen con / startswith
rm(list = ls(pattern = "^prefijo"))

# Obtener una lista con todos los objetos del ambiente (que empiecen con ...)
objetos <- mget(ls(pattern = "^prefijo"))
    
# Evaluar un string como un objeto o variable
eval(parse(text="db"))    

# Convertir un data frame en un tibble (poner "vignette("tibble") para ver las diferencias entre  el data frame y el tibble):
as.tibble(trees)

# Crear una carpeta para el proyecto usando un template
ProjectTemplate::create.project(project.name = "data_analysis", template = "minimal")
# Crea los siguientes subdirectorios: un readme con los tipos de archivos del proyecto, una carpeta data con los insumos, munge para los scripts que pre procesan la data, cache para los resultados del pre procesamiento, src para los scripts del analisis de los datos.
# "Lastly, the load.project() function can be used to “setup” your project each time you open it. For example, if you always need to execute some code or load some packages, calling load.project() with the right config settings will allow you to automate this process."
# Se puede crear un template propio

# Fijar el directorio como el path donde esta el archivo Rproj (prohecto de R)
#primero crear el Rproj
here::here()

# Encontrar el directorio donde esta guardado el script donde estas trabajando
dirname(rstudioapi::getSourceEditorContext()$path)

# Referirte a distintas carpetas dentro del directorio y crear un path desde el directorio
here::here('carpeta1', 'carpeta2', 'nombre_archivo')
# ejemplo guardando o lodeando un archivo:
save(df, file = here::here('carpeta1', 'carpeta2', 'nombre_archivo'))

# Eliminar o esconder el mensaje de advertencia warning message en una funcion:
suppressWarnings(funcion)

# Esconder el resultado de impresion de un objeto / change print mode to invisible
invisible(mtcars)

# Invertir una matriz:
M1<-1:4
M1<-matrix(1:4,nrow=2)
M1
solve(M1)

# Agregar un elemento a un vector en una posicion determinada // agregar un elemento a un vector en la primera posicion:
vector <- c('hola','chau')
vector <- append(vector, 'nuevo_elemento', after=0)
print(vector)

# Obtener todos los archivos de un directorio con determinada extension // loopear a lo largo de todos los archivos de un directorio con determinada extension:
files <- list.files(path=directory, pattern="*csv")

# Loopear / iterar a traves de los elementos de un vector
vector <- c('1','15','17')
for (i in 1:length(vector)){
    print(vector[i])
}

# Crear un data frame vacio, solo con columnas: (mejorar esto, es poco practico tener que definir  el tipo de datos para cada columna)
df <- data.frame(col1=integer(),col2=integer())
names(df) <- c('col1', 'col2')

# Convertir un string en formato fecha:
datestring <- c("January 10, 2012 10:40", "December 9, 2011 9:10")
x <- strptime(datestring, "%B %d, %Y %H:%M")
x

# El tiempo actual / hoy:
Sys.time()

# La fecha actual / hoy:
Sys.Date()

# Obtener el año actual / de hoy
format(Sys.Date(), "%Y")

# Convertir un string en fecha:
x<- as.Date("1970-01-01") # las fechas se almacenan como numero de dias desde esa fecha
# Hay dos formatos de fechas: POSIXct y POSIXlt. El segundo guarda mas info como el dia de la semana
x <-as.POSIXlt(x)
# Ver los elementos en esa lista:
names(unclass(x))
# Obtener los minutos de una fecha
t2<-as.POSIXlt(Sys.time())
t2$min # lo mismo con hora, segundos, dias, años etc.
# Extraer el año de una fecha
format(df$fecha, format="%Y")

# Calcular la diferencia entre dos fechas, expresadas en la unidad de tiemp que quieras
difftime(Sys.time(), Sys.time(), units = 'days')

# Hacer un loop que itere a traves de las posiciones de un vector
vector <- c('a','b','c')
for(i in seq_along(vector)){
  print(vector[i])
}

# Loop / iterar sobre las filas y columnas de una matriz:
x <- matrix(1:6,2,3)
for(i in seq_len(nrow(x))){
  for(j in seq_len(ncol(x))){
    print(x[i,j])
  }
}

# Crear una carpeta / directorio:
dir.create("testdir")
# Crear un directorio junto con un subdirectorio
dir.create(file.path("testdir2", "testdir3"), recursive = TRUE)

# Crear un archivo
file.create("archivo.R")

# Verificar si existe un archivo
file.exists("archivo.R")

# Ver informacion de un archivo como el tamanio, la fecha de modificaciones, de creacion etc.
file.info("mytest.R")

# Cambiar el nombre de un archivo
file.rename("from", "to")

# Hacer una copia de un archivo
file.copy("from", "to")

# Eliminar un archivo
file.remove('mytest.R')

# Crear un path / directorio
file.path('C:', 'programfiles')

# Listar todos los elementos en el workspace (los que se crearon)
ls()

# Lsitar todos los elementos en el directorio
list.files()
# o
dir()
# Seleccionar las filas de un dataframe segun multiples condiciones
db[db$var1 > 31 & db$var2 > 90,]
# ojo que aca se queda con las que tienen NAs

# Seleccionar una columna / variable de un data frame
df[['variable']]

# Calcular el promedio de una variable y omitir los missing values
mean(db$var, na.rm = TRUE)

# Calcular el promedio de todas las variables de un dataframe
apply(db, 2, mean)

# Calcular el maximo de una variable y omitir los missing values
max(db[['var1']], na.rm = TRUE)


# Seleccionar las ultimas N filas de un data frame
db[(nrow(db)-N):nrow(db),]
# o
tail(db, n=N)

# Setear / fijar el directorio
setwd("path")

#Eliminar todos los objetos:
rm(list=ls()) #ls() te tira todos los objetos que ten?s en el enivornment

#Ver donde estan los missing values en una base
is.na(base)
which(is.na(base))

#Contar missing values
length(which(is.na(base$variable)))
# o 
sum(is.na(base[['variable']]))

# Crear una lista con las columnas que tienen missing values
list_na <- colnames(df)[apply(df, 2, anyNA)] #el num 2 dice que busque a lo largo de las columnas. si fuese uno busca los missing values por filas, identificado las observaciones. Ac? va identificar las variables con datos faltantes, no las observaciones

# Mostar la cantidad de missing values por columna / variable
sapply(airquality, function(x) sum(is.na(x)))

#Eliminar todas las filas/observaciones con missing values (con la opción de elegir las variables que tienen missing values): (NO ME ANDUVO BIEN LA PRIMERA!!)
base <- na.omit(base, cols="variable1", "variable2")
base <- base %>% drop_na(variable1, variable2)


#Ver la cantidad de missing values de cada variable en un data frame:
for (yy in 1:length(base)){
  xx <- base[yy]
  print(colnames(xx))
  print(length(which(is.na(xx))))
}

#Transformar todos los missing values de una data frame a cero:
base[is.na(base)] <- 0

#Conditional assignment
base$HDI.rank[which(base$Country=='Argentina')]

#Eliminar una fila de la db:
myData <- myData[-c(2, 4, 6), ] #son los numeros de las filas

#Ver el tipo de un objeto o variable:
class(var)

#Ver las clases de toda las variables de la base:
sapply(db, class) #o lapply(db, class) para la version no simplificada

#Convertir una variable con categorias multiples en muchas binarias:
base <- fastDummies::dummy_cols(base, select_columns = "variable")

# Bajar una base de un archivo dta de Stata 
#install.packages("foreign")
library("foreign")
eph=read.dta("Individual_t215.dta")

# Escribir / crear un archivo .dta de Stata:
write.dta(db, "./Bases en dta/db.dta")

# Escribir un archivo .csv:
write_csv(df_csv, path = "my_csv_file.csv", row.names=FALSE) #importante poner row names false porque sino te crea una columna con los indices

#una consola para editar los datos
fix(dataset)

# Tipo de datos de todas las variables
str(data)

#Estad?stica decriptiva. groupby/Opci?n by 
library("psych")
describeBy(Auto$horsepower, cylinders)

#En que posicion esta la variable / ver la posicion de una variable en un data frame
which( colnames(ch_hh)=="BF5_th" )

#Obtener una muestra aleatoria / filas random al azar de una base de datos (aca 50 observaciones):
df <- df[sample(nrow(df), 50), ]

#Chequear cantidad Nans en toda la base:
a<-0
for (i in (1:ncol(ch_hh4))){
  a<-a+sum(is.nan(unlist(ch_hh4[,i])))
}
a

#Reemplazar los NaNs por algo (cero):
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

df[is.nan(df)] <- 0

#Chequear que columnas tienen Nans:
names(which(sapply(df, function(x) any(is.nan(x)))))

#Obtener una muestra aleatoria (de 200 obs aca) de las observaciones de una base
muestra <- base[sample(nrow(base), 200), ]

#Cambiar todos los espacios en blanco (blank cells) a missing values
base[base==""] <- NA

#Exportar tablas a Latex:
xtable(df, caption="", label="")

#Comparar variables de dos bases de datos
#Variables que tienen diferentes:
setdiff(a, b)
#Variables en común:
intersect(a,b)
# Y después podés usarlo para nombrar un objetco que tenga los nombres de las variables en común, y luego con un pipe y select (dplyr) creás una base nueva solo con esas

# Tabular una variable en la consola
data[unique(data$x),] # no funciona, arreglar

#Tabular una variable, creando un nuevo data frame con una columna para el porcentaje y otra para el porcentaje acumulado (dplyr):
tabulado <-  base%>%
    group_by(variable) %>% 
    summarise(n = n()) %>%
    mutate(totalN = (cumsum(n)),
          percent = round((n / sum(n)), 3),
          cumpercent = round(cumsum(freq = n / sum(n)), 3))

# Tabular todas las variables de un data frame
for  (i in colnames(data)){
  print(i)
  print(unique(data[,i]))
}

#Chequear si hay una variable con todos ceros:
for (i in (1 : length(base))) {
  print(all(base[,i] == 0))
}

#Establecer una condición mayor y menor a la vez (acá mayor o igual a cero y menor o igual a 15:
between(variable, 0, 15)

#Ver los valores unicos de una variable (similar a tabular):
unique(dataset$variable)

# Cantidad de valores unicos de las columnas de una dataframe
sapply(lapply(db,unique),length)

# Ordenar un data frame por una variable:
df <- df[order(df$var, decreasing=TRUE),]

#Ordenar/sort una base de datos por más de una variable (2 acá, primer ascendente segunda descendiente):
base <- base[with(base, order(var1, -var2)), ]

#Guardar un objeto de R:
save(df, file = "otracarpeta/df.Rdata")
saveRDS(df, file = "otracarpeta/df.rds") # guardar un objeto de R para despues volver a cargarlo exactamente igual (con readRDS() )

#Cargar un objeto de R:
load(file = "otracarpeta/df.Rdata")
readRDS(file = "otracarpeta/df.rds")

#Eliminar variables por nombre (contiene, o empieza con):
base <- base %>% select(-contains("hola"))
#Otra opción:
base <- base[,!(names(base) %in% c("variable"))]
# o
base <- base[,-grep("^empieza_con", names(base))]

#Renombrar una variable / cambiar el nombre de una columna:
base <- rename(base, nuevo_nombre = viejo_nombre) #o
base <- base %>% rename(nuevo_nombre=viejo_nombre)

# Cambiar el nombre de muchas columnas / variables al mismo tiempo
data.table::setnames(d, old = c('a','d'), new = c('anew','dnew'))

#Diccionario de variables/etiquetas
diccionario <- labelled::lookfor(base)

#Redondear números:
base$variable <- round(base$variable, digits = 0)

#Pasar de variable fea de censo de datos CABA a número de radio censal (separar texto con substring/str_split, elegir elementos de una lista con sapply, completar dígitos con 0 de acuerdo a la la cantidad de digitods deseada con str_pad)
library(dplyr)
library(stringr)
data1 <- data1 %>%
  mutate(
    sub1 = str_pad(sapply(str_split(data1$variable_a_coregir, "_"), "[[", 1), 2, pad = "0"),
    sub2 = str_pad(sapply(str_split(data1$variable_a_coregir, "_"), "[[", 2), 2, pad = "0"),
    sub3 = str_pad(sapply(str_split(data1$variable_a_coregir, "_"), "[[", 3), 2, pad = "0"),
    radio = paste("020", sub1, sub2, sub3, sep="")
  )

#Encontrar valores duplicados / repetidos
frecuencias <- data.frame(table(base$variable))
frecuencias[frecuencias$Freq > 1,]

# Encontrar duplicados en varias columnas (chequear)
duplicated(cbind(var1, var2, var3))

#Eliminar valores repetidos/duplicados
duplicados <- c(which(duplicated(db$variable)))
db <- db[-duplicados,]
# Otra forma
db <- db[!duplicated(db$x),]
# En base a varias columnas
db <- db[!duplicated(cbind(db$var1, db$var2)),]

# Identificar valores duplicados de acuerdo a una o mas variables.
# En este caso se crea una variable "num_dups" que indica la cantidad de filas idénticas para cada fila, dup_id que ennumera los duplicados desde 1 hasta la cantidad de duplicados, y is_duplicated que identifica a los duplicados cuyo id es mayor a 1 (lo ideal es antes ordenar la base de acuerdo a una variable de interes, entonces te aseguras que el ID 1 sea la primera observacion de acuerdo a ese criteiro) (en este caso despues elimino los duplicados)
db <- db %>% 
    group_by(var1, var2) %>% 
    mutate(num_dups = n(), 
           dup_id = row_number()) %>% 
    ungroup() %>% 
    mutate(is_duplicated = dup_id > 1) %>% 
    filter(is_duplicated==TRUE)

#Agregar digitos a una serie de numeros, add leading zeroes:
db$variable <- sprintf("%04d", db$variable)

#Abrir o leer un csv (hay dos maneras de hacerlo, la primera es la mas rapida creo):
db <- data.table::fread("archivo.csv", encoding = "UTF-8") #1°

db <- read.csv("archivo.csv",sep=",",encoding="UTF-8", na.strings=c("")) #2°

# Geocoding. Pasar de direcciones a latitud y longitud:
db <- db %>% mutate(direc = paste(calle, " ", numero, sep=""), pais = "Argentina", provincia = "Buenos Aires") #esto es para que sea mas preciso
lat_longs <- db %>%
  tidygeocoder::geocode(direc, method = 'osm', lat = latitude , long = longitude) #aca es solo espcificando la direcion
lat_longs <- db %>%
  tidygeocoder::geocode(street = CALLE, city = LOCALIDAD, state = provincia, country = pais, postalcode = CODIGO_POSTAL, method = 'osm', lat = latitude , long = longitude) #aca es especificando toras variables lo que lo hace mas preciso y mas rapido


#Obtener elementos de un vector que no esten en otro vector, lo contrario a intersect / crear la funcion not in:
a <- colnames(base)
b <- c("var1", "var2")
'%!in%' <- function(x,y)!('%in%'(x,y))
c <- a[a%!in%b]

# Ordenar una base con respecto a una columna usando un orden especifico / explicito:
db <- db[match(c("<20", "20-29", "30-39", "40-49", "50-59", "60-69", ">=70"), d$grupetario),] #aunque esta no funciona bien para ggplot, la que sigue si
db$grupetario <- factor(db$grupetario, levels = c("<20", "20-29", "30-39", "40-49", "50-59", "60-69", ">=70"))


# Reshape de long a wide:
db_wide <- tidyr::pivot_wider(db, id_cols = "variable en comun", names_from = "id de base long", values_from = all_of(c), names_prefix = "un prefijo")

# Unir dos bases de datos con o sin las mismas columnas:
db_unif <- dplyr::bind_rows(db1, db2)

# Leer un archivo dbf
library("foreign")
df <- read.dbf("data.dbf", as.is = TRUE) #el "as.is" es para decirle que te lea las columnas como string y no factor

# Cambiar el encoding de una variable (ejemplo con un archivo dbf):
library("foreign")
df <- read.dbf("data.dbf", as.is = TRUE) #el "as.is" es para decirle que te lea las columnas como string y no factor
df2 <- select_if(df, is.character) #me quedo solo con las variables que sean character, y despues llamo a sus nombres (quiza es ineficiente y solo deberia crear un vector con los nombres, no un base nueva)
for (col in colnames(df2)){
  Encoding(df[[col]]) <- "UTF-8"
}

#Cambiar una parte repetia (patrón) de un string dentro de una variable
df$var <- gsub(pattern = "cambiarestepatron", replacement = "poreste", df$date)

# Hcer una tabulación de una variable (ver los valores únicos y sus frecuencias):
table(db$variable)
# Convertir la tabulación a un data frame (por ej par aluego exportar a xlsx)
tab <- as.data.frame(table(db$variable))

# Hacer una tabulación de una variable usando un ponderador / contar valores únicos ponderados
questionr::wtd.table(db$var, weights=db$ponderadores)

# Hacer una tabulación de dos variables / hacer una tabla two way (ver los valores únicos y sus frecuencias): (y convertirla a un data frame)
tab2 <- as.data.frame.matrix((table(db$var1, db$var2)))



#__________________________________________________________________________


### Curso Coursera: R Programming ##################

# Convertir variable a string / texto / character:
db$var <- as.character(db$var)

# Convertir variable a logical:
db$var <- as.logical(db$var)

# Convertir variable a numeric:
db$var <- as.numeric(db$var)

# Sacarle la clase a un vector
unclass(x)


# Crear una matriz:
m <- matrix(1:6, nrow=2, ncol=3)
# O uniendo columnas o filas
x <- 1:3
y <- 12:14
m <- rbind(x,y)
n <- cbind(y,x)
# Dimensiones de una matriz:
dim(m)
# Atributos de una matriz:
attributes(m)


# Crear una clase tipo factor:
x <- factor(c("yes", "yes", "no", "yes", "no"))

# Ponerle orden a los niveles de la clase factor:
x <- factor(c("yes", "yes", "no", "yes", "no"), levels=c("yes","no"))
# El nivel básico o baseline es el primero que le pones, esto es importante cuando usas modelos


# La diferencia entre el NA y NaN es que el NA es una clase mayor que engloba a NaN. NA tiene clases como integer, character, etc. mientras que NaN es una operacion matematica indefinida, y es una clase de Na.

# Obtener un vector logico (true, false) para cada elemento de un vector de la misma dimesion si es NA o NaN:
is.na(x)
is.nan(x)

# Cantidad de filas/columnas en un dataframe:
nrow(df)
ncol(df)

# NOMBRES: los objetos en R pueden tener nombres, lo cual es util para escribir codigo legible y objetos auto descriptivos:
x <- 1:3
names(x) <- c("hola", "chau", "aloja")
# o como una lista:
x <- list(a=1, b=2, c=3)
# o en una matriz:
m <- matrix(1:4, nrow=2, ncol=2)
dimnames(m) <- list(c("a", "b"), c("c", "d"))

## LEER TABLAS
# read.csv es igual a read.table excepto que el separador en csv es la coma, y en table es un espacio

# Para hacer que se lea más rapido, se puede:

#poner como argumento que ignore los cometarios:
read.table("foo.txt", comment.char = "")
#comment.char indica con qué caracter se inician los comentarios en la tabla, entonces lo ignora y no lo lee como dato para el data frame

# especificar las clases de las columnas. Si no especificas nada, R "adivina" de qué clase es cada columna
read.table("foo.txt", colClasses="factor", "Date")
# Si no sabes y quere que R adivine, si la base es muy grande, es mejor que adivine con, por ejemplo, las primeras 100 filas, y luego cargas la base entera con esas clases que adivino par alas primeras 10 filas:
initial <- read.table("database.txt", nrows=100)
classes <- sapply(initial, classes)
tabAll <- read.table("database.txt", colClasses=classes)

# Calcular los requerimientos de memoria para una tabla:
# suponemos que tiene 1.500.000 de filas y 120 columnas, y que la clase es solo numeric (en dodnde cada dato ocup 8 bytes:
# 1.500.000 x 120 x 8 = 1.440.000.000 bytes
# 1.440.000.000 / 2^20 (bytes/MB) = 1.373,29 MB
# 1.373,29 / 2^10 = 1.34 GB
# Y se necesita aproximadamente el DOBLE de esto para cargarla

# Otros comandos para leer datos:
readLines # para leer lineas de un archivo de texto
source o dget # para leer codigno en archivos de codigo en R
load

# Formatos Textuales:
dumping o dputing #preservan la metadata, por ejemplo las clases de las columnas. Pero no son tan eficientes en terminos de espacio
# Ejemplo: toma un objeto y crea el codigo para reconstruir el objeto en R:
y <- data.frame(a=1, b="a")
dput(y) # ver el output
# Dput es una funcion que toma un dataframe (u otro objeto) como argumento y 
#   devuelve un string con el codigo necesario para crear ese objeto
#convertirlo en un archivo y leerlo con dget:
dput(y, file="y.R")
new.y <- dget("y.R")
# Lo mismo con dump y source, que permite varios objetos:
x <- "foo"
y <- data.frame(a=1, b="a")
dump(c("x","y"), file="data.R")
rm(x, y)
source("data.R")


# CONEXIONES/INTERFACES
# El ejemplo mas comun es el argumento "file" que se usa en muhcas funciones para abrir una conexion con un archivo, pero existen otros tipos de conexiones.
gzfile #con un archivo comprimido gzip
url # con una pagina web
# Ver los argumentos de las funciones
str(file)


### SUBSETTING
# TABLES
[] # siempre te da un objeto con la misma clase que el objeto original, y sirve para elegir mas de un elemento. Ej:
x <- 1:15
x[x>5]
# o de otra forma:
x <- 1:15
u <- x>5
x[u]
[[]] #se usa para extraer elementos de una lista o df, el resultado no necesariamente es de la misma clase que el original, solo permite extraer un elemento (ej un numero de una celda de un data frame).
$ # se usa para extraer elementos de una lista o df que tengan un nombre
x <- list(foo=1:4, bar=0.6)
x[1]
x[[1]]
x$bar
x[["bar"]]
x["bar"]

# Seleccionar los elementos de una lista o dataframe segun su posicion
x <- list(foo=1:4, bar=0.6, baz="hello")
x[c(1,3)]
# LA diferencia entre $ y [[]] es que en el segundo se pueden usar indices computados:
x <- list(foo=1:4, bar=0.6, baz="hello")
name<- "foo"
x[[name]] # esta funciona
x$name # esta no funciona
# [[]] sirve para seleccionar elementos usando una secuencia de posiciones o numeros enteros *ej si quiero el tercer elemento del primer elemento de la lista):
x <- list(a = list(10,12,14), b=c(14.3,13.2))
x[[c(1,3)]] # que es lo mismo que:
x[[1]][[3]]

# SUBSETTING - MATRICES
x <- matrix(1:6, nrow=2, ncol=3)
x[1,2] # primera fila, segunda columna
x[,2] #segunda columna entera
# si subseteas solo un elemento o una columna o fila de una matriz, el resultado no es un objeto de tipo matriz sino vector. para que esto cambie le tenes que poner drop igual a FALSE:
x[1,2, drop=FALSE]
x[1, , drop=FALSE]

# PARTIAL MARTCHING
# Si no pones el nombre completo, R aproxima
x <- list(aardvark = 1:5)
x$a
x[['a']] #asi no aproxima
x[['a', exact=FALSE]] # asi si

# Seleccionar todos los elementos de un objeto (un vector, una matriz o un data frame) que no sean missings
x <- c(1,NA,3,NA,5)
nas <- is.na(x)
x[!nas]

# Seleccionar todos los elementos de un objeto que sean NO sean missing values
y <- x[!is.na(x)]

# Seleccionar todos los elementos positivos de un objeto (que NO sean missing values)
x[!is.na(x) & x > 0]

# Selccionar los elementos de un objeto de acuerdo a su posicion
x[c(3,5,7)]

# Seleccionar los elementos de un objeto excepto los que estan en determinada posicion:
x[c(-2, -10)] #o
x[-c(2, 10)]

# Seleccionar las filas de un data frame que no tengan missing values en ninguna de sus columnas:
airquality
filas_ok <- complete.cases(airquality)
airquality[filas_ok,]

# Seleccionar las filas de un data frame que no tengan missing values en algunas de sus columnas:
airquality
vars <- c("Ozone", "Solar.R", "Wind")
filas_ok <- complete.cases(airquality[,vars])
airquality[filas_ok,]

# Contar la cantidad de filas sin missing values en todas sus columnas:
sum(complete.cases(df))

# Se pueden hacer dos operaciones al mismo tiempo con ;
x <- c(1,3, 5);y <- c(3, 2, 10)

# Todos los elementos de un vector deben ser de la misma clase, no así con las listas.

# Ver los atributos de un objeto:
atributes(x)

# Crear una secuencia de números enteros:
x <- 1:20

#Eliminar un objeto:
rm('objeto')

# Ver los argumentos de una funcion un comando de R:
args(dir) #por ej de la funcion dir()

# Buscar la ayuda / documentaciones de operadores )no funciones:
?`:`

# Crear un secuencia de numeros especificando el intervalo entre los elementos:
seq(0, 10, by=0.5)

# Crear una secuencia de numeros especificando la longitud de la secuencia / la cantidad de numeros:
seq(5, 10, length=30)

# Repetir un numero o un objeto x veces:
rep(c(0, 1, 2), times = 10)

# Repetir cada elemento de un objeto x veces:
rep(c(0, 1, 2), each = 10)

# Pegar los elementos de un vector / unir los elementos de un vector
my_char <-c('My', 'name', 'is')
paste(my_char, collapse = " ")

# Pegar / concatenar / unir dos elementos
paste("Hello", "world!", sep = " ")
paste0("Hello", "world!") # no tiene opcion sep, no usa separador

# Crear un vector con realizaciones de una distribucion normal
y <- rnorm(n=1000, mean=5, sd=2)

# Crear un vector con distribucion Poisson:
y <- rpois(n=1000, mean=5, sd=2)

# Verificar si un objeto es identico a otro objeto
identical(x, y)

# Ver los environments
search()
# el orden en el que cargas los paquetes importa
# Ver lo que hay en el environment de una funcion
ln(environment(funcion))

# Difernecia entre el operador & y el operador &&
TRUE & c(TRUE, FALSE, FALSE) # este evalua TRUE contra cada elemento del vector de la derecha
TRUE && c(TRUE, FALSE, FALSE) # este evalua true solo contra el primer elemento del vector.
# Lo mismo pasa entre el operador | y el operador ||

# Evaluar si una condicion es verdadera:
isTRUE(2<3)

# Funcion "exclusive or": da TRUE si un argumento es TRUE y el otro FALSE, pero de otra manera te da false
xor(5 == 6, !FALSE)

# Funcion which(): toma un vector de resultados logicos (FALSE o TRUE) como argumento y devuelve las posiciones en las que estan los TRUE
ints<-sample(10)
which(ints < 2)
# Las funciones all() y any() son parecidas

# Crear un binary operator / operador binario:
"%p%" <- function(a,b){
    paste(a,b)
}

### LOOP FUNCTIONS

# Aplicar una funcion loopeando / iterando a lo largo de los elementos de una lista
x<-list(a=1:5, b=rnorm(10))
lapply(x, mean)
# Ahora con una funcion generica: en este caso una fucnion que te saca la primera columna de una matriz
x<-list(a= matrix(1:4,2,2), b=matrix(1:6,3,2))
lapply(x, function(elt) elt[,1])
# sapply es igual pero trata de simplificar el resultado de lapply. Por ej, si los outputs son vectores de una longitud, te los pone todos en un solo vector

# apply itera sobre una matriz:
x<-matrix(rnorm(200),20,10)
apply(x,2,mean) # saca el promedio de la segunda dimension, que en este caso es el numero de columnas de la matriz, o sea que saca el promedio de cara columna y asi "elimina" la primera dimension
apply(x,1,sum) # aca es la suma por fila, solo preserva la primera dimension y al hacerlo colapsa las columnas sumandolas para cada fila.

# Calcular cuantiles de de columnas de una matriz
x<-matrix(rnorm(200),20,10)
apply(x,2,quantile, probs=c(0.25,0.75))

# mapply: aplica una funcion en paralelo sobre un set de argumentos. Sirve para vectorizar una funcion que no permite vectores como argumentos
list(rep(1,4),rep(2,3),rep(3,2),rep(4,1))
# es equivalente a
mapply(rep, 1:4, 4:1) #seria como loopear sobre un zip como en python, entre los vectores 1:4 y 4:1, uniendo el primer eemento del primer vector con el primer elemento del segundo vector, etc.
# Otro ejemplo:
noise <-function(n,mean,sd){
    rnorm(n,mean,sd)
}
#esto te permite hacer
noise(5,1,2)
#pero no
noise(1:3,1:3,2) #seria como repetir esta funcion asi: noise(1,1,2), noise(2,2,2), noise(3,3,2)
# con mapply se puede
mapply(noise, 1:3,1:3,2) # es quivalente a hacer list(noise(1,1,2), noise(2,2,2), noise(3,3,2))

# tapply: se usa para pasar una funcion a un vector pero subseteado por una variable categorica
x <- c(norm(10),norm(10),norm(10,1))
factor <- gl(3,10)
tapply(x, factor, mean)

# split: no es una funcion loopeadora pero se puede usar con una. Lo que hace es separar un vector o data frame en los grupos designados por una variable categorica. Devuelve una lista sobra la cual se puede usar una funcion loopeadora. Ej:
x <- c(norm(10),norm(10),norm(10,1))
factor <- gl(3,10)
split(x,factor)
# Esto:
lapply(split(x,factor),mean) # es equivalente a
tapply(x, factor, mean)
# Ejemplo con l base airquality: calcular promedios por mes para diferentes variables
s <- split(airquality, airquality$Month)
lapply(s, function(x) colMeans(x[,c("Ozone", "Solar.R", "Wind")], na.rm=TRUE))
# para que salga mas lindo, en una matriz en vez de una lista, usar sapply
sapply(s, function(x) colMeans(x[,c("Ozone", "Solar.R", "Wind")], na.rm=TRUE))
# splitear en mas de un nivel
x<-rnorm(10)
f1<-gl(2,5)
f2<-gl(5,2)
interaction(f1,f2)
str(split(x,list(f1,f2)))

# vapply: similar a sapply, pero te permite especificar el tipo de elementos del vector en el que sapply te agrupa los resultados, en vez de que R 'adivine' el tipo. sirve para que te salte un error si no te das cuenta de algo

### DEBUGGING / ARREGLANDO ERRORES

# Setear una opcion para que siempre abra el debugger cuando encuentra un error

# Poner una opcion en una funcion para abrir el debugger en donde pegues esto:
if(browser==TRUE){browser()} # Opens debugger if broser set to TRUE
# Ej:
sumar(a,b, browser=TRUE){
  uno <- a
  dos <- b
  if(browser==TRUE){browser()}
  return(uno+dos)
}

# Guardar un objeto creado dentro de una funcion cuando estas usando el debugger, para despues cargarlo en otro ambito preservando el estado exacto que tenia en esa parte de la funcion
saveRDS(df, file = "otracarpeta/df.rds") # guardar un objeto de R para despues volver a cargarlo exactamente igual (con readRDS() )
readRDS(file = "otracarpeta/df.rds")

# traceback: te pone la ultima linea en donde ocurio el error exactamente
lm(y-x)
traceback()

# DEBUGGER:
debug(lm)
lm(y-x)

# tocas n para ir al proximo paso

# recover: hace que cuando haya un error se frene, sin devolverte la consola, hasta que no resuelvas el error
options(error=recover)
options(error=NULL) # para revertirlo

#  <<- operator which can be used to assign a value to an object in an environment that is different from the current environment.

# Ver el contenido / los argumentos de una funcion:
str(funcion) #str significa structure
# es la funcion mas util de todo R !!!
# Tambien sirve para tener un resumen de un data frame
str(airquality)

# OPTIMIZAR EL CODIGO

# Medir el tiempo que tardo un comando:
system.time(rnorm(1000000,2,3)) #el user time es el tiempo que le cuesta a la CPU, y el elapsed time es el tiempo que tardas vos en ver el resultado. A veces el user puede ser menor al elapsed (por ej cuando una funcion llama a una pagina web, vos esperas a que internet te devuelva algo, lo cual suma al elapsed time, pero no al user time porque ahi el cpu no hace nada. Otras veces el user puede ser menor al elapsed, por ejemplo cuando usas mas de un nucleo para procesar, ej si usas dos nucleos el user va a ser la suma del tiempo de cada nucleo, pero como vos paralelizaste solo ves que te tarda lo que tardo uno solo)
# !!! aca no se si hay que poner llaves alrededor de "rnorm(...)"

# Otra forma de medir el tiempo de un codigo
library(tictoc)
tic("Timer 1: tiempo total")
tic("Timer 2: tiempo en crear un vector")
x <- 1:2
toc()
assign('b', x)
toc()

# Otra forma de medir el tiempo de un codigo
start_time <- Sys.time()
Sys.sleep(3)
end_time <- Sys.time()
end_time - start_time

# Profiling: usar Rprof() y summaryRprof()
Rprof()
# Rprof te da, cada dos segundos, una lista con el stack de funciones que se estan ejecutando en el momento. Entonces te das cuenta mas o menos en donde tarda mas tu codigo
summaryRprof()
# summaryRprof(): resume los resultados de Rprof y te permite ver de manera mas amigable que parte del codigo tarda mas
# esta ultima funcion tiene dos argumentos utiles: by.total y by.self
# by.total mide el tiempo que pasa R en una funcion determinada sobre el tiempo total de procesamiento del codigo
# by.self mide lo mismo, pero al denominador le quita el tiempo utilizado en correr la funcion de mayor nivel (usualmente las funciones de mayor nivel no hacen la parte importante, sino que llaman a funciones de menor nivel que hacen el verdadero laburo). O quiza es al reves. La cuestion es que este te da el tiempo real que tarda cada funcion.

# Ver cuanto pesa un objeto en la memoria
object.size(plants)

# Scatterplot con la funcion baes de R: plot()
# Titulo:
plot(cars, main="Titulo")
# Subtitulo
plot(cars, sub="Subtitulo")
# Puntos de color rojo:
plot(cars, col=2)
# Limites a los ejes:
plot(cars, xlim = c(10, 15))
# Triangulos en vez de puntos:
plot(cars, pch=2)


## TIBBLES

# Convertir un data frame en tibble
as_tibble(trees)

# Ver n elementos aleatorios de una tibble
slice_sample(trees, n = 10)

# Ver más elementos de una tibble que los que aparecen por default:
as_tibble(trees) %>% 
  print(n = 5, width = Inf)



# Operaciones con strings -------------------------------------------------

# Obtener la posicion de un string
stringr::str_locate("ABCDEC", "C") # la primera vez que aparece
stringr::str_locate_all("ABCDEC", "C") # todas las veces que aparezcqa

# Determinar si un string se encuentra dentro de otro
stringr::str_detect("HOLA JUAN", "HOLA")
grepl("hola", "hola chau")

# REGEX -------------------------------------------------------------------

# "Escapar", encontrar o hacer referencia a un string cuando tiene algun significado determinado en regex (ej ., ,, $)
stringr::str_detect("hola$chau", "\\$") #!!! me suena a que se necesita solo a 1 barra
stringr::str_split("hola$chau", "\\$")

# Empieza con / starts with // termina con / ends with
"^inicio.*fin$"
# el ^ significa "empieza con"
# el . significa "cualquier caracter"
# el * significa "0 o más del caracter anterior" (que en este caso es cualquier caracter)
# el $ significa que ese es el final del caracter


# Comandos para GRÁFICOS --------------------------------------------------


# Partir un título en más de una lína / añadir un line break en un título
plot <- ggplot(data, aes(var1, var2)) +
    geom_bar(stat = "identity") +
    ggtitle("Text in first line\nof and text in second line") # /nof operator

# Show only some of the tick labels
ggplot(data, aes(var1, var2, group=1)) + # Group one is to deal with an error for line graphs
    geom_line() +
    scale_x_discrete(breaks = levels(data$var1)[floor(seq(1, nlevels(data$var1), length.out = 20))]) +# length.out controls the number of labels you want to show
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1, size=8)) # rotate tick labels

# Modificar los ticks de un grafico de manera facil
grafico <- ggplot(df_eph_hog, aes(x = IPCF)) +
  geom_histogram(color = "black", fill="white") +
  xlab("Ingreso per cápita familiar") +
  ylab("Frecuencia") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) # esto


# Mostrar multiples graficos uno al lado del otro con el paquete basico de R graphics
par(mfrow=c(1,2))
plot(mtcars$mpg, mtcars$disp)
plot(mtcars$hp, mtcars$disp)

# Guardar un gráfico como PNG
png(file="C:/grafico.png", width=600, height=350)
hist(Temperature, col="gold") # cualquier grafico
dev.off()
# Guardar un gráfico como PDF
pdf(file="saving_plot4.pdf")
hist(Temperature, col="violet")
dev.off()
#!!! es mejor usar ggsave(), cambiar!

# GGPAIRS: graficos de correlaciones / correlogramas /scatters / coeficientes de correlacion
data(flea)
GGally::ggpairs(flea, columns = 2:4)
    # Cambiar el tamaño (y otra estetica) de los nombre de las variables en un correlograma
GGally::ggpairs(flea, columns = 2:4) + ggplot2::theme(strip.text=element_text(size=15))


# Comandos para manejo de DATOS ESPACIALES --------------------------------

# Dividir un polígono en X partes de aproximadamente el mismo tamaño
# Fuente: https://gis.stackexchange.com/questions/375345/dividing-polygon-into-parts-which-have-equal-area-using-r
library(sf)
library(mapview)
library(tidyverse) 
library(dismo)
library(osmdata)  
library(mapview)
split_poly <- function(sf_poly, n_areas){
    # create random points
    points_rnd <- st_sample(sf_poly, size = 10000)
    #k-means clustering
    points <- do.call(rbind, st_geometry(points_rnd)) %>%
        as_tibble() %>% setNames(c("lon","lat"))
    k_means <- kmeans(points, centers = n_areas)
    # create voronoi polygons
    voronoi_polys <- dismo::voronoi(k_means$centers, ext = sf_poly)
    # clip to sf_poly
    crs(voronoi_polys) <- crs(sf_poly)
    voronoi_sf <- st_as_sf(voronoi_polys)
    equal_areas <- st_intersection(voronoi_sf, sf_poly)
    equal_areas$area <- st_area(equal_areas)
    return(equal_areas)
}

# Combinar poligonos con IDs duplicados
radios <- radios %>%
    group_by(link) %>%
    summarize(geom = if(n() == 1) geom else st_union(geom))
# Esto es relevante para los radios censales de ARG. Algunos archivos tienen
# geometrias repetidas, por un lado, y por otro lado hay algunos radios que se
# componen de más de una geometría. Por lo tanto, si simplemente eliminás los
# duplicados, perdés algunas partes de estos radios combinados.

# Filtrar poligonos en base a la interseccion con otro objeto espacial
# En este caso, nos quedamos con todos los poligonos en "poly" que intersecten
# con al menos un objeto de "sf_objects"
polys_intersected <- polys[lengths(st_intersects(polys, sf_objects)) > 0, ]

# Para bases en las cuales hay mas de un poligono con el mismo ID, podemos 
# unificar todos los polígonos con el mismo ID en una sola geometría, y luego
# eliminar las filas con IDs duplicados.
# Esto pasa con el censo, sobre todo en los polígonos de islas (San Fernando,
# Tierra del Fuego)
poligonos <- poligonos %>%
    group_by(id) %>%
    summarize(geom = if(n() == 1) geom else st_union(geom))
# (Ojo que en ese caso nos estamos quedando solo con la variable ID y la 
# geometria, si hay mas variables hay que contemplarlas en el summarize o luego
# hacer un merge)

# Combinar geometrias
sf::st_union(x, y)
sf::st_combine(x, y)

# Calcular el área de un polígono
rgeos::gArea(pol)
sf::st_area(pol)

# Encontrar los poligonos contiguos a cada poligono
library(sf)
comunas <- st_read("directorio_a_comunas_de_caba.shp")
comunas <- comunas %>% select(COMUNAS) %>% rename(comuna = COMUNAS)
sf_use_s2(FALSE) # apagar geometria esferica
comunas$contiguous <- sapply(st_touches(comunas), function(x) {
    paste(st_drop_geometry(comunas[x, "comuna"]), collapse = ", ")
})
sf_use_s2(TRUE)
comunas$contiguous <- gsub("^c\\(|\\)$", "", as.character(comunas$contiguous))
comunas$contiguous <- gsub("(\\d+)", "COMUNA \\1", comunas$contiguous)

# Disolver varios poligonos en uno solo / unir poligonos
st_combine(x,y)
st_union(x)

# Descargar capas del geoservicio de INDEC
indec <- "WFS:http://geoservicios.indec.gov.ar/geoserver/ows?service=wfs&version=1.3.0&request=GetCapabilities"
capas_indec <- st_layers(indec)
localidades = st_read(indec, "geocenso2010:localidades_codigo") # leer una capa
localidades <- st_transform(localidades,crs=5349) # POSGAR 2007 ARGENTINA 7
rm(capas_indec)

# Crear puntos dentro de un polígono o una línea.
# Su posición puede ser aleatoria (type='random'), regular, estratificada, no alineada, hexagonal, lattice y Fibonacci
points <- spsample(pol, type = "hexagonal", cellsize = 0.5)

# Quedarse con el polígono de mayor área de un grupo de polígonos (de la clase "Spatial polygons")
pol <- sapply(pol@polygons, slot, "area") %>% 
    {which(. == max(.))} %>% 
    pol[.]

# Crear un Hexbin Map / dividir una superficie en polígonos idénticos
# Source: https://medium.com/swlh/spatial-data-analysis-with-hexagonal-grids-961de90a220e
study_area <- raster::getData("GADM", country = "GB", level = 0, path = tempdir(), )
    # Aumentar la resolucion del raster ('disaggregate') y quedarse cona la geometría (es decir, que no sea más un data frame)
study_area <- study_area %>% raster::disaggregate() %>% sp::geometry()
    # Graficar
plot(study_area, col = "grey50", bg = "light blue", axes = TRUE, cex = 20)
    # Crear los puntos que van a ser los centros de los exágonos
hex_points <- sp::spsample(study_area, type = "hexagonal", cellsize = 0.5)
    # Transformar los centros de los hexágonos en polígonos
hex_grid <- sp::HexPoints2SpatialPolygons(hex_points, dx = 0.5) # OJO: El espaciado entre dos puntos ('dx') tiene que ser igual al 'cellsize' de los puntos en la funcion anterior
    # Graficar
plot(study_area, col = "grey50", bg = "light blue", axes = TRUE)
plot(hex_points, col = "black", pch = 20, cex = 0.5, add = T)
plot(hex_grid, border = "orange", add = T)

# Herramienta para crear una bounding box / definir un límite rectangular en un mapa y devuelve coordenadas
# http://bboxfinder.com/

# Ver todos los métodos posibles en la libreria "sf" (simple features for R)
methods(class = "sf")

# Explorar un mapa / mirar un poligono o puntos en un mapa interactivo
mapview::mapview(polygon)
  # si queres mapear dos podes hacer asi
mapview::mapview(polygon) + puntos
  # o asi para darle formaro a ambos
mapview(elemZones, ...) + mapview(precincts, ...)

# Hacer un grafico de un mapa
plot(sf::st_geometry(df))
# ver opciones en: https://r-spatial.github.io/sf/articles/sf5.html
# Otra forma es con ggplots
library(ggplot2)
ggplot() + geom_sf(data = df, color = 'red') + geom_sf(data = df2)

# Leer / cargar un archivo de datos espaciales
shape <- sf::st_read(path)

# Convertir un data frame en un geo data frame
df <- sf::st_as_sf(df, coords = c("LONGITUDE", "LATITUDE"), crs = 4326)

# Convertir un geo data frame a un data frame / eliminar el componente espacial de un spatial data frame
sf::st_drop_geometry(sp_df)
#  o sino ver esto
df %>% sf::st_set_geometry(NULL)

# Crear columnas para latitud y longitud a partir de una geometria
library(dplyr)
db <- db %>%
  dplyr::mutate(LONGITUDE = sf::st_coordinates(.)[,1],
                LATITUDE = sf::st_coordinates(.)[,2])

#· CRS:
# Buscador de CRSs: https://epsg.io/
# CRS de CABA: https://epsg.io/9498
# CRS de CIUDAD DE MEXICO: https://epsg.io/4487

# Ver el CRS de un objeto espacial
sf::st_crs(sp_db)

# Cambiar el CRS de un data frame
df = sf::st_transform(df, crs=5349) # con Simple Forms
df = sp::spTransform(df, CRS('+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0')) # con SP

# Merge espacial / join espacial / unir dos geometrias segun su posicion
df = sf::st_intersection(df, area_polygon)

# Seleccionar / filtrar los puntos dentro de un poligono
good_points <- st_filter(point.sf, poly)

# Crear una caja alrededor de un poligono / bounding box
poligono_box <- sf::st_bbox(poligono)

# Obtener los limites de un poligono (lo mismo que una caja)
limites <- raster::extent(poligono)

# Seleccionar los puntos dentro de una caja creada
df = subset(df, LONGITUDE > poligono_box[1] & LONGITUDE < poligono_box[3] & LATITUDE > poligono_box[2] & LATITUDE < poligono_box[4])

# Exportar un mapa a HTML (refinar)
mapview::mapshot(
  mapview::mapview(df),
  url = "C:/Users/juanb/mapita.html"
)

# Geoservicios de INDEC / cómo usar geoservicios WFS
indec <- "WFS:http://geoservicios.indec.gov.ar/geoserver/ows?service=wfs&version=1.3.0&request=GetCapabilities"
capas_indec <- sf::st_layers(indec)
localidades <-  sf::st_read(indec, "geocenso2010:localidades_codigo")

# Guardar un data frame espacial
st_write(spdf, "path", delete_dsn = T, delete_layer = T) #los delete son para que sobrescriba el archivo si ya existe

# Relacion entre barrios y comunas de la Ciudad de Buenos Aires
  #antes a la variable barrio ponerla en mayuscula
dbs$barrio <- stringi::stri_trans_general(dbs$barrio,"Latin-ASCII")
dbs$barrio <- toupper(dbs$barrio)
# (volver a chequear esto)
comuna1 <- c("CONSTITUCION", "MONTSERRAT", "PUERTO MADERO", "RETIRO", "SAN NICOLAS", "SAN TELMO")
comuna2 <- c("RECOLETA")
comuna3 <- c("BALVANERA", "SAN CRISTOBAL")
comuna4 <- c("BARRACAS", "BOCA", "NUEVA POMPEYA", "PARQUE PATRICIOS")
comuna5 <- c("ALMAGRO", "BOEDO")
comuna6 <- c("CABALLITO")
comuna7 <- c("FLORES", "PARQUE CHACABUCO")
comuna8 <- c("VILLA LUGANO", "VILLA RIACHUELO", "VILLA SOLDATI")
comuna9 <- c("LINIERS", "MATADEROS", "PARQUE AVELLANEDA")
comuna10 <- c("FLORESTA" , "MONTE CASTRO", "VELEZ SARSFIELD", "VERSALLES", "VILLA LURO", "VILLA REAL")
comuna11 <- c("VILLA DEL PARQUE", "VILLA DEVOTO", "VILLA GRAL. MITRE", "VILLA SANTA RITA")
comuna12 <- c("COGHLAN", "SAAVEDRA", "VILLA PUEYRREDON", "VILLA URQUIZA")
comuna13 <- c("BELGRANO", "COLEGIALES", "NUNEZ")
comuna14 <- c("PALERMO")
comuna15 <- c("AGRONOMIA", "CHACARITA", "PARQUE CHAS", "PATERNAL", "VILLA CRESPO", "VILLA ORTUZAR")

# Crear un mapa interactivo. Definir distintos colores para los puntos segun una variable
paleta_de_colores <- RColorBrewer::brewer.pal(length(unique(sp_db$var)), "Set1") # creo la paleta de colres
mapview::mapview(sp_db, zcol = "var", legend=TRUE, col.regions=paleta_de_colores, layer.name = 'Nombre')
# zcol: la variable para usar como grupos de colores
# col.regions: definir la paleta de colores
# legend: mostrar la leyenda de los colores
  # La otra opcion es separar la db en varias segun la variable que quiera, y crear un mapview para cada una. Ej:
mapview::mapview(sp_db[sp_db$var==1,], color="red", col.regions="red", cex=1) +
  mapview::mapview(sp_db[sp_db$var==2,], color="yellow", col.regions="yellow", cex=1)
# color: color de la linea del punto o poligono
# col.region: color del relleno de punto o poligono

# Crear un mapa interactivo. Cambiar el tamaño de los puntos
mapview::mapview(sp_db, cex=3) # el default es 8


## MAPAS COROPLETICOS

# Agregar etiquetas de valores a los poligonos de un mapa
ggplot(sp_db) +
  geom_sf(aes(fill = var_valores)) +
  geom_sf_label(aes(label = var_valores), label.padding = unit(1,"mm"))

# Eliminar los bordes de los poligonos de un mapa /cambiar el grosor de los bordes
ggplot(sp_db) +
  geom_sf(aes(fill = var_valores), color=NA) # eliminar (sin color)
ggplot(sp_db) +
  geom_sf(aes(fill = var_valores), color="white", lwd=0.25) # con color blanco, se puede cambiar el grosor

# Agregar etiquetas con valores de una variable a los polígonos
ggplot(sp_db) +
  geom_sf(aes(fill = var_valores)) +
  geom_sf_text()

# Seleccionar una columna del data frame espacial / usar una columna de un data frame espacial excluyendo la columna geom / slice a column of a spatial data frame excluding geometry and without using "$" operator
ggplot(db) +
  geom_sf(aes(fill = as.data.frame(db)[,"var"]))

# Trabajar con RASTERS: https://www.neonscience.org/resources/learning-hub/tutorials/raster-data-r

# Crear una grilla a partir de un poligono: https://rpubs.com/huanfaChen/grid_from_polygon

# Acceder a los datos dentro de un SpatialPolygonsDataFrame
    # se leen con rgdal::readOGR. Primero leo una base de ejemplo que viene con el paquete
    set_thin_PROJ6_warnings(TRUE)
    ogrDrivers()
    dsn <- system.file("vectors", package = "rgdal")[1]
    ogrListLayers(dsn)
    ogrInfo(dsn)
    ogrInfo(dsn=dsn, layer="cities")
    owd <- getwd()
    setwd(dsn)
    ogrInfo(dsn="cities.shp")
    ogrInfo(dsn="cities.shp", layer="cities")
    setwd(owd)
    ow <- options("warn")$warn
    options("warn"=1)
    cities <- readOGR(dsn=dsn, layer="cities")
# Despues se accede a todo con el simbolo arroba @
cities@data

# Convertir un numero a una unidad / convert numeric to meter units
units::as_units(num, "meter")

# Descargar un mapa base / mapa de fondo
ggmap::register_google(key = "aca_poner_la_api_key", write = TRUE)
bm <- ggmap::get_map(
    location='Buenos Aires',
    # c(lon=-34.60795231689911, lat=-58.37034520538574),
    zoom=11, crop=T,
    scale="auto",color="bw",source="google", maptye="roadmap")
gg <- ggmap::ggmap(bm, extent='panel',padding=0)
gg

# Extraer los limites de un ggmap / extraer el bounding box de un ggmap
bm <- ggmap::get_map(
    location='Buenos Aires',
    zoom=11, crop=T,
    scale="auto",color="bw",source="google", maptye="roadmap")
bb <- attr(bm, "bb")

# Algebra lineal ----------------------------------------------------------

# Crear una matriz (vacia)
matrix(ncol=3, nrow=3)

# Crear una matriz diagonal (todos ceros excepto en la diagonal principal)
diag(ncol=3, nrow=3)
# Tambien sirve para extraer los elementos de la diagonal principal de una matriz
diag(matriz)

# Calcular el determinante de una matrix
det(matriz)

# Invertir una matriz
solve(matriz)

# Calcular la traza de una matriz (la suma de los elemntos de la diagonal principal)
sum(diag(matriz))


# Econometría -------------------------------------------------------------

# Especificar una variable como categorica / Crear una especificacion con una binaria
# por cada valor de una variable categorixa
lm(dep_var ~ factor(categ_var))

# Recuperar los EFECTOS FIJOS de una regresión
    # Para el modelos estimados con el paquete "plm"
plm::fixef(modelo)

# Weak instrument Test / Test Cragg Donald para instrumentos débiles
# https://cran.r-project.org/web/packages/cragg/vignettes/introduction.html

# Calcular la funcion de máxima verosimilitud de un modelo del tipo "plm"
logLik.plm <- function(object){
  out <- -plm::nobs(object) * log(2 * var(object$residuals) * pi)/2 - deviance(object)/(2 * var(object$residuals))
  
  attr(out,"df") <- nobs(object) - object$df.residual
  attr(out,"nobs") <- plm::nobs(object)
  return(out)
}


# Series de tiempo --------------------------------------------------------

# Sacarle la tendencia lineal a una variable
pracma::detrend(as.matrix(var), tt = 'linear')
    # A todas las variables de un df
df_dt <- as.data.frame(pracma::detrend(as.matrix(df), tt = 'linear'))


# Econometría espacial ----------------------------------------------------

# Convertir la matriz de pesos espaciales en sparse (https://cmdlinetips.com/2019/05/introduction-to-sparse-matrices-in-r/)
# crear matrix de contiguidad
wm_q <- spdep::poly2nb(SP,queen=TRUE)
# convierte la matriz a lista
wm_q <- spdep::nb2listw(wm_q ,style="W" , zero.policy = TRUE)
# se queda con los pesos para cada poligono, como un vector
list.we <- unlist(wm_q$weights)
# Se queda con los vecinos de cada poligono, como un vector
list.nb <- unlist(wm_q$neighbours)
# Crea un vector que tiene cada id del poligono repetido tantas veces como vecinos tenga (ej si el pol 1 tiene 3 vecinos, el cetor empieza como 1, 1, 1)
list.id <- c(1:length(wm_q$neighbours))
list.id <- rep(list.id,lengths(wm_q$neighbours))
# Elimina poligonos sin vecinos (de la lista de ids y de vecinos)
list.id <- list.id[-which(list.nb==0)]
list.nb <- list.nb[-which(list.nb==0)]
# Crea la matriz sparse
matriz <- Matrix::sparseMatrix(list.id,list.nb,x=list.we,dims=c(nrow(caba),nrow(caba)))
# La convierte a lista de weights
pol.w <- mat2listw(matriz)

# Libro increible: https://spatialanalysis.github.io/handsonspatialdata/index.html

# Crear el rezago espacial de una variable (https://gist.github.com/chrishanretty/664e337c267a53a7de97)
df$sp.lag <- spdep::lag.listw(df$var, matriz_w)

# Test I de Moran
spdep::lm.morantest(modelo_mco, listw=matriz_como_lista, alternative = "two.sided")


# R Markdown --------------------------------------------------------------

# Funcion para crear un "image carousel" (carrusel de imagenes) en Mardown
library(htmlwidgets)
library(slickR)
create_image_carousel <- function(path, add_dots=T){
    
    # Sources:
    # https://kenwheeler.github.io/slick/
    # https://cran.r-project.org/web/packages/slickR/vignettes/basics.html
    
    images_path <- list.files(path, full.names=T, pattern="\\.png$")
    
    cP1 <- htmlwidgets::JS("function(slick,index) {
                            return '<a>'+(index+1)+'</a>';
                       }") # !!! Eventually try to modify this so that it displays months/dates
    
    opts_dot_number <- settings(
        initialSlide = 0,
        slidesToShow = 1,
        slidesToScroll = 1,
        focusOnSelect = TRUE,
        dots = TRUE,
        customPaging = cP1
    )
    
    slick <- slickR(
        obj = images_path,
        height = "100%",
        width = "100%"
    ) 
    
    if(add_dots==T){
        return(slick + opts_dot_number)
    }
    else{
        return(slick)
    }
}

# Bullets
* unordered list
    + sub-item 1 
    + sub-item 2 
        - sub-sub-item 1

# Listas ordenadas
1.  Bird
2.  McHale
3.  Parish


# Fijar opciones globales para todo el documento
knitr::opts_chunk$set(
    echo = TRUE,
    message = FALSE) # Suprimir mensajes normales de R (ej cuando lees un csv)

# Cambiar el formato de una fila determinada de una tabla
knitr::kable(df_tabla) %>%
    kable_styling()%>%
    kableExtra::row_spec(nro_fila, bold=T)
    # en este caso hago que la fila "nro_fila" sea negrita

# Mejorar el formato de una table de Kable
knitr::kable(df_tabla) %>%
    kable_styling()

# Hacer referencias a objetos definidos en el ambiente en el texto del markdown
`r mean(objeto)`
# Todo lo qie aparezca entre backticks `` y empiece con "r", el markdown lo
# entiende como codigo en R

# Agregar espacios en blanco en el cuerpo del texto (util para agregar espacio entre chunks con graficos)
<br><br><br>

# Pagina util: https://rstudio4edu.github.io/rstudio4edu-book/rmd-fancy.html,
# https://bookdown.org/yihui/rmarkdown/html-document.html

# Incluir un GIF o video
knitr::include_graphics(path="./file.gif") # !!! OJO, el archivo tiene que estar en la misma carpeta que el código del markdown, sino no funciona

# Opciones. Indice al principio
# https://bookdown.org/yihui/rmarkdown/html-document.html
title: "Invoice Contagion Summary"
output:
    html_document:
        toc: yes # add a table of contents
        toc_float: true # 'true' to make it float as you scroll, 'no' to fix it at the beginning
        toc_depth: 6 # define the depth of the subtitles that will be shown in the TOC
        number_sections: no # add section numbering or not
        collapsed: yes #controls whether the TOC appears with only the top-level (e.g., H2) headers. If collapsed initially, the TOC is automatically expanded inline when necessary.
        smooth_scroll: yes # controls whether page scrolls are animated when TOC items are navigated to via mouse clicks.
        theme: simplex # Bootstrap theme

# Que el codigo se ajuste al ancho del documento / wrap code (creo que es solo para PDFs)
# Agregar esto en las opciones iniciales
header-includes:
    \usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}

# Hacer que un código no se corra (el equivalente a comentar el código

```{r eval=FALSE, include=FALSE}
print("Hola")
```

# Elegir si ejecutar un chunk o no
```{r, eval=FALSE}
print("Hola")
```
# por ejemplo, se puede poner un chunk al principio que definia parametros igual a TRUE o a FALSE para elegir que correr, y despues poner eval=un_parametro en el chunk

# Insertar una linea / enter
    #poner dos espacios despues del texto
#o  poner "\" (sin comillas) despues del texto
one\
two

# Formato de texto: bold
**bold**
break

# Parallel processing (in Windows) -----------------------------------------------------

# Step-by-step

    # Define objects
a <- 1
b <- 2

    # Create a function whose argument is a vector with the number of times we will run it
multiply <- function(trials){a*b}

    # Create vector with number of trials
n <- 1:10

    # Load necessary libraries
library(parallel)
library(doParallel)

    # Set the number of cores to use
leave_out_cores <- 2
numCores <- detectCores() - leave_out_cores
    # Create "numCores" copies of R running in parallel
cl <- makeCluster(numCores)
    # Register cluster
registerDoParallel(cl)
    # Export our function to the cluster (together with all objects that will be used in the function)
clusterExport(cl=cl, varlist=list('multiply', 'a', 'b')) # !!! careful: this only handles object defined in the current or upper frame, not in lower frames
    # Run function "n" times
results <- c(parLapply(cl=cl, X=n, fun=multiply))
results
stopCluster(cl=cl)



# Github ------------------------------------------------------------------

# Correr un script almacenado en un repositorio de Github
    # Primero ir al script en la página de Github, cliquear el botón "Raw",
    # y copiar el URL de la página que se abre
source("https://raw.githubusercontent.com/Datos3F/GeoPortal/main/funciones/obtenerCapa.R")
