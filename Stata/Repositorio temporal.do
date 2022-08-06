************************
*** LINEA DE POBREZA ***
************************

*https://sitioanterior.indec.gob.ar/informesdeprensa_anteriores.asp?id_tema_1=4&id_tema_2=27&id_tema_3=65

* La tabla de equivalencias no cambia entre 2012-t3 y 2013-t1 (pero sí es distinta a la de 2021 y 2018)

*!!! probar con un corte de pobreza del percentil 30 del ingpch
*use "C:\Users\juanb\OneDrive\Documentos\Juan\MECON\4-Bienes_Personales\data\data_in\bases_datos_engho2012\ENGHo - Hogares.txt", replace

*insheet using "C:\Users\juanb\OneDrive\Documentos\Juan\MECON\4-Bienes_Personales\data\data_in\bases_datos_engho2012\ENGHo - Hogares.txt", delimiter("|") clear
*destring, replace
*compress

*!!! No sé dónde está guardado esto en la master, lo hago asi nomas y despues corregimos
*do "C:\Users\juanb\OneDrive\Escritorio\Notas de Programación\Stata\comando_cuantiles.do"

*cuantiles ingpch [w=pondera], n(10) o(id componente relacion edad) g(dingpch_m)

/*
rename ch06 edad
recode ch04 (2=0), gen(hombre)


* Adulto equivalente
gen ae=. 
	* Bebes
	replace ae=.33 if edad<1
	replace ae=.43 if edad==1
	replace ae=.50 if edad==2
	replace ae=.56 if edad==3
	replace ae=.63 if edad==4
	replace ae=.63 if edad==5
	replace ae=.63 if edad==6
	replace ae=.72 if edad==7
	replace ae=.72 if edad==8
	replace ae=.72 if edad==9
	* Hombres
	replace ae=.83 if hombre==1 & edad==10
	replace ae=.83 if hombre==1 & edad==11
	replace ae=.83 if hombre==1 & edad==12
	replace ae=.96 if hombre==1 & edad==13
	replace ae=.96 if hombre==1 & edad==14
	replace ae=0.96 if hombre==1 & edad==15
	replace ae=1.05 if hombre==1 & edad==16
	replace ae=1.05 if hombre==1 & edad==17
	replace ae=1.06 if hombre==1 & (edad>=18 & edad<=29)
	replace ae=1 if hombre==1 & (edad>29 & edad<=60)
	replace ae=.82 if hombre==1 & (edad>60 & edad<76)
	replace ae=.82 if hombre==1 & (edad>75)
	* Mujeres
	replace ae=.73 if hombre==0 & edad==10
	replace ae=.73 if hombre==0 & edad==11
	replace ae=.73 if hombre==0 & edad==12
	replace ae=.79 if hombre==0 & edad==13
	replace ae=.79 if hombre==0 & edad==14
	replace ae=.79 if hombre==0 & edad==15
	replace ae=.79 if hombre==0 & edad==16
	replace ae=.79 if hombre==0 & edad==17
	replace ae=.74 if hombre==0 & (edad>17 & edad<30)
	replace ae=.74 if hombre==0 & (edad>29 & edad<60)
	replace ae=.64 if hombre==0 & (edad>60 & edad<76)
	replace ae=.64 if hombre==0 & (edad>75)

* Total de adultos equivalentes del hogar
bys clave: egen tot_ae_hog = total(ae)

* Canastas por trimestre (No hay canastas por región en 2012)
gen canasta_min_pobre_12_t2 = 483
gen canasta_min_pobre_12_t3 = 503
gen canasta_min_pobre_12_t4 = 518
gen canasta_min_pobre_13_t1 = 531

* Ingreso per capita por adulto equivalente
gen ipcae = ingtoth/tot_ae_hog

* POBRE/NO POBRE
gen pobre_12_t2 = (ipcae<canasta_min_pobre_12_t2)
gen pobre_12_t3 = (ipcae<canasta_min_pobre_12_t3)
gen pobre_12_t4 = (ipcae<canasta_min_pobre_12_t4)
gen pobre_13_t1 = (ipcae<canasta_min_pobre_13_t1)

sum pobre*

kdensity ipcae if ipcae<40000, xline(483)
************************

* Usamos todas las canastas básicas del periodo
local CBT_12_t2 = 483
local CBT_12_t3 = 503
local CBT_12_t4 = 518
local CBT_13_t1 = 531

local lista_criterios = "CBT_12_t2 CBT_12_t3 CBT_12_t4 CBT_13_t1"


*/