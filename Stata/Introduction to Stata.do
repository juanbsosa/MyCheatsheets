* Introduction to Stata
* Copyright 2013 by Ani Katchova

* Creating a log file to store output
* log using stata_output.txt, text replace

clear all       
set more off    

* Change directory to folder with data files
cd C:/Econometrics/Data
*dir

* Importing data from the Internet
use http://www.ats.ucla.edu/stat/data/hs0, clear

* Importing csv file
insheet using intro_hs0.csv, clear

* Reading Stata (.dta) file
use intro_hs0, clear

* Summarizing the data
describe
summarize
list gender-read in 1/10
summarize read math science write
summarize if read>=60
summarize if prgtype=="academic"
summarize read, detail

* Summarizing the data by group
tab prgtype 
bysort prgtype: summarize read write
tabstat read write math, by(prgtype) stat(n mean sd)

* Correlations
correlate write read science

* Modifying the data
order id gender
label variable schtyp "type of school"
rename gender female
gen score=read+write+math
gen score2=score^2
gen pass=1 if score>=150
replace pass=0 if pass==.
drop if read<40
drop schtyp

* Creating dummy variables
sort prgtype
xi, prefix() i.prgtype

*Generating variables using functions
egen avgscore=mean(score)
egen avggroupscore=mean(score), by(prgtype)

* t-tests
ttest write=50
ttest write=read
ttest write, by(female)

* Regression
reg write read female

* Regression with dummy variables
xi: reg write read female i.prgtype

* Defining global variables
global ylist write
global xlist read female

* Using global variables 
summarize $ylist $xlist
reg $ylist $xlist

* Using ado files - download "outreg" for paper-ready tables
*outreg using regression_output.txt, replace

* log close

