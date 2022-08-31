#UDEMY - Intro to PYQGIS
#https://www.udemy.com/course/introduction-to-pyqgis/learn/lecture/17871182#overview
#Add a Vector Layer to the QGIS Interface

#Creo un filename para cada capa
fn_reclamos = 'C:\\Users\\juanb\\OneDrive\\Documentos\\Juan\\3F\\Informe de bacheo\\reclamos_2015-2021.shp'
fn_localidades = 'C:\\Users\\juanb\\OneDrive\\Documentos\\Juan\\3F\\Informe de bacheo\\localidades 3F.shp'

#Añado la capa
capa = iface.addVectorLayer(fn_localidades,'','ogr')
#Esto podría hacer con esa capa:
print(capa.capabilitiesString())

#Creo un objeto "capa", pero sin agregarla todavía
reclamos = QgsVectorLayer(fn_reclamos,'reclamos','ogr')


#Añado la capa
lista.addLayer(reclamos)

QgsProject.instance().removeMapLayers([reclamos.id()])

#Esto no me salió
#Pido la lista de todas las capas del archivo
#lista = QgsProject.instance().layerTreeRoot()
#Creo un objeto a eliminar con el nombre de la capa
#eliminar = QgsProject.instance().mapLayersByName('reclamos')[0]

#lista.removeLayer(eliminar)



