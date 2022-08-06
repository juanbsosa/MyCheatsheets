
# Gráficos ----------------------------------------------------------------

### 1) GRAFICAR OBJETOS ESPACIALES EN UN MAPA

# """"

# Toma una cantidad variable (hasta 8 por la cantidad de colores) de objetos espaciales y devuelve un gráfico, con la opción de mostarlo en el Viewer y de exportarlo a un archivo .png.

# Paquetes necesarios: ggplot2, RColorBrewer, glue

# Argumentos:
# - 

# """"

# TO-DOs:
# - Que la cantidad de colores pueda ser infinita.
# - Que se pueda elegir el color que une quiera para cada objeto.
# - Ponerle relleno transparente a los polígonos, ahora está sin relleno.
# - Documentar los argumentos

plot_sp_objects <- function(..., plot_map=FALSE, export_png=TRUE, file, dpi=72, width=20, height=20, browser=FALSE){
    
    # Abrir el debugger si es requerido
    if(browser==TRUE){browser()}
    
    # Guardar en una lista los objetos pasados como argumentos 
    sp_object_list <- list(...)
    
    # Crear paletas de colores
    # para los puntos
    color_palette_points <- RColorBrewer::brewer.pal(n = length(sp_object_list), name = 'Set1')
    # para los polígonos
    color_palette_polygons <- rep(NA, length(sp_object_list)) # sin relleno
    #color_palette_polygons <- RColorBrewer::brewer.pal(n = length(sp_object_list), name = 'Pastel2') # relleno pastel, pero no tiene transparencia
    
    # Crear un grafico vacio
    map_plot <- ggplot2::ggplot()
    
    # para cada objeto espacial...
    for (i in 1:length(sp_object_list)){
        
        # capturar el tipo de geometría para luego definir el color y agregarla al gráfico...
        geometry_type <- as.character(sf::st_geometry_type(sp_object_list[[i]])[1])
        
        # si es un punto
        if (geometry_type=="POINT"){
            
            geometry_color <- color_palette_points[i]
            # agregarlo al gráfico
            map_plot <- map_plot + ggplot2::geom_sf( data = sp_object_list[[i]], color=geometry_color )
            
        }
        # si es un polígono
        if (geometry_type=="MULTIPOLYGON"){
            # definir el color
            geometry_color <- color_palette_polygons[i]
            # agregarlo al gráfico
            map_plot <- map_plot + ggplot2::geom_sf( data = sp_object_list[[i]], fill=geometry_color )
            
        }
        
        
    }
    
    # Exportar el mapa como un archivo .png
    if (export_png == TRUE){ggplot2::ggsave(filename = file, plot = map_plot, width=width, height=height, dpi=dpi)}
    
    # Mostrar el mapa
    if (plot_map==TRUE){print(map_plot)}
    
    # Devuelvo el mapa
    return(map_plot)
    
}
