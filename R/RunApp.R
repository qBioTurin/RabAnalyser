#' @title Run rabanalyser
#' @description function to run the rabanalyser shiny application.
#'
#' @param Null no paramters are needed
#'
#' @author  Pernice Simone Tealdi Simone
#' @import shinyFiles shinyjs stringr readxl ggplot2 ggthemes factoextra dplyr openxlsx uwot reshape2 rstatix fs parallel shinybusy randomForest caret pheatmap clusterCrit bslib writexl
#' @rawNamespace import(DT, except=c(dataTableOutput,renderDataTable))
#' @rawNamespace import(shiny,except=runExample)
#' @rawNamespace import(shinyWidgets,except=alert)
#' @rawNamespace import(xfun,except=c(file_exists,dir_create,dir_exists))
#'
#' @examples
#'\dontrun{
#' rabanalyser.run()
#' }
#' @export

rabanalyser.run <-function()
{
  x = T

  Appui <- system.file("Shiny","ui.R", package = "rabanalyser")
  Appserver <- system.file("Shiny","server.R", package = "rabanalyser")

  source(Appui)
  source(Appserver)

  app <-shinyApp(ui, server,
                 options =  options(shiny.maxRequestSize=1000*1024^2,
                                    shiny.launch.browser = .rs.invokeShinyWindowExternal)
  )

  app$staticPaths <- list(
    `/` = httpuv::staticPath(system.file("Shiny","www", package = "rabanalyser"), indexhtml = FALSE, fallthrough = TRUE)
  )

  runApp(app)
  # runApp(
  #   appDir = system.file("Shiny" package = "rabanalyser"),
  #   launch.browser = T
  # )
}
