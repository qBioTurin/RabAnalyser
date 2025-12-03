#' @title Run rabanalyser
#' @description function to run the rabanalyser shiny application.
#'
#' @param Null no paramters are needed
#'
#' @author  Pernice Simone, Tealdi Simone
#' @import shinyFiles shinyjs stringr readxl ggthemes dplyr openxlsx colourpicker reshape2  fs parallel shinybusy httpuv bslib writexl
#' @rawNamespace import(DT, except=c(dataTableOutput,renderDataTable))
#' @rawNamespace import(shiny,except=runExample)
#' @rawNamespace import(shinyWidgets,except=alert)
#' @rawNamespace import(randomForest,except=combine)
#' @rawNamespace import(ggplot2,except=margin)
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

  Appui <- system.file("Shiny","ui.R", package = "RabAnalyser")
  Appserver <- system.file("Shiny","server.R", package = "RabAnalyser")

  source(Appui)
  source(Appserver)

  app <-shinyApp(ui, server,
                 options =  options(shiny.maxRequestSize=1000*1024^2,
                                    shiny.launch.browser = .rs.invokeShinyWindowExternal)
  )

  app$staticPaths <- list(
    `/` = httpuv::staticPath(system.file("Shiny","www", package = "RabAnalyser"), indexhtml = FALSE, fallthrough = TRUE)
  )

  runApp(app)
  # runApp(
  #   appDir = system.file("Shiny" package = "RabAnalyser"),
  #   launch.browser = T
  # )
}
