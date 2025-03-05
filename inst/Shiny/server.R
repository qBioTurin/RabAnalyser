source(system.file("R","Functions.R", package = "RabAnalyser"))

# for installing umap: install_version("RcppTOML", version = "0.1.3", repos = "http://cran.us.r-project.org")

server <- function(input, output, session) {
  observeEvent(input$NavBar, {
    if(input$NavBar == 2)
      shinyjs::show(id = "Sidebar")
  })

  docker_process <- reactiveVal(NULL)  # Store the process

  results  <- reactiveValues(DockerStep2 = "",
                             TableStep2 = NULL)  # Store the process

  singleCNCTN_results <- reactiveValues(Data = NULL,
                                        FilteredData = NULL,
                                        clustering =NULL)
  ##### START: SC_KS_singlePopulation #####
  # Reactive value to store the selected folder
  wdFolders = stringr::str_split(string = dirname(getwd()),pattern = "/")
  wdFolders= paste0(wdFolders[[1]][1:3],collapse = "/")

  shinyDirChoose(input, "folder", roots = c(wd = wdFolders), session = session)

  folderPath <- reactive({
    if (is.null(input$folder)) return(NULL)
    parseDirPath(roots = c(wd = wdFolders), input$folder)
  })

  output$folderPath <- renderText({ folderPath() })


  observeEvent(input$runDocker, {
    req(input$mat1, input$mat2, folderPath())

    # Get the absolute file paths
    mat1_path <- input$mat1
    mat2_path <- input$mat2
    shared_folder <- folderPath()

    file.copy(from = mat1_path$datapath, to = shared_folder,overwrite = T)
    file.rename(from = paste0(shared_folder,"/", basename( mat1_path$datapath)),
                to = paste0(shared_folder,"/", basename( mat1_path$name)) )

    file.copy(from = mat2_path$datapath, to = shared_folder,overwrite = T)
    file.rename(from = paste0(shared_folder,"/", basename( mat2_path$datapath)),
                to = paste0(shared_folder,"/", basename( mat2_path$name)) )


    if("" %in% input$step2Feature){
      featureFile = system.file("Docker/PythonScripts","featurespython2.xlsx", package = "RabAnalyser")
      features <- read_excel(featureFile)
    }else{
      features = input$step2Feature
    }

    openxlsx::write.xlsx(features,
                     file = paste0(shared_folder,"/featurespython2.xlsx")
                     )

    # Construct the Docker command

    output$dockerLine <- renderPrint({
      cat(
        sprintf(
          "docker run \n--rm -v '%s:/Data'\nspernice/rabanalyser python PythonScripts/Parallel_SC_KS_singlePopulation.py \n/Data/%s \n/Data/%s",
          shared_folder, mat1_path$name, mat2_path$name
        )
      )
    })

    docker_cmd <- c(
      "run", "--rm",
      "-v", paste0(shared_folder, ":/Data"),  # Mount shared volume
      "spernice/rabanalyser",  # Docker image
      "python", "PythonScripts/Parallel_SC_KS_singlePopulation.py",  # Python script to run
      paste0("/Data/", mat1_path$name ),
      paste0("/Data/", mat2_path$name )
    )

    # Start Docker process in background
    p <- process$new(
      "docker", args = docker_cmd,
      stdout = "|", stderr = "|", cleanup = TRUE
    )

    docker_process(p)  # Store process in reactive variable
    updateActionButton(session =session, "runDocker",disabled = T )

  })

  # Poll for logs
  observe({
    invalidateLater(1000)  # Check every second
    p<-req(docker_process())

    isolate({
      if (!is.null(p) && p$is_alive()) {
        restext =  paste0("\n",p$read_output(),"\n",p$read_error(), "\n")
        if(restext == "\n") restext = "."
        results$DockerStep2= paste0( results$DockerStep2, restext)
        output$dockerOutput <- renderPrint({
          cat(results$DockerStep2)
        })
      } else {
        output$dockerOutput <- renderPrint({
          cat("Process completed!\n", results$DockerStep2 , sep = "\n")
        })

        shared_folder <- folderPath()

        resTable.path = paste0(shared_folder,"/KS_singlePopulation.xlsx")
        if(file.exists(resTable.path)){
          results$TableStep2 <- resTable <- readxl::read_excel(resTable.path)

          output$TableStep2 = DT::renderDataTable(
            DT::datatable(resTable,
                          filter = 'top', rownames = FALSE, editable = F,
                          options = list(searching = FALSE, info = FALSE,paging = FALSE,
                                         sort = TRUE, scrollX = TRUE, scrollY = TRUE)
            )
          )

          if(input$step2filename != ""){
            filename = sub(pattern = "(.*)\\..*$", replacement = "\\1", input$step2filename)
            file.rename(from = resTable.path, to = paste0(shared_folder,"/",filename,".xlsx") )
          }
        }
        updateActionButton(session =session, "runDocker",disabled = F )
        results$DockerStep2 = ""
        docker_process(NULL)
      }
    })
  })

  ##### END: SC_KS_singlePopulation #####

  ##### START: SC_singleCNCTN #####

  observeEvent(input$singleCNCTN_excel,{

    file = req(input$singleCNCTN_excel$datapath)
    df <- read_excel(file)

    shinybusy::show_modal_spinner()
    # Feature Selection
    threshold <- 0.75
    reduced_df <- FilterFeat(df, threshold)

    singleCNCTN_results$Data = df
    singleCNCTN_results$FilteredData = reduced_df

    # Correlation Matrix Plot (Before Filtering)
    correlation_matrixOLD <- cor(df, use = "pairwise.complete.obs")

    melted_corr <- melt(correlation_matrixOLD)

    output$CorrMatrixBeforeFiltering = renderPlot({
      ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
        theme_minimal() + labs(x = "", y= "")+
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Correlation Matrix of Original Features")
    })

    # Correlation Matrix Plot (After Filtering)
    correlation_matrixNEW <- cor(reduced_df, use = "pairwise.complete.obs")
    melted_corr_new <- melt(correlation_matrixNEW)

    output$CorrMatrixAfterFiltering = renderPlot({
      ggplot(melted_corr_new, aes(Var1, Var2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
      theme_minimal() + labs(x = "", y= "")+
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      ggtitle("Correlation Matrix of Filtered Features")
    })

    # UMAP Analysis
    scaled_data <- scale(reduced_df)
    reducer <- umap(scaled_data, n_neighbors = 10, min_dist = 0.5, n_components = 2)
    umap_df <- as.data.frame(reducer$layout)
    names(umap_df) <- c("UMAP1", "UMAP2")

    ### cluster SCOREs ###
    singleCNCTN_results$clustering <- plotLIST <- cluster.generation(data = umap_df)

    # Display results

    k = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1]))

    output$clustChoicePlot <- renderPlot({
      plotLIST$silhouette +
        geom_vline(aes(xintercept =k ,
                       color = paste0("Best k considering ",
                                      plotLIST$AllClusteringIndex$bestK[1], " indexes over ",
                                      sum((plotLIST$AllClusteringIndex$bestK)))
        ), linetype = "dashed"
        )+labs(col = "")
    })

    bestk = as.data.frame(plotLIST$AllClusteringIndex$bestK)
    colnames(bestk) = c("Number of\n clusters", "Number of indexes\n in accordance")
    output$ClusterIndexesTable = renderTable(bestk)
    updateSliderInput(session,"clusterSlider",value = k,min =2 ,max = 10)

    shinybusy::remove_modal_spinner()
  })

  observeEvent(input$clusterSlider,{
    plotLIST = req(singleCNCTN_results$clustering)
    slider= req(input$clusterSlider)
    k = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1]))
    output$clustChoicePlot <- renderPlot({
      plotLIST$silhouette +
        geom_vline(aes(xintercept = slider , color = "From slider" ), linetype = "dashed" )+
        geom_vline(aes(xintercept = k ,
                       color = paste0("Best k considering ",
                                      plotLIST$AllClusteringIndex$bestK[1], " indexes over ",
                                      sum((plotLIST$AllClusteringIndex$bestK))
                                      )
        ), linetype = "dashed"
        )+labs(col = "")
    })
  })

  observeEvent(input$clusterStart,{
    shinybusy::show_modal_spinner()
    isolate({
      req(input$clusterSlider!=0)
      req(singleCNCTN_results$clustering)
      data = singleCNCTN_results$clustering$Data

      kmeans_result <- kmeans(data, centers = as.numeric(input$clusterSlider), nstart = 25)
      data$cluster <- as.factor(kmeans_result$cluster)
      singleCNCTN_results$clustering$dataClustered =data

      features = colnames(singleCNCTN_results$FilteredData)
      updateSelectInput(session =session, "colorUMAPselect", choices = c("cluster", features), selected = "cluster")

    })
    shinybusy::remove_modal_spinner()
  })

  observeEvent(input$colorUMAPselect,{
    # I visualize in the UMAP space the features values. I use
    # only the features selected after feature selection

    input$colorUMAPselect -> colorVar

    data =req(singleCNCTN_results$clustering$dataClustered)

    WholeData = cbind(data,singleCNCTN_results$FilteredData)

    output$coloredUMAPplot <- renderPlot({
      ggplot(WholeData, aes(x = UMAP1, y = UMAP2) ) +
        ifelse(colorVar == "",
               geom_point(alpha = 0.8) ,
               geom_point(aes(color = !!sym(colorVar)), alpha = 0.8)) +
        ggtitle("UMAP Data Visualization") +
        theme_minimal()
    })


  })

  ##### END: SC_singleCNCTN #####
}
