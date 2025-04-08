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
                                        clustering =NULL,
                                        violinPlot=NULL,
                                        umapPlot=NULL)

  ##### START: SC_KS_singlePopulation #####
  # Reactive value to store the selected folder
  wdFolders = stringr::str_split(string = dirname(getwd()),pattern = "/")
  wdFolders= paste0(wdFolders[[1]][1:3],collapse = "/")

  vols = RabAnalyser.getVolumes(exclude = "")
  shinyDirChoose(input, "folder", roots = vols,
                 session = session)

  folderPath <- reactive({
    if (is.null(input$folder)) return(NULL)
    parseDirPath(roots = vols, input$folder)
  })

  output$folderPath <- renderText({ folderPath() })


  observeEvent(input$runStep2, {
    req(input$mat1, input$mat2, folderPath())

    # Get the absolute file paths
    mat1_path <- input$mat1
    mat2_path <- input$mat2
    shared_folder <- folderPath()

    if(is.null(input$step2Feature) ){
      featureFile = system.file("Docker/PythonScripts","featurespython2.xlsx", package = "RabAnalyser")
      features <- names(read_excel(featureFile))
    }else{
      features = input$step2Feature
    }

    shinybusy::show_modal_progress_circle()
    # Loading data
    CTRLmat_data <- readMat(mat1_path$datapath)
    ctrl <- CTRLmat_data$ClustersB

    COMPmat_data <- readMat(mat2_path$datapath)
    comp <- COMPmat_data$ClustersB

    # Data cleaning
    ctrl <- cleaning(ctrl)
    comp <- cleaning(comp)
    names(comp) =  colnames(ctrl) = c("id", features)

    ecdf1List = lapply(features,function(n){
      Ref <- ctrl[, n]
      ecdf(Ref)
    })

    names(ecdf1List) = features
    # KS Matrix Generation
    Mcomp <- max(comp[,1])
    comp_groups <- lapply( unique(comp[,1]), function(id) comp[comp[,1] == id,] )
    names(comp_groups) <- unique(comp[,1])

    compute_ks <- function(i) {
      KSv <- sapply(features, function(n,ii) {
        compN <- comp_groups[[as.character(ii)]][, n]
        Ref <- ctrl[, n]
        KS <- two_sample_signed_ks_statistic(Ref, compN, ecdf1List[[n]])
        return(KS[2])
      },ii=i)
      cat(sprintf("Progress: %f %%\n", i/Mcomp))
      return(KSv)
    }

    cl <- makeCluster(getOption("cl.cores", detectCores()-1))
    clusterExport(cl, c("comp_groups", "two_sample_signed_ks_statistic","ctrl","Mcomp","ecdf1List"))
    results <- clusterApply(cl, 1:Mcomp, compute_ks )
    stopCluster(cl)
    KScontrol <- do.call(cbind, results)

    # Dataframe creation
    df <- as.data.frame(t(KScontrol))
    names(df) <- features
    results$TableStep2 <- df

    output$TableStep2 = DT::renderDataTable(
      DT::datatable(df,
                    filter = 'top', rownames = FALSE, editable = F,
                    options = list(searching = FALSE, info = FALSE,paging = FALSE,
                                   sort = TRUE, scrollX = TRUE, scrollY = TRUE)
      )
    )

    if(input$step2filename != ""){
      filename = sub(pattern = "(.*)\\..*$", replacement = "\\1", input$step2filename)
      write_xlsx(df, paste0(shared_folder,"/",filename,".xlsx") )
    }else{
      write_xlsx(df, paste0(shared_folder,"/SC_KS_singlePopulation_Data.xlsx") )
    }

    shinybusy::remove_modal_progress_circle()
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

    features = colnames(singleCNCTN_results$FilteredData)
    updateSelectInput(session =session, "singleCNCTN_colorUMAPselect", choices = c( features), selected = features[1])

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
    scaled_data <- scale(reduced_df, center = TRUE, scale = apply(reduced_df, 2, sd) * sqrt((nrow(reduced_df)-1)/nrow(reduced_df)))

    set.seed(42)
    reducer <- umap(scaled_data, n_neighbors = 10, min_dist = 0.5, n_components = 2)
    umap_df <- as.data.frame(reducer$layout)
    names(umap_df) <- c("UMAP1", "UMAP2")

    #saveRDS(umap_df,file = "umap_df.RDs")
    ### cluster SCOREs ###
    singleCNCTN_results$clustering <- plotLIST <- cluster.generation(data = umap_df)

    # Display results

    k = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1]))

    output$singleCNCTN_clustChoicePlot <- renderPlot({
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
    output$singleCNCTN_ClusterIndexesTable = renderTable(bestk)
    updateSliderInput(session,"singleCNCTN_clusterSlider",value = k,min =2 ,max = 10)

    shinybusy::remove_modal_spinner()
  })

  observeEvent(input$singleCNCTN_clusterSlider,{
    plotLIST = req(singleCNCTN_results$clustering)
    slider= req(input$singleCNCTN_clusterSlider)
    k = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1]))
    output$singleCNCTN_clustChoicePlot <- renderPlot({
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

  observeEvent(input$singleCNCTN_clusterStart,{
    shinybusy::show_modal_spinner()
    isolate({
      req(input$singleCNCTN_clusterSlider!=0)
      req(singleCNCTN_results$clustering)
      data = singleCNCTN_results$clustering$Data

      set.seed(42)
      kmeans_result <- kmeans(data, centers = as.numeric(input$singleCNCTN_clusterSlider), nstart = 25)
      data$cluster <- as.factor(kmeans_result$cluster)
      singleCNCTN_results$clustering$dataClustered =data

      features = colnames(singleCNCTN_results$FilteredData)
      updateSelectInput(session =session, "singleCNCTN_colorUMAPselect", choices = c("cluster", features), selected = "cluster")
      updateSelectInput(session =session, "singleCNCTN_ViolinColor", choices = c( features), selected = features[1])
    })
    shinybusy::remove_modal_spinner()
  })

  observe({
    req(input$singleCNCTN_ViolinColor) -> feature

    ## ViolinPlot
    WholeData = cbind(req(singleCNCTN_results$clustering$dataClustered),
                      req(singleCNCTN_results$FilteredData) )
    isolate({
      #### STATISTICAL TEST ####
      # Perform pairwise Wilcoxon test (Mann-Whitney U test)
      test_results <- WholeData %>%
        rstatix::wilcox_test(as.formula(paste(feature, "~ cluster"))) %>%
        rstatix::adjust_pvalue(method = "bonferroni") %>% # Adjust for multiple testing
        mutate(Significant = ifelse(p.adj < 0.05, "*", "ns"))  # Mark significance

      # Compute median differences for each pair
      cluster_medians <- WholeData %>%
        group_by(cluster) %>%
        summarise(median_val = median(!!sym(feature)))

      test_results <- test_results %>%
        left_join(cluster_medians, by = c("group1" = "cluster")) %>%
        rename(median1 = median_val) %>%
        left_join(cluster_medians, by = c("group2" = "cluster")) %>%
        rename(median2 = median_val) %>%
        mutate(Median_Diff = abs(median1 - median2))  # Compute absolute median difference

      output$singleCNCTN_Stattest <- renderTable({
        test_results[,-1]
      })

      singleCNCTN_results$violinPlot =
        ggplot(WholeData )+
        geom_jitter(
          aes(x = cluster,
              y = !!sym(input$singleCNCTN_ViolinColor),
              col=cluster,
              fill =cluster), width = 0.1)+
        geom_violin(
          aes(x = cluster,
              y = !!sym(input$singleCNCTN_ViolinColor),
              col=cluster,
              fill =cluster), alpha = 0.5) +
        ggtitle("UMAP Data Visualization") +
        scale_color_viridis_d()+
        scale_fill_viridis_d()+
        theme_minimal()

      output$singleCNCTN_ClusterViolinplot <- renderPlot({
        singleCNCTN_results$violinPlot
      })
    })

  })

  observe({
    # I visualize in the UMAP space the features values. I use
    # only the features selected after feature selection

    input$singleCNCTN_colorUMAPselect -> colorVar
    req(singleCNCTN_results$clustering$Data)

    if(!is.null(singleCNCTN_results$clustering$dataClustered)){
      data =singleCNCTN_results$clustering$dataClustered
      WholeData = cbind(data,singleCNCTN_results$FilteredData)
    }else{
      WholeData = cbind(singleCNCTN_results$clustering$Data,singleCNCTN_results$FilteredData)
    }

    if(colorVar == "")
      gp = geom_point(alpha = 0.8)
    else
      gp = geom_point(aes(color = !!sym(colorVar) ), alpha = 0.8)

    if(colorVar == "cluster")
      colorscale = scale_color_viridis_d()
    else
      colorscale = scale_colour_gradient2(mid = "white", low =  "blue",  high = "darkgreen",midpoint = 0)

    singleCNCTN_results$umapPlot = ggplot(WholeData, aes(x = UMAP1, y = UMAP2) ) +
      gp +
      ggtitle("UMAP Data Visualization") +
      theme_minimal()+
      colorscale

    output$singleCNCTN_coloredUMAPplot <- renderPlot({
      singleCNCTN_results$umapPlot
    })
  })

  observe({
    WholeData = cbind(req(singleCNCTN_results$clustering$dataClustered),
                      req(singleCNCTN_results$FilteredData) )

    df_cluster = WholeData %>% select(-UMAP1,-UMAP2) %>%
      tidyr::gather(-cluster, value = "KS", key = "Features" ) %>%
      mutate(Sign = if_else(KS>0,"KS > 0","KS < 0") ) %>%
      group_by(cluster,Features, Sign) %>%
      summarise(Count = n() ) %>%
      group_by(cluster, Features) %>%
      mutate(Proportion = Count / sum(Count)) %>%
      ungroup() %>%
      mutate(cluster = paste0("Cluster ", cluster))

    # Plot
    output$singleCNCTN_SubPopPlot = renderPlot({
      ggplot(df_cluster, aes(x = Features, y = Proportion,  fill = Sign)) +
        geom_bar(position = "stack",stat = "identity") +
        scale_fill_manual(values = c("KS > 0" = "#2ca02c", "KS < 0" = "#1f77b4")) +
        labs(title = paste("Cluster - Feature-wise KS Sign Proportion"),
             x = "Features", y = "Proportion", fill = "Sign") +
        theme_minimal() +
        facet_grid(~cluster) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
    })
  })

  observe({
    ## Feature selection
    WholeData = cbind(req(singleCNCTN_results$clustering$dataClustered),
                      req(singleCNCTN_results$FilteredData) )

    X <- WholeData %>% select(-UMAP1, -UMAP2, -cluster)  # All except last
    y <- as.factor( WholeData %>% select(cluster) %>% pull() )  # Last column
    n_clusters = length(unique(y))

    isolate({
      if (n_clusters <= 2) {
        # Train/test split
        set.seed(42)
        train_index <- createDataPartition(y, p = 0.7, list = FALSE)
        X_train <- X[train_index, ]
        y_train <- y[train_index]
        X_test <- X[-train_index, ]
        y_test <- y[-train_index]

        # Train Random Forest
        rf_model <- randomForest(x = X_train, y = y_train, ntree = 100)
        predictions <- predict(rf_model, X_test)

        # Evaluation
        accuracy <- sum(predictions == y_test) / length(y_test)
        print(paste("Accuracy on Test Set:", round(accuracy, 4)))
        print(confusionMatrix(predictions, y_test))

        # Feature Importance (Full Dataset)
        rf_full <- randomForest(x = X, y = y, ntree = 100, importance=TRUE)
        importance_df <- as.data.frame(importance(rf_full))
        importance_df$Feature <- rownames(importance_df)
        importance_df <- importance_df %>% mutate(MeanDecreaseGini = MeanDecreaseGini/sum(MeanDecreaseGini)) %>% arrange(desc(MeanDecreaseGini))

        # Plot
        ImportancePLot = ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
          geom_bar(stat = "identity", fill = "darkgreen") +
          coord_flip() +
          labs(x = "Feature", y = "Feature importance (Decreasing Gini Index)") +
          theme_minimal()+
          lims(y = c(0,1) )

      } else {
        # One-vs-all Feature Importance Heatmap
        classes <- levels(y)
        feature_importance_matrix <- matrix(0, nrow = length(classes), ncol = ncol(X))
        colnames(feature_importance_matrix) <- colnames(X)
        rownames(feature_importance_matrix) <- paste0("Cluster ", classes)

        for (i in seq_along(classes)) {
          y_binary <- ifelse(y == classes[i], 1, 0)
          rf_model <- randomForest(x = X, y = as.factor(y_binary), ntree = 100)
          feature_importance_matrix[i, ] <- rf_model$importance[, "MeanDecreaseGini"]/sum(rf_model$importance[, "MeanDecreaseGini"])
        }

        df_long <- as.data.frame(feature_importance_matrix) %>% mutate(Cluster = rownames(feature_importance_matrix)) %>%
          tidyr::pivot_longer(-Cluster, names_to = "Feature", values_to = "Importance")

        # Heatmap using ggplot
        ImportancePLot = ggplot(df_long, aes(x = Feature, y = Cluster, fill = Importance)) +
          geom_tile(color = "white") +
          scale_fill_gradient(low = "white", high = "darkgreen",limits = c(0,1)) +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1),
                legend.position = "bottom") +
          labs(title = "Feature Importance Heatmap",
               x = "Feature",
               y = "Cluster",
               fill = "Importance")

      }

      output$singleCNCTN_FeaturePlot = renderPlot({ImportancePLot})
    })

  })
  # Call these functions for each plot
  ##### UMAP Plot Edit #####

  ##### Violin Plot Edit #####
  observeEvent(input$EDIT_singleCNCTN_colorUMAP, {
    req(singleCNCTN_results$umapPlot)
    showPlotModal(session, "singleCNCTN_coloredUMAPplot", singleCNCTN_results$umapPlot)
  })
  generateDownloadHandler(session, input, output, "singleCNCTN_coloredUMAPplot", singleCNCTN_results$umapPlot)
  applyPlotChanges(session, input, output, "singleCNCTN_coloredUMAPplot", singleCNCTN_results$umapPlot)

  observeEvent(input$EDIT_singleCNCTN_violin, {
    req(singleCNCTN_results$violinPlot)
    showPlotModal(session, "singleCNCTN_ClusterViolinplot", singleCNCTN_results$violinPlot)
  })
  generateDownloadHandler(session, input, output, "singleCNCTN_ClusterViolinplot", singleCNCTN_results$violinPlot)
  applyPlotChanges(session, input, output, "singleCNCTN_ClusterViolinplot", singleCNCTN_results$violinPlot)

  ##### END Plot Edit #####

  ##### END: SC_singleCNCTN #####
}
