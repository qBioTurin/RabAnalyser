
source(system.file("R","Functions.R", package = "RabAnalyser"))
#source("files/Functions.R")
# for installing umap: install_version("RcppTOML", version = "0.1.3", repos = "http://cran.us.r-project.org")

server <- function(input, output, session) {
  observeEvent(input$NavBar, {
    if(input$NavBar == 2)
      shinyjs::show(id = "Sidebar")
  })

  docker_process <- reactiveVal(NULL)  # Store the process

  results  <- reactiveValues(DockerStep2 = "",
                             TableStep2 = NULL)  # Store the process

  Clustering_results <- reactiveValues(Data = NULL,
                                       FilteredData = NULL,
                                       clustering =NULL,
                                       violinPlot=NULL,
                                       features = NULL,
                                       umapPlot=NULL)

  ##### START: SC_KS_singlePopulation #####
  # Reactive value to store the selected folder
  wdFolders = stringr::str_split(string = dirname(getwd()),pattern = "/")
  wdFolders= paste0(wdFolders[[1]][1:3],collapse = "/")

  vols = RabAnalyser.getVolumes(exclude = "")
  shinyDirChoose(input, "folder", roots = vols,
                 session = session)

  folderPath <- reactive({
    if (is.null(input$folder) || (length(input$folder) == 1 && input$folder == 0) ) return(NULL)
    parseDirPath(roots = vols, input$folder)
  })

  output$folderPath <- renderText({ folderPath() })


  observeEvent(input$runStep2, {


    if( is.null(input$mat1$datapath) || is.null(input$mat2$datapath) ){
      shinyalert::shinyalert(
        title = "Error",
        text = "Both the mat files must be selected",
        type = "error")
      return()
    }

    if(is.null(folderPath())){
      shinyalert::shinyalert(
        title = "Error",
        text = "A folder path to save the results must be defined",
        type = "error")
      return()
    }


    req(input$mat1, input$mat2, folderPath())

    # Get the absolute file paths
    mat1_path <- input$mat1
    mat2_path <- input$mat2
    shared_folder <- folderPath()

    if(is.null(input$step2Feature) ){
      featureFile = system.file("Shiny/files","featurespython2.xlsx", package = "RabAnalyser")
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

    update_modal_progress(
      0.05,
      text = "loading data",
      session = shiny::getDefaultReactiveDomain()
    )

    # Data cleaning
    ctrl <- cleaning(ctrl)
    comp <- cleaning(comp)
    names(comp) =  colnames(ctrl) = c("id", features)

    update_modal_progress(
      0.1,
      text = "cleaning data",
      session = shiny::getDefaultReactiveDomain()
    )

    ecdf1List = lapply(features,function(n){
      Ref <- ctrl[, n]
      ecdf(Ref)
    })

    update_modal_progress(
      0.25,
      text = "First ecdf calculation",
      session = shiny::getDefaultReactiveDomain()
    )

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

    cl <- makeCluster(getOption("cl.cores", min(1,parallel::detectCores()-2) ))
    #clusterExport(cl, c( "two_sample_signed_ks_statistic","ctrl","Mcomp","ecdf1List"))
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

  ##### START: SC_Clustering #####

  observeEvent(input$Clustering_excel,{

    if(!is.null(Clustering_results$Data)){
    shinyalert::shinyalert(
      title = "New File Detected",
      text = "An existing analysis is already loaded. Do you want to replace it with the new upload?",
      type = "warning",
      showCancelButton = TRUE,
      confirmButtonText = "Yes, replace it",
      cancelButtonText = "Cancel",
      callbackR = function(response) {
        if (isTRUE(response)){
          # Reset structures
          Clustering_results$Data <- NULL
          Clustering_results$FilteredData <- NULL
          Clustering_results$clustering <- NULL
          Clustering_results$violinPlot <- NULL
          Clustering_results$features <- NULL
          Clustering_results$umapPlot <- NULL

          files = req(input$Clustering_excel$datapath)

          shinybusy::show_modal_spinner()

          initializeClusteringStep(session,input,output,Clustering_results,files)

          updateSelectInput(session =session, "Clustering_colorUMAPselect", choices = Clustering_results$features, selected = Clustering_results$features[1])
          shinybusy::remove_modal_spinner()
        }
      }
    )
    }else{
      files = req(input$Clustering_excel$datapath)

      shinybusy::show_modal_spinner()

      initializeClusteringStep(session,input,output,Clustering_results,files)

      shinybusy::remove_modal_spinner()
    }
  })

  observeEvent(input$Clustering_clusterSlider,{
    plotLIST = req(Clustering_results$clustering)
    slider= req(input$Clustering_clusterSlider)

    output$Clustering_clustChoicePlot <- renderPlot({
      plotLIST$silhouette +
        geom_vline(aes(xintercept = slider , color = "From slider" ), linetype = "dashed" )+
        geom_vline(aes(xintercept =  as.numeric(names(plotLIST$AllClusteringIndex$bestK[1])) ,
                       color = paste0("Best k considering ",
                                      plotLIST$AllClusteringIndex$bestK[1], " indexes over ",
                                      sum((plotLIST$AllClusteringIndex$bestK))
                       )
        ), linetype = "dashed"
        )+labs(col = "")
    })
  })

  observeEvent(input$Clustering_clusterStart,{
    shinybusy::show_modal_spinner()
    isolate({
      req(input$Clustering_clusterSlider!=0)
      req(Clustering_results$clustering)
      req(Clustering_results$features) -> features

      data = Clustering_results$clustering$Data

      set.seed(42)
      kmeans_result <- kmeans(data, centers = as.numeric(input$Clustering_clusterSlider), nstart = 25)
      data$cluster <- as.factor(kmeans_result$cluster)
      Clustering_results$clustering$dataClustered = data

      updateSelectInput(session =session, "Clustering_colorUMAPselect", choices = c("cluster", features), selected = "cluster")

      if(length(features[features == "Treatment"]) > 0 ) features = features[features != "Treatment"]
      updateSelectInput(session =session, "Clustering_ViolinColor", choices = c(features), selected = features[1])

    })
    shinybusy::remove_modal_spinner()
  })

  #### Violin Plot + statistics ####
  observe({
    input$Group_Clustering_violin -> groupTreat

    output$UI_statTables = renderUI({
      if(!is.null(groupTreat) && groupTreat){
        tagList(
          h3("Homogeneity of proportions inside each group"),
          DT::dataTableOutput("Clustering_Stattest"),
          h3("Homogeneity of proportions between groups"),
          DT::dataTableOutput("Clustering_GruopsStattest")
        )
      }else{
        tagList(
        h3("Statistical differences between groups"),
        DT::dataTableOutput("Clustering_Stattest")
        )
      }
      })
  })

  observe({
    req(input$Clustering_ViolinColor) -> feature
    input$Group_Clustering_violin -> groupTreat
    shinybusy::show_modal_spinner()
    ## ViolinPlot
    WholeData = cbind(req(Clustering_results$clustering$dataClustered),
                      req(Clustering_results$FilteredData) )

    isolate({
      #### STATISTICAL TEST ####
      # Perform pairwise Wilcoxon test (Mann-Whitney U test)

      if(!is.null(groupTreat) && groupTreat ){
        tab = WholeData %>% group_by(Treatment, cluster) %>%
          count() %>%
          spread(key = cluster, value = n) %>% ungroup() %>% as.data.frame()
        rownames(tab) = tab$Treatment

        xtab = as.matrix(tab %>% select(-Treatment))

        tabTot = WholeData %>% group_by(Treatment, cluster) %>%
          count()%>% group_by(cluster) %>%
          group_by(Treatment) %>%
          mutate(total = sum(n)) %>%
          ungroup() %>% as.data.frame()

        do.call(rbind, lapply(unique(WholeData$cluster), function(c){
          tabTot2=tabTot %>% filter(cluster ==c)%>% mutate(no = total - n )  %>% rename(yes = n) %>% select(-cluster,-Treatment, -total)
          xtab = as.matrix(tabTot2 )
          rownames(xtab) = tabTot %>% filter(cluster ==c) %>% pull(Treatment)
          if(dim(xtab)[1] >2){
            xtab = t(xtab)
            proptab = pairwise_prop_test(xtab)
          }else{
            proptab= prop_test(xtab, detailed = TRUE)
            proptab = proptab %>% select(cluster,n1,n2,p, p.signif,conf.low, conf.high )
              #rename(!!sym(rownames(xtab)[1]) = n1, !!sym(rownames(xtab)[2]) = n2)

            proptab$group1 =  rownames(xtab)[1]
            proptab$group2 =  rownames(xtab)[2]
          }
          proptab$cluster = c
          proptab
        }) ) -> proptab

        if(ncol(xtab)==2)
          xtab = t(xtab)

        do.call(rbind, lapply(unique(WholeData$Treatment), function(t){
          tabTot2=tabTot %>% filter(Treatment ==t)%>% mutate(no = total - n )  %>% rename(yes = n) %>% select(-cluster,-Treatment, -total)
          xtab = as.matrix(tabTot2 )
          rownames(xtab) = tabTot %>% filter(Treatment ==t) %>% pull(cluster)
          if(dim(xtab)[1] >2){
            xtab = t(xtab)
            proptab = pairwise_prop_test(xtab)
          }else{
            proptab= prop_test(xtab, detailed = TRUE)
            proptab = proptab %>% select(cluster,n1,n2,p, p.signif,conf.low, conf.high )
            #rename(!!sym(rownames(xtab)[1]) = n1, !!sym(rownames(xtab)[2]) = n2)

            proptab$group1 =  rownames(xtab)[1]
            proptab$group2 =  rownames(xtab)[2]
          }
          proptab$Treatment = t
          proptab
        }) ) -> prop_alongCluster

        output$Clustering_GruopsStattest <- DT::renderDT({
          prop_alongCluster
        },
        options = list(scrollX = TRUE, dom = 't'),
        rownames = FALSE)

        output$Clustering_Stattest <- DT::renderDT({
          proptab
        },
        options = list(scrollX = TRUE, dom = 't'),
        rownames = FALSE)

        max_y_by_cluster <- WholeData %>%
          group_by(cluster) %>%
          summarise(max_y = max(.data[[feature]], na.rm = TRUE))

        proptab
      }else{
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
          mutate(Median_Diff = abs(median1 - median2) )%>% # Compute absolute median difference
          mutate(across(where(is.numeric), round, 16))

        output$Clustering_Stattest <- DT::renderDT({
          test_results[, -1]
        },
        options = list(scrollX = TRUE, dom = 't'),
        rownames = FALSE)

        maxY <- max(WholeData[[feature]], na.rm = TRUE)
        stepY <- (maxY * 0.10)  # spacing between bars

        test_results <- test_results %>%
          mutate(y.position = maxY + stepY * (row_number()))

      }


      if(!is.null(groupTreat) && groupTreat){
        Clustering_results$violinPlot =
          ggplot(WholeData )+
          geom_point(aes(x = cluster,
                         y = !!sym(feature),
                         col=Treatment,
                         fill =Treatment),
                     position = position_jitterdodge(seed = 1, dodge.width = 0.9))+
          geom_violin(
            aes(x = cluster,
                y = !!sym(feature),
                col=Treatment,
                fill =Treatment), alpha = 0.5)+
          ggtitle(paste("Violin Plot of", feature , "Across Clusters and Treatments") )+
          scale_color_viridis_d(option = "plasma")+
          scale_fill_viridis_d(option = "plasma")+
          theme_minimal()
      }else{
        Clustering_results$violinPlot =
          ggplot(WholeData )+
          geom_jitter(
            aes(x = cluster,
                y = !!sym(feature),
                col=cluster,
                fill =cluster), width = 0.1)+
          geom_violin(
            aes(x = cluster,
                y = !!sym(feature),
                col=cluster,
                fill =cluster), alpha = 0.5) +
          ggtitle(paste("Violin Plot of", feature , "Across Clusters") )+
          scale_color_viridis_d()+
          scale_fill_viridis_d()+
          theme_minimal()+
          ggpubr::stat_pvalue_manual(data = test_results,
                                     label = "Significant",
                                     y.position = "y.position",
                                     xmin = "group1", xmax = "group2")
      }


      output$Clustering_ClusterViolinplot <- renderPlot({
        Clustering_results$violinPlot
      })

    })
    shinybusy::remove_modal_spinner()

  })

  #### FoldChangePlot ####
  observe({
    WholeData = cbind(req(Clustering_results$clustering$dataClustered),
                      req(Clustering_results$FilteredData) )

    shinybusy::show_modal_spinner()
    isolate({
      lapply(colnames(Clustering_results$FilteredData %>% select(-Treatment)), function(feature) {
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
          mutate(Fold_Change = abs(median1/median2) )

        SM_final <- test_results %>%
          mutate(Size = ifelse(p.adj <= 0.05, pmin(-log10(p.adj), 20), NA))

        SM_final$Color <- scales::col_numeric(
          palette = colorRampPalette(c("#997FD2", "#F3A341"))(100),
          domain = range(SM_final$Fold_Change, na.rm = TRUE)
        )(SM_final$Fold_Change)
        SM_final$Feature = feature

        SM_final$Cluster_pairs = paste0(SM_final$group1," vs ",SM_final$group2)
        SM_final
      }) -> StatList

      SM_final = do.call(rbind, StatList)

      Clustering_results$FoldChangePlot = ggplot(SM_final, aes(y = Cluster_pairs, x = Feature)) +
        geom_point(aes(size = Size, fill = Fold_Change),
                   shape = 21, stroke = 0.5, alpha = 0.75) +
        scale_size_continuous(range = c(5, 20)) +
        scale_fill_gradientn(colors = c("#997FD2", "#F3A341"), limits = c(0, max(SM_final$Fold_Change))) +
        labs(y = "Cluster Pairs", x = "Feature",
             fill = "Fold Change Median", size = "-log10(p.adj)",
             title = "Fold Change & p-value") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")

      output$Clustering_FoldChangePlot <- renderPlot({
        Clustering_results$FoldChangePlot
      })
    })
    shinybusy::remove_modal_spinner()

  })

  #### UMAP Plot ####
  observe({
    # I visualize in the UMAP space the features values. I use
    # only the features selected after feature selection

    input$Clustering_colorUMAPselect -> colorVar
    req(Clustering_results$clustering$Data)

    shinybusy::show_modal_spinner()
    isolate({
      if(!is.null(Clustering_results$clustering$dataClustered)){
        data =Clustering_results$clustering$dataClustered
        WholeData = cbind(data,Clustering_results$FilteredData)

        if(colorVar == "Treatment"){
          df = WholeData %>% select(Treatment, cluster ) %>%
            group_by(Treatment, cluster) %>% summarise( n = length(cluster)) %>%
            group_by(cluster) %>% mutate( Prop = n/sum(n))

          Clustering_results$proportionPlot = ggplot(df )+
            geom_bar(
              aes(x = cluster,
                  y = Prop,
                  col=Treatment,
                  fill =Treatment),stat = "identity",position = "dodge")+
            ggtitle(paste("Proportion of data Across Clusters and Treatments") )+
            scale_y_continuous(labels = scales::percent,limits = c(0,1))+
            scale_color_viridis_d(option = "plasma")+
            scale_fill_viridis_d(option = "plasma")+
            theme_minimal()+coord_flip()
        }else{
          df = WholeData %>% select(cluster)

          Clustering_results$proportionPlot = ggplot(df )+
            geom_bar(
              aes(x = "1", col= cluster,
                  fill =cluster),position = "fill")+
            scale_y_continuous(labels = scales::percent)+
            labs(y = "Prop", title = paste("Proportion of Data Across Clusters") )+
            scale_color_viridis_d()+
            scale_fill_viridis_d()+
            theme_minimal()+coord_flip()
        }
        output$Clustering_ProportionPlot<- renderPlot({
          Clustering_results$proportionPlot
        })

      }else{
        WholeData = cbind(Clustering_results$clustering$Data,Clustering_results$FilteredData)
      }

      if(is.null(Clustering_results$clustering$dataClustered) && colorVar == "cluster")
        colorVar = ""

      if(colorVar == "")
        gp = geom_point(alpha = 0.8)
      else
        gp = geom_point(aes(color = !!sym(colorVar) ), alpha = 0.8)

      if(colorVar == "cluster")
        colorscale = scale_color_viridis_d()
      else if(colorVar == "Treatment")
        colorscale = scale_color_viridis_d(option = "plasma")
      else
        colorscale = scale_colour_gradient2(mid = "white", low =  "blue",  high = "darkgreen",midpoint = 0)

      Clustering_results$umapPlot = ggplot(WholeData, aes(x = UMAP1, y = UMAP2) ) +
        gp +
        ggtitle("UMAP Data Visualization") +
        theme_minimal()+
        colorscale

      output$Clustering_coloredUMAPplot <- renderPlot({
        Clustering_results$umapPlot
      })
    })

    shinybusy::remove_modal_spinner()

  })

  observe({
    WholeData = cbind(req(Clustering_results$clustering$dataClustered),
                      req(Clustering_results$FilteredData) )

    if(!is.null(input$Group_Clustering_subpop) &&  input$Group_Clustering_subpop){
      df_cluster = WholeData %>% select(-UMAP1,-UMAP2) %>%
        tidyr::gather(-cluster,-Treatment, value = "KS", key = "Features" ) %>%
        mutate(Sign = if_else(KS>0,"KS > 0","KS < 0") ) %>%
        group_by(cluster,Treatment,Features, Sign) %>%
        summarise(Count = n() ) %>%
        group_by(cluster, Features,Treatment) %>%
        mutate(Proportion = Count / sum(Count)) %>%
        ungroup() %>%
        mutate(cluster = paste0("Cluster ", cluster))

      # Plot
      output$Clustering_SubPopPlot = renderPlot({
        ggplot(df_cluster, aes(x = Features, y = Proportion,  fill = Sign)) +
          geom_bar(position = "stack",stat = "identity") +
          scale_fill_manual(values = c("KS > 0" = "#2ca02c", "KS < 0" = "#1f77b4")) +
          labs(title = paste("Cluster - Feature-wise KS Sign Proportion"),
               x = "Features", y = "Proportion", fill = "Sign") +
          scale_y_continuous(labels = scales::percent)+
          theme_minimal() +
          facet_grid(Treatment~cluster) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
      })
    }else{
      df_cluster = WholeData %>% select(-UMAP1,-UMAP2,-Treatment) %>%
        tidyr::gather(-cluster, value = "KS", key = "Features" ) %>%
        mutate(Sign = if_else(KS>0,"KS > 0","KS < 0") ) %>%
        group_by(cluster,Features, Sign) %>%
        summarise(Count = n() ) %>%
        group_by(cluster, Features) %>%
        mutate(Proportion = Count / sum(Count)) %>%
        ungroup() %>%
        mutate(cluster = paste0("Cluster ", cluster))

      # Plot
      output$Clustering_SubPopPlot = renderPlot({
        ggplot(df_cluster, aes(x = Features, y = Proportion,  fill = Sign)) +
          geom_bar(position = "stack",stat = "identity") +
          scale_fill_manual(values = c("KS > 0" = "#2ca02c", "KS < 0" = "#1f77b4")) +
          labs(title = paste("Cluster - Feature-wise KS Sign Proportion"),
               x = "Features", y = "Proportion", fill = "Sign") +
          theme_minimal() +
          facet_grid(~cluster) +
          scale_y_continuous(labels = scales::percent)+
          theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
      })
    }

  })

  observe({
    ## Feature selection
    WholeData = cbind(req(Clustering_results$clustering$dataClustered),
                      req(Clustering_results$FilteredData) )

    X <- WholeData %>% select(-UMAP1, -UMAP2, -cluster, - Treatment)  # All except last
    y <- as.factor( WholeData %>% select(cluster) %>% pull() )  # Last column
    n_clusters = length(unique(y))

    shinybusy::show_modal_spinner()
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
        importance_df <- importance_df %>% mutate(Importance = MeanDecreaseGini/sum(MeanDecreaseGini)) %>% arrange(desc(MeanDecreaseGini))
        importance_df <- importance_df %>% gather(-MeanDecreaseAccuracy, -MeanDecreaseGini, -Feature, -Importance,key = "Cluster", value = "values")
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

        importance_df <- as.data.frame(feature_importance_matrix) %>% mutate(Cluster = rownames(feature_importance_matrix)) %>%
          tidyr::pivot_longer(-Cluster, names_to = "Feature", values_to = "Importance")

      }


      # Heatmap using ggplot
      ImportancePLot = ggplot(importance_df, aes(x = Feature, y = Cluster, fill = Importance)) +
        geom_tile(color = "white") +
        scale_fill_gradient(low = "white", high = "darkgreen",limits = c(0,1)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "bottom") +
        labs(title = "Feature Importance Heatmap",
             x = "Feature",
             y = "Cluster",
             fill = "Importance")

      output$Clustering_FeaturePlot = renderPlot({ImportancePLot})
    })
    shinybusy::remove_modal_spinner()

  })

  # Call these functions for each plot
  ##### UMAP Plot Edit #####

  ##### Violin Plot Edit #####
  observeEvent(input$EDIT_Clustering_colorUMAP, {
    req(Clustering_results$umapPlot)
    showPlotModal(session, "Clustering_coloredUMAPplot", Clustering_results$umapPlot)
  })
  generateDownloadHandler(session, input, output, "Clustering_coloredUMAPplot", Clustering_results$umapPlot)
  applyPlotChanges(session, input, output, "Clustering_coloredUMAPplot", Clustering_results$umapPlot)

  observeEvent(input$EDIT_Clustering_violin, {
    req(Clustering_results$violinPlot)
    showPlotModal(session, "Clustering_ClusterViolinplot", Clustering_results$violinPlot)
  })
  generateDownloadHandler(session, input, output, "Clustering_ClusterViolinplot", Clustering_results$violinPlot)
  applyPlotChanges(session, input, output, "Clustering_ClusterViolinplot", Clustering_results$violinPlot)

  ##### END Plot Edit #####

  ##### END: SC_Clustering #####
}
