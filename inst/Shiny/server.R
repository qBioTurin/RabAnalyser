server <- function(input, output, session) {
  
  # Reactive values to store data across workflow steps
  rv <- reactiveValues(
    features_data = NULL,
    ks_results = NULL,
    filtered_data = NULL,
    corr_result = NULL,
    clustering_results = NULL,
    umap_df = NULL,
    df_features = NULL,
    feature_names = NULL,
    input_folder_selected = NULL
  )
  
  # ========== STEP 1: FEATURE EXTRACTION ==========
  # Console log for feature extraction
  rv$feature_console_log <- character(0)
  output$feature_console <- renderText({
    paste(rv$feature_console_log, collapse = "\n")
  })

  # Handle folder selection
  shinyDirChoose(input, "input_folder", roots = c(home = "~"))

  observe({
    if (!is.null(input$input_folder)) {
      rv$input_folder_selected <- parseDirPath(roots = c(home = "~"), input$input_folder)
      output$input_folder_path <- renderText({
        if (length(rv$input_folder_selected) > 0) {
          paste("Selected:", rv$input_folder_selected)
        } else {
          "No folder selected"
        }
      })
    }
  })

  # Run feature extraction
  observeEvent(input$run_extraction, {
    req(rv$input_folder_selected)
    output$extraction_status <- renderText("Running feature extraction...")
    rv$feature_console_log <- c("[INFO] Starting feature extraction...", rv$feature_console_log)
    output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
    tryCatch({
      rv$feature_console_log <- c("[INFO] Running RabAnalyser::extract_features()...", rv$feature_console_log)
      output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
      # Stream output in real time
      rv$features_data <- NULL
      withCallingHandlers({
        rv$features_data <- RabAnalyser::extract_features(
          rv$input_folder_selected,
          min_spot_size = input$min_spot_size,
          neighbor_radius = input$neighbor_radius,
          n_jobs = input$n_jobs,
          spot_folder = input$spot_folder,
          nucleus_folder = input$nucleus_folder,
          cell_folder = input$cell_folder,
          rab_folder = input$rab_folder
        )
      },
      message = function(m) {
        isolate({
          rv$feature_console_log <- c(paste0("[MSG] ", m$message), rv$feature_console_log)
          output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
        })
      },
      warning = function(w) {
        isolate({
          rv$feature_console_log <- c(paste0("[WARN] ", w$message), rv$feature_console_log)
          output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
        })
      },
      error = function(e) {
        isolate({
          rv$feature_console_log <- c(paste0("[ERROR] ", e$message), rv$feature_console_log)
          output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
        })
      })
      rv$feature_console_log <- c("[SUCCESS] Feature extraction completed!", rv$feature_console_log)
      output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
      output$extraction_status <- renderText("Feature extraction completed successfully!")
    }, error = function(e) {
      rv$feature_console_log <- c(paste("[ERROR]", e$message), rv$feature_console_log)
      output$feature_console <- renderText({ paste(rv$feature_console_log, collapse = "\n") })
      output$extraction_status <- renderText(paste("Error:", e$message))
    })
  })
  
  # ========== STEP 2: KS ANALYSIS ==========
  
  observeEvent(input$run_ks, {
    req(input$reference_file, input$comparison_files)
    
    output$ks_status <- renderText("Running KS analysis...")
    
    tryCatch({
      # Load reference population
      reference_data <- read_excel(input$reference_file$datapath)
      reference_data$ID_image <- 1
      
      # Rename cell label column if needed
      if (input$id_column %in% colnames(reference_data)) {
        reference_data <- reference_data %>% rename(Cell_label = !!sym(input$id_column))
      }
      
      # Process each comparison file
      ks_results_list <- list()
      
      for (i in seq_along(input$comparison_files$datapath)) {
        comp_data <- read_csv(input$comparison_files$datapath[i])
        features <- colnames(comp_data)[-(1:2)]
        
        ks_result <- RabAnalyser::perform_ks_analysis(
          ctrl_matrix = reference_data,
          comp_matrix = comp_data,
          features = features,
          cores = input$ks_cores
        )
        
        ks_result$file_name <- input$comparison_files$name[i]
        ks_results_list[[i]] <- ks_result
      }
      
      rv$ks_results <- bind_rows(ks_results_list)
      
      output$ks_results_table <- DT::renderDataTable({
        DT::datatable(rv$ks_results, options = list(scrollX = TRUE, pageLength = 10))
      })
      
      output$ks_status <- renderText("KS analysis completed successfully!")
    }, error = function(e) {
      output$ks_status <- renderText(paste("Error:", e$message))
    })
  })
  
  # ========== STEP 3: FEATURE FILTERING ==========
  
  observeEvent(input$run_filtering, {
    req(input$clustering_input)
    
    output$filtering_status <- renderText("Loading data and filtering features...")
    
    tryCatch({
      # Load data
      if (grepl("\\.xlsx?$", input$clustering_input$name)) {
        df <- read_excel(input$clustering_input$datapath)
      } else {
        df <- read_csv(input$clustering_input$datapath)
      }
      
      # Filter correlated features
      rv$corr_result <- RabAnalyser::filter_correlated_features(
        df, 
        threshold = input$corr_threshold
      )
      
      rv$filtered_data <- rv$corr_result$filtered_data
      rv$feature_names <- colnames(rv$filtered_data)[colnames(rv$filtered_data) != "Class"]
      
      # Plot original correlation matrix
      output$corr_original <- renderPlot({
        RabAnalyser::plot_correlation_matrix(
          rv$corr_result$corr_original,
          title = "Correlation Matrix - Original Features"
        )
      })
      
      # Plot filtered correlation matrix
      output$corr_filtered <- renderPlot({
        RabAnalyser::plot_correlation_matrix(
          rv$corr_result$corr_filtered,
          title = "Correlation Matrix - Filtered Features"
        )
      })
      
      # Update feature selection dropdown
      updateSelectInput(session, "feature_color", choices = rv$feature_names)
      
      output$filtering_status <- renderText(
        paste("Filtering completed! Features reduced from", 
              ncol(df) - 1, "to", length(rv$feature_names))
      )
    }, error = function(e) {
      output$filtering_status <- renderText(paste("Error:", e$message))
    })
  })
  
  # ========== STEP 4: CLUSTERING ==========
  
  observeEvent(input$run_clustering, {
    req(rv$filtered_data)
    
    output$clustering_status <- renderText("Running UMAP and Leiden clustering...")
    
    tryCatch({
      # Save filtered data temporarily
      temp_file <- tempfile(fileext = ".csv")
      write.csv(rv$filtered_data, temp_file, row.names = FALSE)
      
      # Run UMAP and Leiden clustering
      rv$clustering_results <- RabAnalyser::run_umap_leiden(
        data_path = temp_file,
        n_neighbors = input$n_neighbors,
        min_dist = input$min_dist,
        resolution = input$resolution,
        n_bootstrap = input$n_bootstrap
      )
      
      rv$umap_df <- rv$clustering_results$umap_df
      
      # Create df_features for analysis
      rv$df_features <- rv$filtered_data %>%
        rename(Condition = Class) %>%
        mutate(Clusters = rv$umap_df$Cluster)
      
      # Plot resolution scan
      output$resolution_scan <- renderPlot({
        RabAnalyser::plot_resolution_scan(rv$clustering_results$resolution_scan)
      })
      
      # Plot cluster stability
      output$cluster_stability <- renderPlot({
        RabAnalyser::plot_cluster_stability(rv$clustering_results$stability)
      })
      
      output$clustering_status <- renderText("Clustering completed successfully!")
      
      # Clean up temp file
      unlink(temp_file)
    }, error = function(e) {
      output$clustering_status <- renderText(paste("Error:", e$message))
    })
  })
  
  # ========== STEP 4: VISUALIZATION ==========
  
  # UMAP plot by cluster or class
  output$umap_plot <- renderPlot({
    req(rv$umap_df)
    
    discrete <- input$umap_color %in% c("Cluster", "Class")
    
    RabAnalyser::plot_umap(
      rv$umap_df,
      color_by = input$umap_color,
      discrete = discrete
    )
  })
  
  # UMAP plot colored by feature
  output$umap_feature <- renderPlot({
    req(rv$umap_df, rv$filtered_data, input$feature_color)
    
    if (!is.null(input$feature_color) && input$feature_color != "") {
      umap_res <- cbind(
        rv$umap_df,
        rv$filtered_data %>% select(-Class)
      )
      
      RabAnalyser::plot_umap(
        umap_res,
        color_by = input$feature_color,
        high_color = "#228B22",
        values_interval = c(-0.5, 0.5)
      )
    }
  })
  
  # Subpopulation proportions
  output$proportions_plot <- renderPlot({
    req(rv$df_features)
    
    RabAnalyser::plot_clusters_proportions(rv$df_features)
  })
  
  output$proportions_table <- DT::renderDataTable({
    req(rv$df_features)
    
    result <- RabAnalyser::analyze_subpopulation_proportions(rv$df_features)
    DT::datatable(result, options = list(scrollX = TRUE, pageLength = 10))
  })
  
  # ========== STEP 5: STATISTICAL ANALYSIS ==========
  
  output$cluster_stats <- renderPlot({
    req(rv$df_features)
    
    stats_res <- RabAnalyser::cluster_feature_stats(rv$df_features)
    stats_res$plot
  })
  
  output$fingerprint_heatmap <- renderPlot({
    req(rv$df_features)
    
    fingerprint_res <- RabAnalyser::ks_cluster_fingerprint_heatmap(
      rv$df_features,
      values_interval = c(-0.3, 0.3),
      midpoint_val = 0
    )
    fingerprint_res$plot
  })
  
  output$feature_importance <- renderPlot({
    req(rv$df_features)
    
    importance_res <- RabAnalyser::feature_importance_analysis(
      rv$df_features %>% select(-Condition)
    )
    importance_res$plot
  })
  
  # ========== EXPORT FUNCTIONALITY ==========
  
  output$download_filtered <- downloadHandler(
    filename = function() {
      paste0("filtered_data_", Sys.Date(), ".csv")
    },
    content = function(file) {
      req(rv$filtered_data)
      write.csv(rv$filtered_data, file, row.names = FALSE)
    }
  )
  
  output$download_umap <- downloadHandler(
    filename = function() {
      paste0("umap_results_", Sys.Date(), ".csv")
    },
    content = function(file) {
      req(rv$umap_df)
      write.csv(rv$umap_df, file, row.names = FALSE)
    }
  )
  
  output$download_stats <- downloadHandler(
    filename = function() {
      paste0("statistics_", Sys.Date(), ".csv")
    },
    content = function(file) {
      req(rv$df_features)
      stats_res <- RabAnalyser::cluster_feature_stats(rv$df_features)
      write.csv(stats_res$data, file, row.names = FALSE)
    }
  )
  
}