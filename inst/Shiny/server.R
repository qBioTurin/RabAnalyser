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
    input_folder_selected = NULL,
    umap_results = NULL,
    resolution_scan = NULL,
    graph_path = NULL
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
  # Observer to load features when files are uploaded
  observe({
    req(input$comparison_files)

    tryCatch({
      # Load first comparison file to get available features
      file_path <- input$comparison_files$datapath[1]
      file_name <- input$comparison_files$name[1]

      if (grepl("\\.csv$", file_name, ignore.case = TRUE)) {
        comp_data <- readr::read_csv(file_path, show_col_types = FALSE)
      } else {
        comp_data <- readxl::read_excel(file_path)
      }

      # Get all feature names
      available_features <- colnames(comp_data)

      # Update selectizeInput with available features
      updateSelectizeInput(session, "ks_selected_features",
                          choices = available_features,
                          selected = available_features)

      # Display preview table
      output$ks_preview_table <- DT::renderDataTable({
        DT::datatable(head(comp_data), options = list(scrollX = TRUE, pageLength = 5))
      })
    }, error = function(e) {
      # Silently fail if file can't be read
      NULL
    })
  })

  observe({
    req(input$reference_file)

    tryCatch({
      # Load first comparison file to get available features
      file_path <- input$reference_file$datapath[1]
      file_name <- input$reference_file$name[1]

      if (grepl("\\.csv$", file_name, ignore.case = TRUE)) {
        ref_data <- readr::read_csv(file_path, show_col_types = FALSE)
      } else {
        ref_data <- readxl::read_excel(file_path)
      }

      # Display preview table
      output$ref_preview_table <- DT::renderDataTable({
        DT::datatable(head(ref_data), options = list(scrollX = TRUE, pageLength = 5))
      })
    }, error = function(e) {
      # Silently fail if file can't be read
      NULL
    })
  })

  observeEvent(input$run_ks, {
    req(input$reference_file, input$comparison_files)

    output$ks_status <- renderText("Running KS analysis...")
    shinybusy::show_modal_spinner() # show the modal window
    tryCatch({
      # Load reference population

      file_path <- input$reference_file$datapath
      file_name <- input$reference_file$name

      if (grepl("\\.csv$", file_name, ignore.case = TRUE)) {
        reference_data <- readr::read_csv(file_path)
      } else {
        reference_data <- readxl::read_excel(file_path)
      }

      reference_data$ID_image <- 1

      # Rename cell label column if needed
      if (input$id_column %in% colnames(reference_data)) {
        reference_data <- reference_data %>% rename(Cell_label = !!sym(input$id_column))
      }

      # Process each comparison file
      ks_results_list <- list()

      for (i in seq_along(input$comparison_files$datapath)) {
        if (grepl("\\.csv$", file_name, ignore.case = TRUE)) {
          comp_data <- readr::read_csv(input$comparison_files$datapath[i], show_col_types = FALSE)
        } else {
          comp_data <- readxl::read_excel(input$comparison_files$datapath[i])
        }

        if(!"ID_image" %in% colnames(comp_data)) { comp_data$ID_image <- 1 }

        # Rename cell label column if needed
        if (input$id_column %in% colnames(comp_data)) {
          comp_data <- comp_data %>% rename(Cell_label = !!sym(input$id_column))
        }


        # Use selected features from the selectizeInput
        if (!is.null(input$ks_selected_features) && length(input$ks_selected_features) > 0) {
          features <- input$ks_selected_features
          if (input$id_column %in% features) { features = features[-which(input$id_column %in% features)] }
        } else {
          output$ks_status <- renderText("No features were selected!")
          return()
        }

        ks_result <- RabAnalyser::perform_ks_analysis(
          ctrl_matrix = reference_data,
          comp_matrix = comp_data,
          features = features,
          cores = input$ks_cores
        )

        ks_result$Class <- tools::file_path_sans_ext(input$comparison_files$name[i])
        ks_results_list[[i]] <- ks_result
      }

      rv$ks_results <- bind_rows(ks_results_list)

      output$ks_results_table <- DT::renderDT({
        DT::datatable(rv$ks_results, extensions = "Buttons", options = list(scrollX = TRUE, pageLength = 10,dom = "Bfrtip",buttons = c("copy", "csv", "excel", "pdf") ) )
      })

      output$ks_status <- renderText("KS analysis completed successfully!")
    }, error = function(e) {
      output$ks_status <- renderText(paste("Error:", e$message))
    })
    shinybusy::remove_modal_spinner() # remove the modal window
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
      rv$feature_names <- colnames(rv$filtered_data)

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

  # ========== STEP 4: UMAP + RESOLUTION SCAN ==========

  observeEvent(input$run_umap_scan, {
    req(rv$filtered_data)
    shinybusy::show_modal_spinner() # show the modal window

    output$umap_scan_status <- renderText("Running UMAP embedding and resolution scan...")

    tryCatch({
      # Run UMAP + resolution scan
      rv$umap_results <- RabAnalyser::run_umap_resolution_scan(
        data = rv$filtered_data,
        n_neighbors = input$n_neighbors,
        min_dist = input$min_dist,
        gamma_min = input$gamma_min,
        gamma_max = input$gamma_max,
        n_gamma_steps = input$n_gamma_steps,
        save_graph = TRUE
      )

      rv$resolution_scan <- rv$umap_results$resolution_scan
      rv$graph_path <- rv$umap_results$graph_path

      # Store UMAP coordinates
      rv$umap_df <- rv$umap_results$umap_df

      # Plot resolution scan
      output$resolution_scan <- renderPlot({
        RabAnalyser::plot_resolution_scan(rv$resolution_scan)
      })

      output$umap_scan_status <- renderText("UMAP and resolution scan completed! Select gamma and run clustering.")
    }, error = function(e) {
      output$umap_scan_status <- renderText(paste("Error:", e$message))
    })
    shinybusy::remove_modal_spinner() # remove it when done
  })

  # ========== STEP 5: LEIDEN CLUSTERING ==========

  observeEvent(input$run_clustering, {
    req(rv$umap_results, rv$graph_path)
    shinybusy::show_modal_spinner() # show the modal window
    output$clustering_status <- renderText("Running Leiden clustering with selected gamma...")

    tryCatch({

      # Run Leiden clustering
      rv$clustering_results <- RabAnalyser::run_leiden_clustering(
        umap_data = rv$umap_results$umap_df,
        graph_path = rv$graph_path,
        gamma = input$selected_gamma,
        n_bootstrap = input$n_bootstrap,
        subsample_prop = input$subsample_prop,
        stability_analysis = TRUE
      )

      rv$clustered_umap_df <- rv$clustering_results$umap_df
      rv$df_clustered_features <- rv$filtered_data %>%
        rename(Condition = Class) %>%
        mutate(Clusters = rv$clustering_results$umap_df$Cluster)

      # Plot cluster stability
      output$cluster_stability <- renderPlot({
        RabAnalyser::plot_cluster_stability(rv$clustering_results$stability)
      })

      output$clustering_status <- renderText("Clustering completed successfully!")

    }, error = function(e) {
      output$clustering_status <- renderText(paste("Error:", e$message))
    })
    shinybusy::remove_modal_spinner() # show the modal window
  })

  # ========== STEP 4: VISUALIZATION ==========

  # UMAP plot by cluster or class
  output$umap_plot <- renderPlot({
    req(rv$clustered_umap_df)

    RabAnalyser::plot_umap(
      rv$clustered_umap_df,
      color_by = "Cluster",
      discrete = T
    )
  })

  # UMAP plot colored by feature
  output$feature_color <- renderPlot({
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
  observe({
    req(rv$clustering_results)
    df_features = rv$df_clustered_features

    output$proportions_plot <- renderPlot({
      RabAnalyser::plot_clusters_proportions(df_features)
    })

    output$proportions_table <- DT::renderDataTable({
      result <- RabAnalyser::analyze_subpopulation_proportions(df_features)
      DT::datatable(result$pairwise_tests, options = list(scrollX = TRUE, pageLength = 10))
    })

  })


  # ========== STEP 5: STATISTICAL ANALYSIS ==========

  output$cluster_stats <- renderPlot({
    req(rv$df_clustered_features)

    stats_res <- RabAnalyser::cluster_feature_stats(rv$df_clustered_features)
    stats_res$plot
  })

  output$fingerprint_heatmap <- renderPlot({
    req(rv$df_clustered_features)

    fingerprint_res <- RabAnalyser::ks_cluster_fingerprint_heatmap(
      rv$df_clustered_features,
      values_interval = c(-0.3, 0.3),
      midpoint_val = 0
    )
    fingerprint_res$plot
  })

  output$feature_importance <- renderPlot({
    req(rv$df_clustered_features)

    importance_res <- RabAnalyser::feature_importance_analysis(
      rv$df_clustered_features %>% select(-Condition)
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
