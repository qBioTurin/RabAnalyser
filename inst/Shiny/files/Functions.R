
initializeClusteringStep = function(session,input,output,Clustering_results,files){
  df = do.call(rbind,
               lapply(seq_along(files),function(i){
                 df <- read_excel(files[i])
                 df$Treatment = gsub(pattern = ".xlsx$",replacement = "", x= input$Clustering_excel$name[i])
                 df
               })
  )

  # Feature Selection
  threshold <- 0.75
  reduced_df <- FilterFeat(df%>%select(-Treatment), threshold)

  Clustering_results$Data = df
  Clustering_results$FilteredData = cbind(reduced_df,df%>%select(Treatment))

  features = colnames(reduced_df)

  if(length(unique(df$Treatment))>1 ){
    features = c( "Treatment",features)
    output$UI_clustviolin = renderUI({
      column(4,offset=1,
             checkboxInput("Group_Clustering_violin",label = "Group by Treatment",value = F )
      )
    })
    output$UI_clustsubpop = renderUI({
      fluidRow(
        column(4,
               checkboxInput("Group_Clustering_subpop",label = "Group by Treatment",value = F )
        )
      )
    })
  }
  Clustering_results$features = features

  updateSelectInput(session =session, "Clustering_colorUMAPselect", choices = c( features), selected = features[1])
  # Correlation Matrix Plot (Before Filtering)
  correlation_matrixOLD <- cor(df%>%select(-Treatment), use = "pairwise.complete.obs")

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
  umap_df <- as.data.frame(reducer)
  names(umap_df) <- c("UMAP1", "UMAP2")

  ### cluster SCOREs ###
  Clustering_results$clustering <- plotLIST <- cluster.generation(data = umap_df)

  # Display results
  k = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1]))

  output$Clustering_clustChoicePlot <- renderPlot({
    plotLIST$silhouette +
      geom_vline(aes(xintercept = as.numeric(names(plotLIST$AllClusteringIndex$bestK[1])),
                     color = paste0("Best k considering ",
                                    plotLIST$AllClusteringIndex$bestK[1], " indexes over ",
                                    sum((plotLIST$AllClusteringIndex$bestK)))
      ), linetype = "dashed"
      )+labs(col = "")
  })

  bestk = as.data.frame(plotLIST$AllClusteringIndex$bestK)
  colnames(bestk) = c("Number of\n clusters", "Number of indexes\n in accordance")
  output$Clustering_ClusterIndexesTable = renderTable(bestk)
  updateSliderInput(session,"Clustering_clusterSlider",value = k,min =2 ,max = 10)
}

FilterFeat <- function(df, threshold) {
  # Compute correlation matrix
  correlation_matrix <- cor(df, use = "pairwise.complete.obs")

  # Find feature pairs with correlation above the threshold
  correlated_pairs <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)
  correlated_pairs <- correlated_pairs[correlated_pairs[, 1] < correlated_pairs[, 2], ]

  removed_features <- c()

  for (i in seq_len(nrow(correlated_pairs))) {
    feature1 <- colnames(df)[correlated_pairs[i, 1]]
    feature2 <- colnames(df)[correlated_pairs[i, 2]]

    if (feature1 %in% removed_features || feature2 %in% removed_features) {
      next
    }

    # Compare variances to decide which feature to remove
    if (var(df[[feature1]], na.rm = TRUE) > var(df[[feature2]], na.rm = TRUE)) {
      removed_features <- c(removed_features, feature2)
    } else {
      removed_features <- c(removed_features, feature1)
    }
  }

  # Return the DataFrame with redundant features removed
  df <- df[, !colnames(df) %in% removed_features]
  return(df)
}

cluster_indexes <-function(data){
  AllIndexes = do.call(rbind,
                       lapply(2:10,function(k){
                         # Perform the kmeans algorithm
                         set.seed(42)
                         cl <- kmeans(data, k)
                         df = clusterCrit::intCriteria(as.matrix(data),cl$cluster,"all")
                         data.frame(k = k, as.data.frame(df) )
                       }
                       )
  )


  vals <- vector()
  for(nIndex in names(AllIndexes %>% select(-k))){
    vals = c(vals, clusterCrit::bestCriterion(AllIndexes[[nIndex]],nIndex))
  }

  bestK = sort(table(AllIndexes$k[vals]),decreasing = T)

  return(list(bestK = bestK, AllIndexes = AllIndexes))
}

cluster.generation <- function(data) {
    #sil_values <- factoextra::fviz_nbclust(data, kmeans, method = "silhouette")
    allCl = cluster_indexes(data)

    sil_values = rbind(allCl$AllIndexes %>% select(k,silhouette), c(1,0)) %>%
      ggplot() + geom_line(aes(x = k, y = silhouette))+ geom_point(aes(x = k, y = silhouette)) +
      theme_minimal()+labs(x = "Number of Clusters", y = "Silhouette score (to maximise)")

  return(list(Data = data, silhouette = sil_values, AllClusteringIndex = allCl))
}

# Function to clean data (remove NaN rows)
cleaning <- function(matrix) {
  matrix <- matrix[complete.cases(matrix), ]
  return(as.data.frame(matrix))
}

# Function to compute two-sample signed KS statistic
two_sample_signed_ks_statistic <- function(sample1, sample2,ecdf1) {
  #ecdf1 <- ecdf(sample1)
  ecdf2 <- ecdf(sample2)

  all_x <- sort(unique(c(sample1, sample2)))

  y1_interp <- ecdf1(all_x)
  y2_interp <- ecdf2(all_x)

  differences <- y1_interp - y2_interp
  ks_statistic <- max(abs(differences))
  signed_ks_statistic <- differences[which.max(abs(differences))]

  return(c(ks_statistic, signed_ks_statistic))
}

RabAnalyser.getVolumes=function(exclude, from="~", custom_name="Home"){
  osSystem <- Sys.info()["sysname"]
  userHome <- path_expand(from)  # Get the user's home directory

  if (osSystem == "Darwin") {
    #volumes <- fs::dir_ls(userHome)
    #names(volumes) <- basename(volumes)
    volumes <- userHome
    names(volumes) <- basename(volumes)
  }
  else if (osSystem == "Linux") {
    volumes <- c(setNames(userHome, custom_name))
    media_path <- file.path(userHome, "media")
    if (isTRUE(dir_exists(media_path))) {
      media <- dir_ls(media_path)
      names(media) <- basename(media)
      volumes <- c(volumes, media)
    }
  }
  else if (osSystem == "Windows") {
    userHome <- gsub("\\\\", "/", userHome)  # Convert Windows path format
    volumes <- c(setNames(userHome, custom_name))

    # Check for mounted drives inside user home (e.g., OneDrive, Network Drives)
    possible_drives <- fs::dir_ls(userHome, type = "directory")
    names(possible_drives) <- basename(possible_drives)
    volumes <- c(volumes, possible_drives)
  }
  else {
    stop("unsupported OS")
  }

  if (!is.null(exclude)) {
    volumes <- volumes[!names(volumes) %in% exclude]
  }

  return(volumes)
}

################### PLOT functions ####################

# Function to show the customization modal for any given plot
showPlotModal <- function(session, plot_id, plot_obj) {
  layer_tabs <- generateLayerParameters(plot_obj)

  showModal(modalDialog(
    title = paste("Customize or Download", plot_id),
    tabsetPanel(
      tabPanel("Layer Customization", do.call(tagList, layer_tabs)),
      tabPanel("Common Parameters", generatePlotParameters(plot_obj)),
      tabPanel("Save Plot", generateSavePlotTab())
    ),
    footer = tagList(
      actionButton(paste0("applyChanges_", plot_id), "Apply Changes"),
      downloadButton(paste0("downloadPlotButton_", plot_id), "Download Plot"),
      modalButton("Close")
    ),
    easyClose = FALSE
  ))
}

# Function to generate a download handler for any plot
generateDownloadHandler <- function(session, input, output, plot_id, plot_obj) {
  output[[paste0("downloadPlotButton_", plot_id)]] <- downloadHandler(
    filename = function() {
      paste0("plot_", plot_id, "_", Sys.Date(), ".png")
    },
    content = function(file) {
      manageSpinner(TRUE)
      width <- input$plotWidth
      height <- input$plotHeight
      dpi <- input$plotResolution
      savePlotAsPNG(plot_obj, file, width, height, dpi)
      manageSpinner(FALSE)
    }
  )
}

# Function to apply changes and update the plot dynamically
applyPlotChanges <- function(session, input, output,plot_id, plot_obj,listSavePlot) {
  observeEvent(input[[paste0("applyChanges_", plot_id)]], {
    updatedPlot <- customizePlot(plot_obj, input)
    listSavePlot <- updatedPlot
    output[[plot_id]] <- renderPlot({ updatedPlot })
    removeModal()
  })
}

generateLayerParameters <- function(plot) {
  # Build the plot to extract computed data and aesthetics
  plot_build <- ggplot_build(plot)

  # Dynamically generate tabs for each layer
  layer_tabs <- lapply(seq_along(plot$layers), function(i) {
    layer <- plot$layers[[i]]
    layer_type <- class(layer$geom)[1]

    aes_mapping <- layer$mapping
    data <- plot_build$data[[i]]
    default_params <- layer$aes_params

    if(!is.null(aes_mapping$colour)){
      colour_mapping<-rlang::quo_name(aes_mapping$colour)
      data<-data %>% mutate(!!colour_mapping := data[["color"]])
    }
    if(!is.null(aes_mapping$shape)){
      shape_mapping<-rlang::quo_name(aes_mapping$shape)
      data<-data %>% mutate(!!shape_mapping := data[["shape"]])
    }

    #Function to extract specific aesthetic values
    extract_aes_value <- function(aesthetic, default,x_val) {
      if (!is.null(aes_mapping[[aesthetic]])) {
        if (aesthetic == "colour") {
          filtered_data <- data[data[["colour"]] == x_val, ]
          return(unique(filtered_data[[aesthetic]]))
        }
        else if (aesthetic == "alpha") {
          filtered_data <- data[data[["alpha"]] == x_val, ]
          return(unique(filtered_data[[aesthetic]]))
        }
        else if (aesthetic == "shape") {
          filtered_data <- data[data[["shape"]] == x_val, ]
          return(unique(filtered_data[[aesthetic]]))
        }
        else{
          filtered_data <- data[data[["group"]] == x_val, ]
          return(unique(filtered_data[[aesthetic]]))
        }
      }
      if (!is.null(default_params[[aesthetic]])) {
        return(default_params[[aesthetic]])
      }
      return(default)
    }

    # Generate inputs dynamically for layer-specific parameters
    tabPanel(
      paste("Layer", i, "(", layer_type, ")"),
      wellPanel(
        h4(paste("Customize Layer", i, "-", layer_type)),

        # Common controls for all layers
        sliderInput(
          paste0("layerSize_", i), "Layer Size",
          min = 1, max = 5, value = extract_aes_value("size", 1,1)
        ),

        # Layer-specific customization
        if (layer_type == "GeomPoint") {
          tagList(
            if ("colour" %in% names(aes_mapping)) {
              lapply(unique(data[["colour"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("pointColor_", i, "_", gsub("#", "", x_val)),
                  paste0("Point Color for ",colour_mapping," ", unique(data[data[["colour"]] == x_val, ][[colour_mapping]])),
                  value = x_val
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("pointColor_", i),
                "Point Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            },

            if ("shape" %in% names(aes_mapping)) {
              lapply(unique(data[["shape"]]), function(x_val) {
                selectInput(
                  paste0("pointShape_", i, "_", x_val),
                  paste0("Point Shape for ",shape_mapping," ", unique(data[data[["shape"]] == x_val, ][[shape_mapping]])),
                  choices = c("Circle" = 16, "Triangle" = 17, "Square" = 15, "Cross" = 4, "Plus" = 3),
                  selected = x_val
                )
              })
            } else {
              selectInput(
                paste0("pointShape_", i),
                "Point Shape",
                choices = c("Circle" = 16, "Triangle" = 17, "Square" = 15, "Cross" = 4, "Plus" = 3),
                selected = extract_aes_value("shape", 16,1)
              )
            }
          )
        }
        else if (layer_type == "GeomLine") {
          tagList(
            # Line Type
            if ("linetype" %in% names(aes_mapping)) {
              lapply(unique(data[["group"]]), function(x_val) {
                selectInput(
                  paste0("lineType_", i, "_", x_val),
                  paste("Line Type for Group", x_val),
                  choices = c("Solid" = "solid", "Dashed" = "dashed", "Dotted" = "dotted", "Dotdash" = "dotdash"),
                  selected = extract_aes_value("linetype", "solid",x_val)
                )
              })
            } else {
              selectInput(
                paste0("lineType_", i),
                "Line Type",
                choices = c("Solid" = "solid", "Dashed" = "dashed", "Dotted" = "dotted", "Dotdash" = "dotdash"),
                selected = extract_aes_value("linetype", "solid",1)
              )
            },

            # Line Color
            if ("colour" %in% names(aes_mapping)) {
              lapply(unique(data[["colour"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("lineColor_", i, "_",  gsub("#", "", x_val)),
                  paste0("Line Color for ", colour_mapping, " ", unique(data[data[["colour"]] == x_val, ][[colour_mapping]])),
                  #value = extract_aes_value("colour", "#000000",x_val)
                  value = x_val
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("lineColor_", i),
                "Line Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            }
          )
        }
        else if (layer_type == "GeomBar") {
          tagList(
            # Bar Fill Color
            if ("fill" %in% names(aes_mapping)) {
              lapply(unique(data[["group"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("barFillColor_", i, "_", x_val),
                  paste("Bar Fill Color for Group", x_val),
                  value = extract_aes_value("fill", "#FF9999",x_val)
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("barFillColor_", i),
                "Bar Fill Color",
                value = extract_aes_value("fill", "#FF9999",1)
              )
            },
            sliderInput(
              paste0("barWidth_", i),
              "Bar Width",
              min = 0.1, max = 1, value = extract_aes_value("width", 0.5)
            ),
            if("color" %in% names(aes_mapping)){
              lapply(unique(data[["colour"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("barColor_", i, "_",  gsub("#", "", x_val)),
                  paste0("Bar Color for", colour_mapping, " ", unique(data[data[["colour"]] == x_val, ][[colour_mapping]])),
                  #value = extract_aes_value("colour", "#000000",x_val)
                  value = x_val
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("barColor_", i),
                "Bar Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            }
          )
        }
        else if (layer_type == "GeomErrorbar") {
          tagList(
            sliderInput(
              paste0("errorBarWidth_", i),
              "Error Bar Width",
              min = 0.1, max = 2, value = extract_aes_value("width", 0.5,1)
            )
            ,
            if ("colour" %in% names(aes_mapping)) {
              lapply(unique(data[["colour"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("errorBarColor_", i, "_",  gsub("#", "", x_val)),
                  paste("Error Bar Color for", colour_mapping, " ", unique(data[data[["colour"]] == x_val, ][[colour_mapping]])),
                  #value = extract_aes_value("colour", "#000000",x_val)
                  value = x_val
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("errorBarColor_", i),
                "Error Bar Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            }
          )
        }
        else if (layer_type == "GeomBoxplot") {
          tagList(
            # Notch
            checkboxInput(
              paste0("notch_", i),
              "Add Notch",
              value = extract_aes_value("notch", FALSE,1)
            ),

            if ("fill" %in% names(aes_mapping)) {
              lapply(unique(data[["group"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("boxFillColor_", i, "_", x_val),
                  paste("Boxplot Fill Color for Group", x_val),
                  value = extract_aes_value("fill", "#FF9999",x_val)
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("boxFillColor_", i),
                "Boxplot Fill Color",
                value = extract_aes_value("fill", "#FF9999",1)
              )
            },

            if("colour" %in% names(aes_mapping)){
              lapply(unique(data[["colour"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("boxOutlineColor_", i, "_",  gsub("#", "", x_val)),
                  paste("Boxplot Outline Color for", colour_mapping, " ", unique(data[data[["colour"]] == x_val, ][[colour_mapping]])),
                  #value = extract_aes_value("colour", "#000000",x_val)
                  value = x_val
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("boxOutlineColor_", i),
                "Boxplot Outline Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            },
            colourpicker::colourInput(
              paste0("outlierColor_", i),
              "Outlier Color",
              value = extract_aes_value("outlier.colour", "#FF0000",1)
            ),
            sliderInput(
              paste0("outlierSize_", i),
              "Outlier Size",
              min = 1, max = 5, value = extract_aes_value("outlier.size", 2,1)
            ),
            sliderInput(
              paste0("boxWidth_", i),
              paste("Box Width"),
              min = 0.1, max = 1, value = extract_aes_value("width", 0.5,1)
            )
          )
        }
        else if (layer_type == "GeomViolin") {
          tagList(
            # Notch
            checkboxInput(
              paste0("notch_", i),
              "Add Notch",
              value = extract_aes_value("notch", FALSE,1)
            ),

            if ("fill" %in% names(data)) {
              lapply(unique(data[["group"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("violinFillColor_", i, "_", x_val),
                  paste("Violin Fill Color for Group", x_val),
                  value = extract_aes_value("fill", "#FF9999",x_val)
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("violinFillColor_", i),
                "Violin Fill Color",
                value = extract_aes_value("fill", "#FF9999",1)
              )
            },

            if("colour" %in% names(data)){
              lapply(unique(data[["group"]]), function(x_val) {
                colourpicker::colourInput(
                  paste0("violinOutlineColor_", i, "_", x_val),
                  paste("Violin Outline Color for Group", x_val),
                  value = extract_aes_value("colour", "#FF9999",x_val)
                )
              })
            } else {
              colourpicker::colourInput(
                paste0("violinOutlineColor_", i),
                "Violin Outline Color",
                value = extract_aes_value("colour", "#000000",1)
              )
            },
            sliderInput(
              paste0("violinAlpha_", i),
              paste("Violin Fill Alpha"),
              min = 0, max = 1, value = extract_aes_value("alpha", 0,1)
            ),
            sliderInput(
              paste0("violinWidth_", i),
              paste("Violin Width"),
              min = 0.1, max = 1, value = extract_aes_value("width", 0.5,1)
            )
          )
        } else {
          p("No specific customization available for this layer type.")
        }
      )
    )
  })

  return(layer_tabs)
}

customizePlot <- function(plot, input) {
  # General theme and title customization
  backgroundColor <- input$backgroundColor

  updatedPlot <- plot +
    ggtitle(input$plotTitle) +
    labs(x = input$xAxisLabel, y = input$yAxisLabel) +
    theme(
      plot.title = element_text(size = input$plotTitleFontSize, color = input$plotTitleColor),
      axis.text.x = element_text(size = input$xAxisFontSize),
      axis.text.y = element_text(size = input$yAxisFontSize),
      panel.background = element_rect(fill = backgroundColor, color = backgroundColor),
      plot.background = element_rect(fill = backgroundColor, color = backgroundColor)
    )

  updatedPlot$layers <- list()
  plot_build=ggplot_build(plot)

  for (i in seq_along(plot$layers)) {
    layer <- plot$layers[[i]]
    layer_type <- class(layer$geom)[1]
    current_mapping <- layer$mapping

    # Determine if properties are mapped
    is_colour_mapped <- "colour" %in% names(current_mapping)
    is_fill_mapped <- "fill" %in% names(current_mapping)
    is_shape_mapped <- "shape" %in% names(current_mapping)
    is_linetype_mapped <- "linetype" %in% names(current_mapping)


    unique_x<-unique(plot_build$data[[i]][["group"]] )
    unique_x_colour<-unique(plot_build$data[[i]][["colour"]] )
    unique_x_shape<-unique(plot_build$data[[i]][["shape"]] )
    # Customize each type of geom layer
    if (layer_type == "GeomPoint") {

      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x_colour, function(x_val) {
            input[[paste0("pointColor_", i, "_",  gsub("#", "", x_val))]]
          }))
        )
      } else NULL

      shape_scale <- if (is_shape_mapped) {
        scale_shape_manual(
          values = unlist(lapply(unique_x_shape, function(x_val) {
            as.integer(input[[paste0("pointShape_", i, "_", x_val)]])
          }))
        )
      } else NULL

      if(!is_shape_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_point(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            shape = as.integer(input[[paste0("pointShape_", i)]]),
            colour= input[[paste0("pointColor_", i)]]
          )+
          colour_scale +
          shape_scale
      }
      else if(is_shape_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_point(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            colour= input[[paste0("pointColor_", i)]]
          )+
          colour_scale +
          shape_scale
      }
      else if(!is_shape_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_point(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            shape = as.integer(input[[paste0("pointShape_", i)]]),
          )+
          colour_scale +
          shape_scale
      }
      else if(is_shape_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_point(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
          )+
          colour_scale +
          shape_scale
      }

    }
    else if (layer_type == "GeomBar") {

      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x_colour, function(x_val) {
            input[[paste0("barColor_", i,"_", x_val)]]
          }))
        )
      } else NULL
      fill_scale <- if (is_fill_mapped) {
        scale_fill_manual(
          values = unlist(lapply(unique_x, function(x_val) {
            input[[paste0("barFillColor_", i,"_", x_val)]]
          }))
        )
      } else NULL

      if(!is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_bar(
            mapping = layer$mapping,
            position = layer$position,
            stat = "identity",
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("barWidth_", i)]],
            fill= input[[paste0("barFillColor_", i)]],
            color= input[[paste0("barColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_bar(
            #data = layer$data,
            mapping = layer$mapping,
            position = layer$position,
            stat = "identity",
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("barWidth_", i)]],
            color= input[[paste0("barColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(!is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_bar(
            mapping = layer$mapping,
            position = layer$position,
            stat = "identity",
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("barWidth_", i)]],
            fill= input[[paste0("barFillColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_bar(
            mapping = layer$mapping,
            position = layer$position,
            stat = "identity",
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("barWidth_", i)]],
          ) +
          colour_scale +
          fill_scale
      }

    }
    else if (layer_type == "GeomLine") {
      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x_colour, function(x_val) {
            input[[paste0("lineColor_", i,"_", x_val)]]
          }))
        )
      } else NULL
      linetype_scale <- if (is_linetype_mapped) {
        scale_linetype_manual(
          values = unlist(lapply(unique_x, function(x_val) {
            input[[paste0("lineType_", i,"_", x_val)]]
          }))
        )
      } else NULL

      if(!is_linetype_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_line(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            linetype= input[[paste0("lineType_", i)]],
            color= input[[paste0("lineColor_", i)]]
          ) +
          colour_scale +
          linetype_scale
      }
      else if(is_linetype_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_line(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            color= input[[paste0("lineColor_", i)]]
          ) +
          colour_scale +
          linetype_scale
      }
      else if(!is_linetype_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_line(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            linetype= input[[paste0("lineType_", i)]]
          ) +
          colour_scale +
          linetype_scale
      }
      else if(is_linetype_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_line(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]]
          ) +
          colour_scale +
          linetype_scale
      }

    }
    else if (layer_type == "GeomBoxplot") {

      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x_colour, function(x_val) {
            input[[paste0("boxOutlineColor_", i,"_", x_val)]]
          }))
        )
      } else NULL
      fill_scale <- if (is_fill_mapped) {
        scale_fill_manual(
          values = unlist(lapply(unique_x, function(x_val) {
            input[[paste0("boxFillColor_", i,"_", x_val)]]
          }))
        )
      } else NULL

      if(!is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_boxplot(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            notch = input[[paste0("notch_", i)]],
            width = input[[paste0("boxWidth_", i)]],
            outlier.size = input[[paste0("outlierSize_", i)]],
            outlier.colour = input[[paste0("outlierColor_", i)]],
            color= input[[paste0("boxOutlineColor_", i)]],
            fill = input[[paste0("boxFillColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_boxplot(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            notch = input[[paste0("notch_", i)]],
            width = input[[paste0("boxWidth_", i)]],
            outlier.size = input[[paste0("outlierSize_", i)]],
            outlier.colour = input[[paste0("outlierColor_", i)]],
            color= input[[paste0("boxOutlineColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(!is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_boxplot(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            notch = input[[paste0("notch_", i)]],
            width = input[[paste0("boxWidth_", i)]],
            outlier.size = input[[paste0("outlierSize_", i)]],
            outlier.colour = input[[paste0("outlierColor_", i)]],
            fill= input[[paste0("boxFillColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_boxplot(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            notch = input[[paste0("notch_", i)]],
            width = input[[paste0("boxWidth_", i)]],
            outlier.size = input[[paste0("outlierSize_", i)]],
            outlier.colour = input[[paste0("outlierColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }

    }
    else if (layer_type == "GeomViolin") {

      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x, function(x_val) {
            input[[paste0("violinOutlineColor_", i,"_", x_val)]]
          }))
        )
      } else NULL
      fill_scale <- if (is_fill_mapped) {
        scale_fill_manual(
          values = unlist(lapply(unique_x, function(x_val) {
            input[[paste0("violinFillColor_", i,"_", x_val)]]
          }))
        )
      } else NULL

      if(!is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_violin(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            alpha = input[[paste0("violinAlpha_", i)]],
            width = input[[paste0("violinWidth_", i)]],
            color= input[[paste0("violinOutlineColor_", i)]],
            fill = input[[paste0("violinFillColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_violin(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            alpha = input[[paste0("violinAlpha_", i)]],
            width = input[[paste0("violinWidth_", i)]],
            color= input[[paste0("violinOutlineColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(!is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_violin(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            alpha = input[[paste0("violinAlpha_", i)]],
            width = input[[paste0("violinWidth_", i)]],
            fill= input[[paste0("violinFillColor_", i)]]
          ) +
          colour_scale +
          fill_scale
      }
      else if(is_fill_mapped&&is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_violin(
            mapping = layer$mapping,
            position = layer$position,
            alpha = input[[paste0("violinAlpha_", i)]],
            size = input[[paste0("layerSize_", i)]],
            linewidth = input[[paste0("violinWidth_", i)]]
          )  +
          colour_scale +
          fill_scale
      }

    } else if (layer_type == "GeomErrorbar") {
      colour_scale <- if (is_colour_mapped) {
        scale_color_manual(
          values = unlist(lapply(unique_x_colour, function(x_val) {
            input[[paste0("errorBarColor_", i,"_", x_val)]]
          }))
        )
      } else NULL

      if(!is_colour_mapped){
        updatedPlot <- updatedPlot +
          geom_errorbar(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("errorBarWidth_", i)]],
            color= input[[paste0("errorBarColor_", i)]]
          ) +
          colour_scale
      }
      else{
        updatedPlot <- updatedPlot +
          geom_errorbar(
            mapping = layer$mapping,
            position = layer$position,
            size = input[[paste0("layerSize_", i)]],
            width = input[[paste0("errorBarWidth_", i)]]
          ) +
          colour_scale
      }

    } else {
      warning(paste("No specific customizations applied for layer type:", layer_type))
    }
  }

  return(updatedPlot)
}

generatePlotParameters <- function(plot) {

  plot_title <- plot$labels$title
  x_axis_label <- plot$labels$x
  y_axis_label <- plot$labels$y

  global_theme <- ggplot2::theme_get()
  plot_theme <- plot$theme %||% list()
  theme_elements <- utils::modifyList(global_theme, plot_theme)

  title_size <- theme_elements$plot.title$size %||% 14
  title_color <- theme_elements$plot.title$colour %||% "#000000"
  x_axis_size <- theme_elements$axis.text.x$size %||% 12
  y_axis_size <- theme_elements$axis.text.y$size %||% 12
  background_color <- theme_elements$panel.background$fill %||% "#FFFFFF"

  tagList(
    textInput(
      "plotTitle",
      "Plot Title",
      value = plot_title %||% ""
    ),
    sliderInput(
      "plotTitleFontSize",
      "Plot Title Font Size",
      min = 8,
      max = 30,
      value = title_size
    ),
    colourpicker::colourInput(
      "plotTitleColor",
      "Title Color",
      value = title_color
    ),
    textInput(
      "xAxisLabel",
      "X-Axis Label",
      value = x_axis_label %||% "X Axis"
    ),
    textInput(
      "yAxisLabel",
      "Y-Axis Label",
      value = y_axis_label %||% "Y Axis"
    ),
    sliderInput(
      "xAxisFontSize",
      "X-Axis Font Size",
      min = 8,
      max = 20,
      value = x_axis_size
    ),
    sliderInput(
      "yAxisFontSize",
      "Y-Axis Font Size",
      min = 8,
      max = 20,
      value = y_axis_size
    ),
    colourpicker::colourInput(
      "backgroundColor",
      "Background Color",
      value = background_color
    )
  )
}

generateSavePlotTab <- function() {
  tagList(
    numericInput("plotWidth", "Plot Width (inches)", value = 10, min = 1, max = 20),
    numericInput("plotHeight", "Plot Height (inches)", value = 4, min = 1, max = 20),
    numericInput("plotResolution", "Resolution (dpi)", value = 300, min = 72, max = 600)
  )
}

savePlotAsPNG <- function(plot, filePath, width, height, dpi) {
  req(plot) # Require the plot to be present

  ggsave(
    filename = filePath,
    plot = plot,
    device = "png",
    width = width,
    height = height,
    dpi = dpi
  )
}


