# Load required packages
library(shiny)
library(shinyFiles)
library(shinyjs)
library(stringr)
library(readxl)
library(readr)
library(DT)
library(ggplot2)
library(dplyr)
library(bslib)
library(writexl)

# Load RabAnalyser package
library(RabAnalyser)

# Set maximum upload size to 1000 MB
options(shiny.maxRequestSize = 1000 * 1024^2)

# Define UI cards for each workflow step
cards <- list(
  "home" = card(
    full_screen = TRUE,
    card_header(
      class = "bg-gradient-primary text-white",
      style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px;",
      tags$div(
        tags$h1(icon("microscope", style = "font-size: 1.2em;"), "RabAnalyser Workflow", 
                style = "margin-bottom: 10px; font-weight: 700;"),
        tags$h4("Single-Cell Rab Protein Analysis Platform", 
                style = "font-weight: 300; opacity: 0.95;")
      )
    ),
    tags$div(
      style = "padding: 20px;",
      tags$p(
        style = "font-size: 1.1em; color: #555; margin-bottom: 30px;",
        "This application implements a complete three-step workflow for single-cell Rab protein analysis:"
      ),
      
      # Step 1
      tags$div(
        class = "card mb-4",
        style = "border-left: 5px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
        tags$div(
          class = "card-body",
          tags$h3(
            icon("image", class = "text-primary"), 
            "Step 1: FEATURE EXTRACTION",
            style = "color: #667eea; font-weight: 600; margin-bottom: 15px;"
          ),
          tags$p(strong(icon("info-circle"), "Description:"), "The framework extracts 11 features from microscopy images."),
          tags$div(
            style = "background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("folder-open"), "INPUT:"), "Experimental conditions (e.g., FPW1, JK2, WK1) containing:"),
            tags$ul(
              style = "margin-bottom: 0;",
              tags$li(icon("circle", style = "font-size: 0.5em;"), " Cell segmentation mask (e.g., cell_mask)"),
              tags$li(icon("circle", style = "font-size: 0.5em;"), " Nucleus segmentation mask (e.g., nucleus_mask)"),
              tags$li(icon("circle", style = "font-size: 0.5em;"), " Rab spots segmentation mask (e.g., Rab5_mask)"),
              tags$li(icon("circle", style = "font-size: 0.5em;"), " Raw images of Rab spots (e.g., Rab5)"),
              tags$li(icon("circle", style = "font-size: 0.5em;"), " Excel file with feature names")
            )
          ),
          tags$div(
            style = "background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("file-export"), "OUTPUT:"), "CSV files where each row represents a Rab spot with Cell_ID and feature values (e.g., FPW1.csv, JK2.csv, WK1.csv).", style = "margin: 0;")
          )
        )
      ),
      
      # Step 2
      tags$div(
        class = "card mb-4",
        style = "border-left: 5px solid #f5576c; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
        tags$div(
          class = "card-body",
          tags$h3(
            icon("chart-line", class = "text-danger"), 
            "Step 2: KS ANALYSIS",
            style = "color: #f5576c; font-weight: 600; margin-bottom: 15px;"
          ),
          tags$p(strong(icon("info-circle"), "Description:"), "Identify a reference population and compare experimental conditions using the Kolmogorov-Smirnov (KS) statistic."),
          tags$div(
            style = "background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("folder-open"), "INPUT:"), "Excel files containing extracted features (e.g., FPW1.xlsx, JK2.xlsx, WK1.xlsx) and a reference population file (e.g., Reference_populationRab5_V2.xlsx) - a concatenation of all experimental conditions.", style = "margin: 0;")
          ),
          tags$div(
            style = "background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("file-export"), "OUTPUT:"), "Excel files where each row is a single cell with KS values representing distributional dissimilarity (e.g., FPW1_KSRab5_WholeRef_V2.xlsx).", style = "margin: 0;")
          )
        )
      ),
      
      # Step 3
      tags$div(
        class = "card mb-4",
        style = "border-left: 5px solid #38f9d7; box-shadow: 0 2px 8px rgba(0,0,0,0.1);",
        tags$div(
          class = "card-body",
          tags$h3(
            icon("project-diagram", class = "text-info"), 
            "Step 3: DATA CLUSTERING AND DOWNSTREAM ANALYSIS",
            style = "color: #38f9d7; font-weight: 600; margin-bottom: 15px;"
          ),
          tags$p(strong(icon("info-circle"), "Description:"), "Perform data clustering and characterize identified clusters."),
          tags$div(
            style = "background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("folder-open"), "INPUT:"), "Complete dataset containing all KS values per feature and per condition (e.g., GlioCells_KSvaluesRab5WholeRef_V2.xlsx) with a 'Class' column.", style = "margin: 0;")
          ),
          tags$div(
            style = "background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;",
            tags$p(strong(icon("file-export"), "OUTPUTS:"), style = "margin-bottom: 10px;"),
            tags$ul(
              style = "margin-bottom: 0; columns: 2;",
              tags$li(icon("check", style = "color: #28a745;"), " Correlation matrices"),
              tags$li(icon("check", style = "color: #28a745;"), " UMAP visualization"),
              tags$li(icon("check", style = "color: #28a745;"), " Clustering plots"),
              tags$li(icon("check", style = "color: #28a745;"), " Cluster stability"),
              tags$li(icon("check", style = "color: #28a745;"), " Feature heatmaps"),
              tags$li(icon("check", style = "color: #28a745;"), " Statistical analysis"),
              tags$li(icon("check", style = "color: #28a745;"), " Feature importance")
            )
          ),
          tags$div(
            class = "alert alert-info",
            style = "margin: 15px 0 0 0;",
            icon("lightbulb"), strong(" Recommended Parameters:"), " n_neighbors = 15 (UMAP), resolution = 0.42 (Leiden) for stable results."
          )
        )
      )
    )
  ),
  
  # Step 1: Feature Extraction
  "step1_params" = card(
    full_screen = TRUE,
    card_header(
      style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;",
      tags$h3(icon("image"), "Step 1: FEATURE EXTRACTION", style = "margin: 0; font-weight: 600;")
    ),
    tags$div(
      style = "padding: 20px;",
      tags$p(icon("info-circle"), "Extract 11 features from microscopy images containing segmentation masks and raw Rab images.", 
             style = "font-size: 1.05em; color: #555; margin-bottom: 25px;"),
      
      tags$div(
        class = "card bg-light",
        style = "padding: 20px; margin-bottom: 25px;",
        tags$h5(icon("folder-open"), "Input Folder Selection", style = "color: #667eea; margin-bottom: 15px;"),
        fluidRow(
          column(6,
                 shinyDirButton("input_folder", "Select Input Folder", "Choose folder", 
                               style = "background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px;")
          ),
          column(6,
                 verbatimTextOutput("input_folder_path")
          )
        )
      ),
      
      tags$div(
        class = "card",
        style = "padding: 20px; margin-bottom: 25px; border: 2px solid #e3e6f0;",
        tags$h5(icon("sliders-h"), "Extraction Parameters", style = "color: #667eea; margin-bottom: 15px;"),
        fluidRow(
          column(4, numericInput("min_spot_size", tags$span(icon("circle"), "Min Spot Size:"), value = 8, min = 1)),
          column(4, numericInput("neighbor_radius", tags$span(icon("bullseye"), "Neighbor Radius:"), value = 15, min = 1)),
          column(4, numericInput("n_jobs", tags$span(icon("cogs"), "Number of Jobs:"), value = 4, min = 1))
        )
      ),
      
      tags$div(
        class = "card",
        style = "padding: 20px; margin-bottom: 25px; border: 2px solid #e3e6f0;",
        tags$h5(icon("folder"), "Folder Structure", style = "color: #667eea; margin-bottom: 15px;"),
        tags$p(style = "color: #888; font-size: 0.9em;", icon("exclamation-triangle"), "Folder names must match your directory structure"),
        fluidRow(
          column(3, textInput("spot_folder", tags$span(icon("dot-circle"), "Spot Mask:"), value = "rab5_mask")),
          column(3, textInput("nucleus_folder", tags$span(icon("circle-notch"), "Nucleus Mask:"), value = "nucleus_mask")),
          column(3, textInput("cell_folder", tags$span(icon("draw-polygon"), "Cell Mask:"), value = "cell_mask")),
          column(3, textInput("rab_folder", tags$span(icon("image"), "Rab Images:"), value = "Rab5"))
        )
      ),
      
      tags$div(
        style = "text-align: center; margin-top: 30px;",
        actionButton("run_extraction", 
                    tags$span(icon("play-circle"), "Run Feature Extraction"), 
                    class = "btn-lg",
                    style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 40px; font-size: 1.1em; border-radius: 25px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);"),
        tags$br(), tags$br(),
        tags$div(
          textOutput("extraction_status"),
          style = "font-size: 1.1em; font-weight: 500; margin-top: 15px;"
        )
      )
    )
  ),
  
  # Step 2: KS Analysis
  "step2_params" = card(
    full_screen = TRUE,
    card_header(
      style = "background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;",
      tags$h3(icon("chart-line"), "Step 2: KS ANALYSIS", style = "margin: 0; font-weight: 600;")
    ),
    tags$div(
      style = "padding: 20px;",
      tags$p(icon("info-circle"), "Compare experimental conditions to a reference population using Kolmogorov-Smirnov statistic.", 
             style = "font-size: 1.05em; color: #555; margin-bottom: 15px;"),
      tags$div(
        class = "alert alert-info",
        style = "background: #e3f2fd; border-left: 4px solid #2196F3;",
        icon("lightbulb"), strong(" Note:"), " The reference population file should be a concatenation of all experimental conditions (e.g., Reference_populationRab5_V2.xlsx)."
      ),
      
      tags$div(
        class = "card",
        style = "padding: 20px; margin: 25px 0; border: 2px solid #e3e6f0;",
        tags$h5(icon("file-upload"), "File Upload", style = "color: #f5576c; margin-bottom: 20px;"),
        fluidRow(
          column(6,
                 fileInput("reference_file", 
                          tags$span(icon("database"), "Reference Population (Excel):"),
                          accept = c(".xlsx", ".xls"),
                          placeholder = "Reference_populationRab5_V2.xlsx",
                          buttonLabel = tags$span(icon("folder-open"), "Browse..."))
          ),
          column(6,
                 fileInput("comparison_files", 
                          tags$span(icon("files-o"), "Experimental Conditions (CSV/Excel):"),
                          accept = c(".csv", ".xlsx", ".xls"), 
                          multiple = TRUE,
                          placeholder = "FPW1.csv, JK2.csv, WK1.csv...",
                          buttonLabel = tags$span(icon("folder-open"), "Browse..."))
          )
        )
      ),
      
      tags$div(
        class = "card bg-light",
        style = "padding: 20px; margin-bottom: 25px;",
        tags$h5(icon("sliders-h"), "Analysis Parameters", style = "color: #f5576c; margin-bottom: 15px;"),
        fluidRow(
          column(6, textInput("id_column", tags$span(icon("tag"), "Cell Label Column Name:"), value = "Cell_label")),
          column(6, numericInput("ks_cores", tags$span(icon("microchip"), "Number of Cores:"), value = 4, min = 1))
        )
      ),
      
      tags$div(
        style = "text-align: center; margin-top: 30px;",
        actionButton("run_ks", 
                    tags$span(icon("calculator"), "Run KS Analysis"), 
                    class = "btn-lg",
                    style = "background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; padding: 15px 40px; font-size: 1.1em; border-radius: 25px; box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);"),
        tags$br(), tags$br(),
        tags$div(
          textOutput("ks_status"),
          style = "font-size: 1.1em; font-weight: 500; margin-top: 15px;"
        )
      )
    )
  ),
  
  "step2_results" = card(
    full_screen = TRUE,
    card_header(
      style = "background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); color: white;",
      tags$h4(icon("table"), "KS Analysis Results", style = "margin: 0; font-weight: 600;")
    ),
    DT::dataTableOutput("ks_results_table")
  ),
  
  # Step 3: Feature Filtering & Clustering
  "step3_load" = card(
    full_screen = TRUE,
    card_header(
      style = "background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;",
      tags$h3(icon("filter"), "Data Loading & Feature Filtering", style = "margin: 0; font-weight: 600;")
    ),
    tags$div(
      style = "padding: 20px;",
      tags$div(
        class = "alert alert-info",
        style = "background: #e8f5e9; border-left: 4px solid #4CAF50;",
        icon("info-circle"), " Upload the complete dataset containing all KS values with a 'Class' column (e.g., GlioCells_KSvaluesRab5WholeRef_V2.xlsx)."
      ),
      
      tags$div(
        class = "card",
        style = "padding: 20px; margin: 20px 0; border: 2px solid #e3e6f0;",
        fluidRow(
          column(8,
                 fileInput("clustering_input", 
                          tags$span(icon("database"), "Complete KS Dataset (Excel/CSV):"),
                          accept = c(".xlsx", ".csv"),
                          placeholder = "GlioCells_KSvaluesRab5WholeRef_V2.xlsx",
                          buttonLabel = tags$span(icon("folder-open"), "Browse..."))
          ),
          column(4,
                 numericInput("corr_threshold", 
                             tags$span(icon("percentage"), "Correlation Threshold:"),
                             value = 0.7, min = 0, max = 1, step = 0.05)
          )
        )
      ),
      
      tags$div(
        style = "text-align: center; margin-top: 30px;",
        actionButton("run_filtering", 
                    tags$span(icon("filter"), "Load Data & Filter Features"), 
                    class = "btn-lg",
                    style = "background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; padding: 15px 40px; font-size: 1.1em; border-radius: 25px; box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);"),
        tags$br(), tags$br(),
        tags$div(
          textOutput("filtering_status"),
          style = "font-size: 1.1em; font-weight: 500; margin-top: 15px;"
        )
      )
    )
  ),
  
  "step3_correlation" = card(
    h4("Correlation Matrices"),
    fluidRow(
      column(6, plotOutput("corr_original", height = "400px")),
      column(6, plotOutput("corr_filtered", height = "400px"))
    )
  ),
  
  "step3_clustering" = card(
    h4("UMAP & Leiden Clustering Parameters"),
    p("Recommended: n_neighbors = 15, resolution = 0.42 for stable clustering."),
    fluidRow(
      column(3, numericInput("n_neighbors", "N Neighbors (UMAP):", value = 15, min = 2)),
      column(3, numericInput("min_dist", "Min Distance (UMAP):", value = 0.1, min = 0, max = 1, step = 0.01)),
      column(3, numericInput("resolution", "Resolution (Leiden):", value = 0.42, min = 0.1, max = 2, step = 0.01)),
      column(3, numericInput("n_bootstrap", "N Bootstrap:", value = 100, min = 10))
    ),
    fluidRow(
      column(12,
             actionButton("run_clustering", "Run UMAP & Leiden Clustering", class = "btn-primary btn-lg"),
             tags$br(), tags$br(),
             textOutput("clustering_status")
      )
    )
  ),
  
  # Step 4: Visualization
  "step4_stability" = card(
    h4("Clustering Stability"),
    fluidRow(
      column(6, plotOutput("resolution_scan", height = "350px")),
      column(6, plotOutput("cluster_stability", height = "350px"))
    )
  ),
  
  "step4_umap" = card(
    h4("UMAP Visualization"),
    p("Visualize the dataset in UMAP space colored by clusters, condition, or feature values."),
    fluidRow(
      column(4, selectInput("umap_color", "Color by:", choices = c("Cluster", "Class"))),
      column(4, selectInput("feature_color", "Feature to display:", choices = NULL)),
      column(4, checkboxInput("discrete_color", "Discrete colors", value = TRUE))
    ),
    fluidRow(
      column(6, plotOutput("umap_plot", height = "400px")),
      column(6, plotOutput("umap_feature", height = "400px"))
    )
  ),
  
  "step4_proportions" = card(
    h4("Subpopulation Proportions"),
    fluidRow(
      column(12, plotOutput("proportions_plot", height = "400px"))
    ),
    fluidRow(
      column(12, DT::dataTableOutput("proportions_table"))
    )
  ),
  
  # Step 5: Statistical Analysis
  "step5_stats" = card(
    h4("Statistical Analysis"),
    fluidRow(
      column(12, plotOutput("cluster_stats", height = "500px"))
    )
  ),
  
  "step5_fingerprint" = card(
    h4("KS Cluster Fingerprint Heatmap"),
    fluidRow(
      column(12, plotOutput("fingerprint_heatmap", height = "500px"))
    )
  ),
  
  "step5_importance" = card(
    h4("Feature Importance Analysis"),
    fluidRow(
      column(12, plotOutput("feature_importance", height = "500px"))
    )
  ),
  
  "export" = card(
    card_header(
      style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;",
      tags$h3(icon("download"), "Export Results", style = "margin: 0; font-weight: 600;")
    ),
    tags$div(
      style = "padding: 40px; text-align: center;",
      tags$p(style = "font-size: 1.1em; color: #555; margin-bottom: 40px;",
             icon("info-circle"), " Download your analysis results in various formats for further processing or publication."),
      fluidRow(
        column(4, 
               tags$div(
                 style = "padding: 20px;",
                 downloadButton("download_filtered", 
                               tags$span(icon("table"), tags$br(), "Filtered Data"),
                               style = "background: #667eea; color: white; border: none; padding: 30px 20px; width: 100%; font-size: 1.1em; border-radius: 10px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);")
               )
        ),
        column(4, 
               tags$div(
                 style = "padding: 20px;",
                 downloadButton("download_umap", 
                               tags$span(icon("project-diagram"), tags$br(), "UMAP Results"),
                               style = "background: #f5576c; color: white; border: none; padding: 30px 20px; width: 100%; font-size: 1.1em; border-radius: 10px; box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);")
               )
        ),
        column(4, 
               tags$div(
                 style = "padding: 20px;",
                 downloadButton("download_stats", 
                               tags$span(icon("chart-bar"), tags$br(), "Statistics"),
                               style = "background: #38f9d7; color: white; border: none; padding: 30px 20px; width: 100%; font-size: 1.1em; border-radius: 10px; box-shadow: 0 4px 15px rgba(56, 249, 215, 0.3);")
               )
        )
      )
    )
  )
)

# Main UI
ui <- page_navbar(
  theme = bslib::bs_theme(
    preset = "minty",
    primary = "#2C3E50",
    secondary = "#18BC9C",
    success = "#18BC9C",
    info = "#3498DB",
    warning = "#F39C12",
    danger = "#E74C3C",
    base_font = bslib::font_google("Roboto"),
    heading_font = bslib::font_google("Montserrat")
  ),
  id = "NavBar",
  title = tags$span(
    icon("microscope"),
    "RabAnalyser",
    style = "font-weight: 600; font-size: 1.3em;"
  ),
  nav_spacer(),
  
  nav_panel(
    title = tags$span(icon("home"), "Home"),
    value = "home",
    fluidRow(column(width = 12, cards$home))
  ),
  
  nav_panel(
    title = tags$span(icon("image"), "STEP 1: Feature Extraction"),
    value = "step1",
    fluidRow(column(width = 12, cards$step1_params))
  ),
  
  nav_panel(
    title = tags$span(icon("chart-line"), "STEP 2: KS Analysis"),
    value = "step2",
    fluidRow(column(width = 12, cards$step2_params)),
    fluidRow(column(width = 12, cards$step2_results))
  ),
  
  nav_panel(
    title = tags$span(icon("project-diagram"), "STEP 3: Clustering & Analysis"),
    value = "step3",
    tags$div(
      style = "background: linear-gradient(to right, #667eea 0%, #764ba2 100%); padding: 15px; margin-bottom: 20px; border-radius: 8px;",
      tags$h2(icon("filter"), "Feature Filtering", style = "color: white; margin: 0;")
    ),
    fluidRow(column(width = 12, cards$step3_load)),
    fluidRow(column(width = 12, cards$step3_correlation)),
    
    tags$div(
      style = "background: linear-gradient(to right, #f093fb 0%, #f5576c 100%); padding: 15px; margin: 30px 0 20px 0; border-radius: 8px;",
      tags$h2(icon("sitemap"), "Clustering", style = "color: white; margin: 0;")
    ),
    fluidRow(column(width = 12, cards$step3_clustering)),
    fluidRow(column(width = 12, cards$step4_stability)),
    
    tags$div(
      style = "background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); padding: 15px; margin: 30px 0 20px 0; border-radius: 8px;",
      tags$h2(icon("eye"), "Visualization", style = "color: white; margin: 0;")
    ),
    fluidRow(column(width = 12, cards$step4_umap)),
    fluidRow(column(width = 12, cards$step4_proportions)),
    
    tags$div(
      style = "background: linear-gradient(to right, #43e97b 0%, #38f9d7 100%); padding: 15px; margin: 30px 0 20px 0; border-radius: 8px;",
      tags$h2(icon("chart-bar"), "Statistical Analysis", style = "color: white; margin: 0;")
    ),
    fluidRow(column(width = 12, cards$step5_stats)),
    fluidRow(column(width = 12, cards$step5_fingerprint)),
    fluidRow(column(width = 12, cards$step5_importance))
  ),
  
  nav_panel(
    title = tags$span(icon("download"), "Export Results"),
    value = "export",
    fluidRow(column(width = 12, cards$export))
  ),
  
  nav_spacer(),
  nav_item(tags$a(
    icon("github"), "GitHub",
    href = "https://github.com/qBioTurin/RabAnalyser",
    target = "_blank",
    style = "color: inherit;"
  ))
)

# shinyApp(ui, server,
#          options =  options(shiny.maxRequestSize=1000*1024^2,
#                             shiny.launch.browser = .rs.invokeShinyWindowExternal)
# )
