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
    h1("RabAnalyser Workflow"),
    h2("Welcome to RabAnalyser Shiny App"),
    p("This application implements a complete three-step workflow for single-cell Rab protein analysis:"),
    tags$hr(),
    h3("Step 1: FEATURE EXTRACTION"),
    p(strong("Description:"), "The framework extracts 11 features from microscopy images."),
    p(strong("INPUT:"), "Experimental conditions (e.g., FPW1, JK2, WK1) containing segmentation masks:"),
    tags$ul(
      tags$li("Cell segmentation mask (e.g., cell_mask)"),
      tags$li("Nucleus segmentation mask (e.g., nucleus_mask)"),
      tags$li("Rab spots segmentation mask (e.g., Rab5_mask)"),
      tags$li("Raw images of Rab spots (e.g., Rab5)"),
      tags$li("Excel file with feature names (e.g., Feature_extraction_labels2.xlsx)")
    ),
    p(strong("OUTPUT:"), "CSV files where each row represents a Rab spot with Cell_ID and feature values (e.g., FPW1.csv, JK2.csv, WK1.csv)."),
    tags$hr(),
    h3("Step 2: KS ANALYSIS"),
    p(strong("Description:"), "Identify a reference population and compare experimental conditions using the Kolmogorov-Smirnov (KS) statistic."),
    p(strong("INPUT:"), "Excel files containing extracted features (e.g., FPW1.xlsx, JK2.xlsx, WK1.xlsx) and a reference population file (e.g., Reference_populationRab5_V2.xlsx) - a concatenation of all experimental conditions to serve as the reference distribution. Also requires Feature_labels2.xlsx with the 11 feature names."),
    p(strong("OUTPUT:"), "Excel files where each row is a single cell with KS values representing distributional dissimilarity between the reference and single-cell distributions (e.g., FPW1_KSRab5_WholeRef_V2.xlsx, JK2_KSRab5_WholeRef_V2.xlsx, WK1_KSRab5_WholeRef_V2.xlsx)."),
    tags$hr(),
    h3("Step 3: DATA CLUSTERING AND DOWNSTREAM ANALYSIS"),
    p(strong("Description:"), "Perform data clustering and characterize identified clusters."),
    p(strong("INPUT:"), "Complete dataset containing all KS values per feature and per condition (e.g., GlioCells_KSvaluesRab5WholeRef_V2.xlsx) - concatenation of all KS analysis outputs with a 'Class' column specifying the condition/subtype."),
    p(strong("OUTPUTS:")),
    tags$ul(
      tags$li("Correlation matrices before and after feature selection"),
      tags$li("UMAP visualization of the dataset"),
      tags$li("Number of clusters vs resolution parameter plot"),
      tags$li("Best data clustering visualization"),
      tags$li("Cluster stability (Jaccard indices boxplot)"),
      tags$li("KS values per feature in UMAP space"),
      tags$li("Statistical comparison of KS values across clusters"),
      tags$li("Heatmap of KS values per cluster and feature"),
      tags$li("Feature importance analysis")
    ),
    p(strong("Parameters:"), "Recommended n_neighbors = 15 for UMAP and resolution = 0.42 for Leiden clustering for stable results.")
  ),
  
  # Step 1: Feature Extraction
  "step1_params" = card(
    h3("Step 1: FEATURE EXTRACTION"),
    p("Extract 11 features from microscopy images containing segmentation masks and raw Rab images."),
    fluidRow(
      column(6,
             shinyDirButton("input_folder", "Select Input Folder", "Choose folder containing experimental condition images")
      ),
      column(6,
             verbatimTextOutput("input_folder_path")
      )
    ),
    h4("Parameters"),
    fluidRow(
      column(3, numericInput("min_spot_size", "Min Spot Size:", value = 8, min = 1)),
      column(3, numericInput("neighbor_radius", "Neighbor Radius:", value = 15, min = 1)),
      column(3, numericInput("n_jobs", "Number of Jobs:", value = 4, min = 1))
    ),
    h4("Folder Names (must match your directory structure)"),
    fluidRow(
      column(3, textInput("spot_folder", "Spot Mask Folder:", value = "rab5_mask")),
      column(3, textInput("nucleus_folder", "Nucleus Mask Folder:", value = "nucleus_mask")),
      column(3, textInput("cell_folder", "Cell Mask Folder:", value = "cell_mask")),
      column(3, textInput("rab_folder", "Rab Raw Images Folder:", value = "Rab5"))
    ),
    fluidRow(
      column(12, 
             actionButton("run_extraction", "Run Feature Extraction", class = "btn-primary btn-lg"),
             tags$br(), tags$br(),
             textOutput("extraction_status")
      )
    )
  ),
  
  # Step 2: KS Analysis
  "step2_params" = card(
    h3("Step 2: KS ANALYSIS"),
    p("Compare experimental conditions to a reference population using Kolmogorov-Smirnov statistic."),
    p(strong("Note:"), "The reference population file should be a concatenation of all experimental conditions (e.g., Reference_populationRab5_V2.xlsx)."),
    fluidRow(
      column(6,
             fileInput("reference_file", "Upload Reference Population (Excel):", 
                       accept = c(".xlsx", ".xls"),
                       placeholder = "e.g., Reference_populationRab5_V2.xlsx")
      ),
      column(6,
             fileInput("comparison_files", "Upload Experimental Condition Files (CSV/Excel):", 
                       accept = c(".csv", ".xlsx", ".xls"), multiple = TRUE,
                       placeholder = "e.g., FPW1.csv, JK2.csv, WK1.csv")
      )
    ),
    h4("Parameters"),
    fluidRow(
      column(4, textInput("id_column", "Cell Label Column Name:", value = "Cell_label")),
      column(4, numericInput("ks_cores", "Number of Cores:", value = 4, min = 1)),
      column(4, 
             tags$br(),
             actionButton("run_ks", "Run KS Analysis", class = "btn-primary btn-lg"))
    ),
    fluidRow(
      column(12, 
             tags$br(),
             textOutput("ks_status"))
    )
  ),
  
  "step2_results" = card(
    h4("KS Analysis Results"),
    DT::dataTableOutput("ks_results_table")
  ),
  
  # Step 3: Feature Filtering & Clustering
  "step3_load" = card(
    h3("Step 3: DATA CLUSTERING AND DOWNSTREAM ANALYSIS"),
    p("Perform clustering analysis on KS values to identify and characterize cell subpopulations."),
    p(strong("Note:"), "Upload the complete dataset containing all KS values with a 'Class' column (e.g., GlioCells_KSvaluesRab5WholeRef_V2.xlsx)."),
    fluidRow(
      column(8,
             fileInput("clustering_input", "Upload Complete KS Dataset (Excel/CSV):", 
                       accept = c(".xlsx", ".csv"),
                       placeholder = "e.g., GlioCells_KSvaluesRab5WholeRef_V2.xlsx")
      ),
      column(4,
             numericInput("corr_threshold", "Correlation Threshold:", 
                         value = 0.7, min = 0, max = 1, step = 0.05)
      )
    ),
    fluidRow(
      column(12,
             actionButton("run_filtering", "Load Data & Filter Correlated Features", class = "btn-primary btn-lg"),
             tags$br(), tags$br(),
             textOutput("filtering_status")
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
    h4("Export Results"),
    fluidRow(
      column(4, downloadButton("download_filtered", "Download Filtered Data")),
      column(4, downloadButton("download_umap", "Download UMAP Results")),
      column(4, downloadButton("download_stats", "Download Statistics"))
    )
  )
)

# Main UI
ui <- page_navbar(
  theme = bslib::bs_theme(preset = "united"),
  id = "NavBar",
  title = "RabAnalyser",
  nav_spacer(),
  
  nav_panel("Home", value = "home",
            fluidRow(column(width = 12, cards$home))
  ),
  
  nav_panel("STEP 1: Feature Extraction", value = "step1",
            fluidRow(column(width = 12, cards$step1_params))
  ),
  
  nav_panel("STEP 2: KS Analysis", value = "step2",
            fluidRow(column(width = 12, cards$step2_params)),
            fluidRow(column(width = 12, cards$step2_results))
  ),
  
  nav_panel("STEP 3: Clustering & Analysis", value = "step3",
            h2("Feature Filtering"),
            fluidRow(column(width = 12, cards$step3_load)),
            fluidRow(column(width = 12, cards$step3_correlation)),
            tags$hr(),
            h2("Clustering"),
            fluidRow(column(width = 12, cards$step3_clustering)),
            fluidRow(column(width = 12, cards$step4_stability)),
            tags$hr(),
            h2("Visualization"),
            fluidRow(column(width = 12, cards$step4_umap)),
            fluidRow(column(width = 12, cards$step4_proportions)),
            tags$hr(),
            h2("Statistical Analysis"),
            fluidRow(column(width = 12, cards$step5_stats)),
            fluidRow(column(width = 12, cards$step5_fingerprint)),
            fluidRow(column(width = 12, cards$step5_importance))
  ),
  
  nav_panel("Export Results", value = "export",
            fluidRow(column(width = 12, cards$export))
  ),
  
  nav_item(tags$a("GitHub", href = "https://github.com/qBioTurin/RabAnalyser", target = "_blank"))
)

# shinyApp(ui, server,
#          options =  options(shiny.maxRequestSize=1000*1024^2,
#                             shiny.launch.browser = .rs.invokeShinyWindowExternal)
# )
