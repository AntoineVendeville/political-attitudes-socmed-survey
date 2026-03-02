# --- Load required packages ---
library(survey)
library(jtools)
library(reshape2)
library(ggplot2)



# --- Initialization ---

## Load the data
data <- read.csv("data/ESS11_preprocessed.csv")

## Define variables of interest 
vars <- c('Left.Right', 'Immigration', 'Environment', 'Social.liberalism', 'Redistribution',
          'Anti.elitism', 'EU.integration', 'Direct.democracy')
names(data) <- make.names(names(data)) # Rename variables to valid R names

## Allow strata with only 1 PSU
options(survey.lonely.psu = "adjust")

## Define survey design
design <- svydesign(
  ids = ~psu,        # PSU variable
  strata = ~stratum, # Stratum variable
  weights = ~anweight, # Sampling weight
  data = data,
  nest = TRUE
)

# --- Correlations ---
cor_results <- svycor(as.formula(paste("~", paste(vars, collapse = " + "))), design, sig.stats=TRUE)
cor_mat <- cor_results$cors # Extract correlation matrix
p_mat <- cor_results$p.values # Extract p-values
sig_mask <- p_mat < 0.05  # Define significance mask # TRUE = significant
cor_df <- melt(cor_mat) # Melt correlation matrix
names(cor_df) <- c("Var1", "Var2", "Correlation")

## Add significance
p_df <- melt(sig_mask)
cor_df$Significant <- p_df$value

## save
write.csv(cor_mat, "ESS_results/Corr_matrix.csv", row.names = TRUE)
write.csv(p_mat, "ESS_results/Corr_pval.csv", row.names = TRUE)



# --- Principal Component Analysis (PCA) ---

## Perform survey-weighted PCA
svy_pca <- svyprcomp(
  formula = as.formula(paste("~", paste(vars, collapse = " + "))),
  design = design,
  scale. = TRUE # standardize variables before PCA
)

## Save coordinates of each individual in PCA space
pca_scores <- predict(svy_pca, newdata = data)
pca_coords <- cbind(ID = 1:nrow(data), pca_scores)
write.csv(pca_coords, "ESS_results/PCA_coordinates.csv", row.names = FALSE)

## Extract and save variable loadings
pca_loadings <- as.data.frame(svy_pca$rotation)
pca_loadings <- cbind(Variable = rownames(pca_loadings), pca_loadings)
rownames(pca_loadings) <- NULL
write.csv(pca_loadings, "ESS_results/PCA_loadings.csv", row.names = FALSE)

## Extract and save variance explained
eigvals <- svy_pca$sdev^2
explained_var <- eigvals / sum(eigvals)
explained_df <- data.frame(
  PC = paste0("PC", seq_along(eigvals)),
  Eigenvalue = eigvals,
  Proportion = explained_var,
  Cumulative = cumsum(explained_var)
)
write.csv(explained_df, "ESS_results/PCA_explained_variance.csv", row.names = FALSE)


# --- Get survey-weighted user counts for each variable ---
weighted_counts_list <- list()

## Loop through each variable
for (var in vars) {
  # Create a formula like ~Left.Right
  fml <- as.formula(paste0("~", var))
  
  # Compute weighted counts
  tbl <- svytable(fml, design)
  
  # Convert to data frame
  df <- as.data.frame(tbl)
  colnames(df) <- c("value", "weighted_count")
  df$variable <- var
  
  # Store
  weighted_counts_list[[var]] <- df
}

## Combine all into one data frame
all_counts <- do.call(rbind, weighted_counts_list)

## Reorder columns for clarity
all_counts <- all_counts[, c("variable", "value", "weighted_count")]

## Save to CSV
write.csv(all_counts, "ESS_results/weighted_counts_by_variable.csv", row.names = FALSE)

          