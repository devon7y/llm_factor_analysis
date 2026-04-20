#!/usr/bin/env Rscript

library(psych)
library(EFAtools)

ENABLE_POLYCHORIC <- FALSE

data_file <- "/Users/devon7y/VS Code/LLM_Factor_Analysis/scale_responses/HEXACO_data.csv"
items_file <- "/Users/devon7y/VS Code/LLM_Factor_Analysis/scale_items/HEXACO_items.csv"

df <- read.csv(data_file, check.names = FALSE)
if (ncol(df) == 1) df <- read.delim(data_file, check.names = FALSE)

items <- read.csv(items_file, check.names = FALSE)
codes <- items$code
scoring <- items$scoring

X <- df[, codes, drop = FALSE]

for (i in seq_along(scoring)) {
  if (scoring[i] == -1) {
    max_val <- max(X[[i]], na.rm = TRUE)
    X[[i]] <- max_val - X[[i]]
  }
}

X <- X[complete.cases(X), , drop = FALSE]

cat("X shape:", nrow(X), "x", ncol(X), "\n")
cat("Missing values:", sum(is.na(X)), "\n")
cat("First codes:", paste(colnames(X)[1:6], collapse = ", "), "\n")

row_cap <- 10000
if (nrow(X) > row_cap) {
  set.seed(42)
  X_work <- X[sample.int(nrow(X), row_cap), , drop = FALSE]
  cat("Using", nrow(X_work), "rows for correlation-based analyses (cap", row_cap, ").\n")
} else {
  X_work <- X
}

cor_mat <- stats::cor(X_work)
cat("Non-finite in cor:", sum(!is.finite(cor_mat)), "\n")
obs_eigs <- sort(eigen(cor_mat, only.values = TRUE)$values, decreasing = TRUE)

set.seed(42)
n_iter <- 1000
percentile <- 0.99
n_items <- ncol(X_work)
n_obs <- nrow(X_work)
n_obs_sim <- min(n_obs, 1000)
if (n_obs_sim < n_obs) {
  cat("Using", n_obs_sim, "rows for parallel analysis simulations (cap for speed).\n")
}
rand_eigs <- matrix(NA_real_, n_iter, n_items)
for (i in seq_len(n_iter)) {
  rand_data <- matrix(rnorm(n_obs_sim * n_items), n_obs_sim, n_items)
  rand_corr <- cor(rand_data)
  rand_eigs[i, ] <- sort(eigen(rand_corr, only.values = TRUE)$values, decreasing = TRUE)
}
pct_eigs <- apply(rand_eigs, 2, quantile, probs = percentile)
nf <- sum(obs_eigs > pct_eigs)

cat("Obs eigs (top 6):", paste(round(obs_eigs[1:6], 4), collapse = ", "), "\n")
cat("Pct eigs (top 6):", paste(round(pct_eigs[1:6], 4), collapse = ", "), "\n")

n_max <- min(10, ncol(X))
vss <- psych::VSS(X_work, n = n_max, rotate = "oblimin", fm = "minres", n.obs = n_obs, plot = FALSE)
nf_map <- which.min(vss$map)

hull <- EFAtools::HULL(X_work, method = "ULS", gof = "CFI", use = "everything")

if (ENABLE_POLYCHORIC) {
  max_categories <- max(sapply(X_work, function(x) length(unique(x))))
  if (max_categories > 8) {
    nf_poly <- NA
    cat("Skipping polychoric PA: max categories =", max_categories, "\n")
  } else {
    poly_rho <- psych::polychoric(X_work)$rho
    fa_par_poly <- psych::fa.parallel(poly_rho, fa = "fa", fm = "minres", n.obs = n_obs_sim, plot = FALSE)
    nf_poly <- fa_par_poly$nfact
  }
} else {
  nf_poly <- NA
  cat("Polychoric PA disabled.\n")
}

fa_fit <- psych::fa(
  r = cor_mat,
  nfactors = nf,
  fm = "minres",
  rotate = "oblimin",
  n.obs = n_obs
)

cat("Suggested number of factors (parallel analysis):", nf, "\n")
cat("Suggested number of factors (MAP):", nf_map, "\n")
cat("Suggested number of factors (Hull, CFI):", hull$n_fac_CFI, "\n")
cat("Suggested number of factors (parallel analysis, polychoric):", nf_poly, "\n")
fa_fit
