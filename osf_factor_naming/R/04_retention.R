# 04_retention.R -- factor-retention battery (supplementary; CPU only)
#
# The paper's primary presentation fixes each scale's factor count to the
# instrument's documented count, isolating the naming question from the
# contested retention question. This script documents what a data-driven
# retention decision would have returned for each scale, across the
# retention methods implemented in semanticfa: embedding parallel analysis,
# the Kaiser criterion, TEFI, the empirical Kaiser criterion (EKC), and
# Velicer's MAP. EGA is included when the EGAnet package is available.

source(file.path("R", "00_config.R"))

# Fixed method list so the battery's output does not depend on which
# optional packages happen to be installed (EGA, which needs EGAnet, is
# therefore not part of the shipped battery).
methods <- c("parallel", "kaiser", "TEFI", "EKC", "MAP")

out <- list()
for (s in SCALES$scale) {
  fit <- readRDS(file.path(DIR_FITS, paste0(s, "_confirmatory.rds")))
  nf <- sfa_nfactors(fit, methods = methods, seed = 42L, parallel_iter = 100L)
  ks <- stats::setNames(nf$methods$n_factors, nf$methods$method)
  out[[s]] <- c(list(k_doc = k_doc_for(s), consensus = nf$consensus),
                as.list(ks))
  message(s, ": documented ", k_doc_for(s), " | ",
          paste(names(ks), ks, sep = "=", collapse = " "))
}

save_json(out, file.path(DIR_RESULTS, "retention_battery.json"))
message("04_retention.R complete.")
