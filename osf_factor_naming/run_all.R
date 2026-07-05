# run_all.R -- run the complete factor-naming analysis pipeline
#
# Usage (from the osf_factor_naming/ directory):
#   Rscript run_all.R
#
# Stages (each is an independent script under R/):
#   01_fit.R        embed items + fit confirmatory/exploratory models (GPU)
#   02_name.R       name every factor with the naming model (GPU)
#   03_evaluate.R   evaluate labels against the frozen alias table (CPU)
#   04_retention.R  retention-method battery, supplementary (CPU)
#   05_emit.R       LaTeX macros, tables, figures, shipped cache (CPU)
#
# A GPU is required only for 01 and 02, and only when the embedding cache
# (embeddings/cache/) has not been restored into the semanticfa cache
# directory; with the cache restored, the whole pipeline runs on a laptop
# and reproduces the shipped results bit-for-bit (see README).

t0 <- Sys.time()
for (step in c("01_fit.R", "02_name.R", "03_evaluate.R",
               "04_retention.R", "05_emit.R")) {
  message("\n########## ", step, " ##########")
  source(file.path("R", step))
}
message("\nPipeline complete in ",
        round(difftime(Sys.time(), t0, units = "mins"), 1), " minutes.")
