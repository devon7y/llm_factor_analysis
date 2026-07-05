# 01_fit.R -- semantic factor analysis of every scale (extraction model)
#
# For each scale this script:
#   1. embeds the raw item text with the extraction model
#      (Qwen3-Embedding-8B; no instruction on the extraction side),
#   2. fits the confirmatory model: semantic factor analysis with the
#      factor count fixed to the instrument's documented count (k_doc),
#   3. fits the exploratory model: the same analysis with the factor count
#      retained by embedding parallel analysis (Horn-style, 100 random
#      unit-vector iterations, 95th percentile, seed 42),
#   4. names both fits with the EXTRACTION model as well (the single-encoder
#      column of the benchmark), at fp16 and int8 pool precision,
#   5. saves the fits, the item embeddings, and the labels.
#
# The naming model's labels (the paper's headline column) are produced in
# 02_name.R, after this script releases the extraction model from the GPU.

source(file.path("R", "00_config.R"))

# Candidate pools for BOTH models are fetched up front, before any GPU
# work, so a transient download failure cannot waste a partly finished
# allocation. Downloads land once in the semanticfa cache; the cache
# location is controlled by R_USER_CACHE_DIR (set in the SLURM script).
pool_q_fp16 <- sfa_pool(MODEL_EXTRACT, precision = "fp16", download = TRUE)
pool_q_int8 <- sfa_pool(MODEL_EXTRACT, precision = "int8", download = TRUE)
invisible(sfa_pool(MODEL_NAME, precision = "fp16", download = TRUE))
invisible(sfa_pool(MODEL_NAME, precision = "int8", download = TRUE))

np <- reticulate::import("numpy", convert = FALSE)

for (s in SCALES$scale) {
  message("== ", s, " ==")
  df <- read_scale(s)
  k  <- k_doc_for(s)

  # ---- confirmatory fit (documented factor count) -------------------------
  fit_c <- sfa(df, model = MODEL_EXTRACT, nfactors = k)

  # ---- exploratory fit (embedding parallel analysis) ----------------------
  # The items were just embedded, so this reuses the cached embeddings.
  fit_e <- sfa(df, model = MODEL_EXTRACT,
               n_factors_method = "parallel",
               parallel_iter = 100L, seed = 42L)

  saveRDS(fit_c, file.path(DIR_FITS, paste0(s, "_confirmatory.rds")))
  saveRDS(fit_e, file.path(DIR_FITS, paste0(s, "_exploratory.rds")))

  # ---- save the raw item embeddings (shipped on OSF) ----------------------
  emb <- fit_c$input_embeddings
  saveRDS(emb, file.path(DIR_EMB, paste0(s, "_items_",
          gsub("[^A-Za-z0-9.]+", "-", MODEL_EXTRACT), ".rds")))
  np$savez_compressed(
    file.path(DIR_EMB, paste0(s, "_items_",
              gsub("[^A-Za-z0-9.]+", "-", MODEL_EXTRACT), ".npz")),
    embeddings = np$asarray(reticulate::r_to_py(unname(emb)), dtype = "float16"),
    codes  = reticulate::r_to_py(df$code),
    items  = reticulate::r_to_py(df$item),
    factors = reticulate::r_to_py(df$factor)
  )

  # ---- single-encoder naming (extraction model names its own factors) -----
  lab_c_fp16 <- sfa_name(fit_c, model = MODEL_EXTRACT, pool = pool_q_fp16)
  lab_c_int8 <- sfa_name(fit_c, model = MODEL_EXTRACT, pool = pool_q_int8)
  lab_e_fp16 <- sfa_name(fit_e, model = MODEL_EXTRACT, pool = pool_q_fp16)

  save_json(labels_to_list(lab_c_fp16),
            file.path(DIR_LABELS, paste0(s, "_confirmatory_qwen_fp16.json")))
  save_json(labels_to_list(lab_c_int8),
            file.path(DIR_LABELS, paste0(s, "_confirmatory_qwen_int8.json")))
  save_json(labels_to_list(lab_e_fp16),
            file.path(DIR_LABELS, paste0(s, "_exploratory_qwen_fp16.json")))

  message(s, ": k_doc = ", k, ", k_explore = ", fit_e$factors)
}

message("01_fit.R complete.")
