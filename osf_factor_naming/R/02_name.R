# 02_name.R -- factor naming with the naming model (headline analyses)
#
# Loads the fits from 01_fit.R and names every factor with the naming model
# (Harrier-OSS-27B). Loading this model evicts the extraction model from
# the GPU (semanticfa keeps a single resident encoder), which is why the
# naming analyses are grouped in this second script.
#
# Analyses produced here, per scale:
#   a. HEADLINE: confirmatory fit, naming model, fp16 pool
#   b. precision robustness: same, int8 pool
#   c. instruction ensemble: same, fp16 pool, the three instruction
#      rewordings (the default instruction is analysis (a))
#   d. exploratory column: exploratory fit, naming model, fp16 pool
#   e. determinism check: analysis (a) repeated with the embedding cache
#      disabled, so the items are re-embedded from scratch on the GPU
#
# Every call is deterministic given the item embeddings; (e) verifies that
# the embeddings themselves reproduce on re-computation within this
# hardware/software configuration.

source(file.path("R", "00_config.R"))

pool_h_fp16 <- sfa_pool(MODEL_NAME, precision = "fp16", download = TRUE)
pool_h_int8 <- sfa_pool(MODEL_NAME, precision = "int8", download = TRUE)

np <- reticulate::import("numpy", convert = FALSE)

for (s in SCALES$scale) {
  message("== ", s, " ==")
  fit_c <- readRDS(file.path(DIR_FITS, paste0(s, "_confirmatory.rds")))
  fit_e <- readRDS(file.path(DIR_FITS, paste0(s, "_exploratory.rds")))

  # (a) headline: two-encoder pipeline at research precision
  lab_a <- sfa_name(fit_c, model = MODEL_NAME, pool = pool_h_fp16)
  save_json(labels_to_list(lab_a),
            file.path(DIR_LABELS, paste0(s, "_confirmatory_harrier_fp16.json")))

  # (b) pool-precision robustness (queries identical; pool int8)
  lab_b <- sfa_name(fit_c, model = MODEL_NAME, pool = pool_h_int8)
  save_json(labels_to_list(lab_b),
            file.path(DIR_LABELS, paste0(s, "_confirmatory_harrier_int8.json")))

  # (c) instruction ensemble: rewordings 2-4 (1 is the default, analysis a)
  for (v in 2:4) {
    lab_v <- suppressWarnings(
      sfa_name(fit_c, model = MODEL_NAME, pool = pool_h_fp16,
               instruction = INSTRUCTION_VARIANTS[v])
    )
    save_json(labels_to_list(lab_v),
              file.path(DIR_LABELS,
                        paste0(s, "_confirmatory_harrier_fp16_instr", v, ".json")))
  }

  # (d) exploratory column
  lab_d <- sfa_name(fit_e, model = MODEL_NAME, pool = pool_h_fp16)
  save_json(labels_to_list(lab_d),
            file.path(DIR_LABELS, paste0(s, "_exploratory_harrier_fp16.json")))

  # (e) determinism: re-embed the naming queries from scratch (cache off).
  # This is the one analysis that must run on the original GPU: it verifies
  # that recomputing the embeddings reproduces the labels. It is skipped on
  # CPU-only reruns (the shipped-cache replication path) because its whole
  # point is fresh GPU inference.
  has_gpu <- tryCatch(
    reticulate::import("torch")$cuda$is_available(), error = function(e) FALSE)
  if (isTRUE(has_gpu)) {
    lab_e <- sfa_name(fit_c, model = MODEL_NAME, pool = pool_h_fp16,
                      cache = FALSE)
    save_json(labels_to_list(lab_e),
              file.path(DIR_LABELS, paste0(s, "_confirmatory_harrier_fp16_rerun.json")))
  } else {
    message(s, ": no GPU; skipping the re-embedding determinism check (e).")
  }

  # (f) parity column for the development reference: the development
  # cross-pipeline artifact predates the duplicate-label (collision)
  # resolution step, so an exact comparison against it needs the keeper
  # switched off. Not used for any number in the paper.
  lab_f <- sfa_name(fit_c, model = MODEL_NAME, pool = pool_h_fp16,
                    collision = FALSE)
  save_json(labels_to_list(lab_f),
            file.path(DIR_LABELS, paste0(s, "_confirmatory_harrier_fp16_nokeeper.json")))

  # ---- save the naming-space (instruction-conditioned) item embeddings ----
  # These are the queries analysis (a) retrieved with; shipped on OSF so the
  # naming stage can be reproduced without the 27B model.
  queries <- sprintf("Instruct: %s\nQuery: %s",
                     sfa_naming_instruction(), fit_c$item_data$item)
  qemb <- sfa_embed(queries, model = MODEL_NAME)   # cache hit from (a)
  np$savez_compressed(
    file.path(DIR_EMB, paste0(s, "_queries_",
              gsub("[^A-Za-z0-9.]+", "-", MODEL_NAME), ".npz")),
    embeddings = np$asarray(reticulate::r_to_py(unname(qemb)), dtype = "float16"),
    codes = reticulate::r_to_py(fit_c$item_data$code),
    items = reticulate::r_to_py(fit_c$item_data$item)
  )
}

message("02_name.R complete.")
