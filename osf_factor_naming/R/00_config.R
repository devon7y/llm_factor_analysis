# 00_config.R -- shared configuration for the factor-naming analyses
#
# Everything the pipeline depends on is fixed here: the scale roster, the
# documented (confirmatory) factor counts, the two embedding models, the
# pool precision, and the output locations. All later scripts source this
# file first. Nothing in the pipeline depends on interactive input; running
# run_all.R from the repository root reproduces every number in the paper.
#
# Reproducibility notes
# - Extraction and naming are deterministic given the item embeddings.
#   Item embeddings are computed on a GPU; bitwise values can differ across
#   GPU hardware/driver stacks, so we ship the embeddings we computed (see
#   embeddings/ and the semanticfa cache in embeddings/cache/). Restoring
#   that cache reproduces the full pipeline bit-for-bit with no GPU.
# - Model weights are loaded in bfloat16 (their native training dtype).
#   The candidate-pool embeddings and all shipped item embeddings are
#   stored in float16 ("fp16"); pool precision is set below.

suppressPackageStartupMessages({
  library(semanticfa)   # >= 0.2.0
  library(jsonlite)
})

# ---- models -----------------------------------------------------------------
# Two-encoder division of labor (see paper, Method): the extraction model
# recovers factor structure from plain item text; the naming model retrieves
# construct names for the factors under the naming instruction.
MODEL_EXTRACT <- "Qwen/Qwen3-Embedding-8B"
MODEL_NAME    <- "microsoft/harrier-oss-v1-27b"
POOL_PRECISION <- "fp16"    # research-parity precision; int8 is the package
                            # default for smaller downloads and is compared
                            # against fp16 as a robustness analysis (02/03)

# Weight dtype for the sentence-transformers backend (semanticfa option).
# bfloat16 is the native dtype of both models; float32 would not fit the
# 27B naming model on a single 80 GB GPU.
options(semanticfa.torch_dtype = "bfloat16")

# Pool downloads are several GB; R's default 60 s timeout is too short.
options(timeout = 7200)

# ---- scales -----------------------------------------------------------------
# The 11-scale benchmark. `k_doc` is the instrument's documented factor
# count, used for the confirmatory (fixed-k) presentation; the exploratory
# presentation retains factors by embedding parallel analysis instead.
SCALES <- data.frame(
  scale = c("DASS", "Big5", "431PTQ", "MFQ", "BAI", "TMA",
            "BDI", "EPQ", "OSRI", "TAS-20", "AQ-50"),
  k_doc = c(3L, 5L, 2L, 5L, 1L, 1L, 1L, 5L, 2L, 3L, 5L),
  stringsAsFactors = FALSE
)

# ---- pool provenance constants ----------------------------------------------
# Fixed design constants of the candidate-pool build (documented in the
# semanticfa pool build manifest; see the package's data-raw/ tools).
POOL_CENSUS_BASE <- 1001041L   # WordNet + filtered Wikipedia title census
POOL_ADDITIONS   <- 3174L      # ontology completeness additions

# ---- naming-instruction ensemble ----------------------------------------------
# The default instruction is sfa_naming_instruction(); the three rewordings
# below are the robustness ensemble (data/instruction_variants.txt lines 2-4).
INSTRUCTION_VARIANTS <- readLines(file.path("data", "instruction_variants.txt"))
stopifnot(length(INSTRUCTION_VARIANTS) == 4L)

# ---- paths --------------------------------------------------------------------
DIR_SCALES  <- file.path("data", "scales")
DIR_RESULTS <- "results"
DIR_FITS    <- file.path(DIR_RESULTS, "fits")
DIR_LABELS  <- file.path(DIR_RESULTS, "labels")
DIR_FIGS    <- file.path(DIR_RESULTS, "figures")
DIR_EMB     <- "embeddings"
DIR_GEN     <- file.path(DIR_RESULTS, "generated")   # LaTeX values + tables

for (d in c(DIR_RESULTS, DIR_FITS, DIR_LABELS, DIR_FIGS, DIR_EMB, DIR_GEN)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ---- helpers -------------------------------------------------------------------
read_scale <- function(scale) {
  df <- utils::read.csv(file.path(DIR_SCALES, paste0(scale, "_items.csv")),
                        stringsAsFactors = FALSE)
  stopifnot(all(c("code", "item", "factor", "scoring") %in% names(df)))
  df
}

k_doc_for <- function(scale) SCALES$k_doc[match(scale, SCALES$scale)]

# Write a data.frame of labels (sfa_labels) to JSON, flattening the
# candidate list-column so the file is portable.
labels_to_list <- function(lab) {
  lapply(seq_len(nrow(lab)), function(i) {
    list(factor          = lab$factor[i],
         label           = lab$label[i],
         rule            = lab$rule[i],
         n_items         = lab$n_items[i],
         collision_moved = lab$collision_moved[i],
         candidates      = as.character(lab$candidates[[i]]))
  })
}

save_json <- function(x, path) {
  jsonlite::write_json(x, path, auto_unbox = TRUE, pretty = TRUE, digits = 6)
}

message("Config loaded: ", nrow(SCALES), " scales, extraction = ",
        MODEL_EXTRACT, ", naming = ", MODEL_NAME,
        ", pool precision = ", POOL_PRECISION)
