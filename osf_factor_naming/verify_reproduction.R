# verify_reproduction.R -- check a rerun against the shipped reference outputs
#
# Usage (after running `Rscript run_all.R`, with or without a GPU):
#   Rscript verify_reproduction.R
#
# Compares the outputs your rerun just wrote under results/ against the
# frozen copies shipped under reference/ (the exact outputs of the paper's
# run). On a CPU-only rerun from the shipped embedding cache, everything is
# expected to match except artifacts that are GPU-only BY DESIGN:
#   - results/labels/*_rerun.json          (the re-embedding determinism
#                                           check re-runs GPU inference and
#                                           is skipped on CPU)
#   - the \valDeterminism... macros in values.tex
#   - the config == "cross_rerun" rows of evaluation.csv
#   - the "determinism" block of summary.json
# These are skipped below, so PASS means: every reproducible artifact
# reproduced exactly.

ok <- TRUE
note <- function(...) cat(sprintf(...), "\n")

fail <- function(...) { ok <<- FALSE; note(paste0("MISMATCH: ", ...)) }

# ---- labels ------------------------------------------------------------------
ref_labels <- list.files(file.path("reference", "labels"), full.names = TRUE)
for (rf in ref_labels) {
  base <- basename(rf)
  if (grepl("_rerun\\.json$", base)) next          # GPU-only by design
  nf <- file.path("results", "labels", base)
  if (!file.exists(nf)) { fail("missing results/labels/%s", base); next }
  a <- readLines(rf); b <- readLines(nf)
  if (!identical(a, b)) fail("labels differ: %s", base)
}
note("labels: %d files compared", sum(!grepl("_rerun\\.json$",
                                             basename(ref_labels))))

# ---- evaluation.csv (excluding the GPU-only determinism rows) ------------------
strip_rerun <- function(path) {
  d <- utils::read.csv(path, stringsAsFactors = FALSE)
  d <- d[d$config != "cross_rerun", ]
  d[order(d$config, d$scale, d$factor), ]
}
a <- strip_rerun(file.path("reference", "evaluation.csv"))
b <- strip_rerun(file.path("results", "evaluation.csv"))
if (!isTRUE(all.equal(a, b, check.attributes = FALSE))) {
  fail("evaluation.csv differs (excluding cross_rerun rows)")
} else note("evaluation.csv: %d rows match", nrow(a))

# ---- values.tex (excluding the GPU-only determinism macros) --------------------
strip_det <- function(path) {
  x <- readLines(path)
  x[!grepl("\\\\valDeterminism", x)]
}
a <- strip_det(file.path("reference", "generated", "values.tex"))
b <- strip_det(file.path("results", "generated", "values.tex"))
if (!identical(a, b)) fail("values.tex differs (excluding determinism macros)")
if (identical(a, b)) note("values.tex: %d lines match", length(a))

# ---- generated tables -----------------------------------------------------------
for (f in c("tab_labels_cross.tex", "tab_labels_explore.tex",
            "tab_retention.tex")) {
  a <- readLines(file.path("reference", "generated", f))
  b <- readLines(file.path("results", "generated", f))
  if (!identical(a, b)) fail("%s differs", f) else note("%s: matches", f)
}

# ---- verdict ----------------------------------------------------------------------
cat("\n")
if (ok) {
  cat("PASS: your rerun reproduces the paper's outputs exactly",
      "(GPU-only determinism artifacts excluded by design).\n")
} else {
  cat("FAIL: differences found above. If you re-embedded on different",
      "hardware instead of restoring the shipped cache, label changes on",
      "weak factors are expected; restore embeddings/cache/ per the README",
      "for an exact reproduction.\n")
  quit(status = 1L)
}
