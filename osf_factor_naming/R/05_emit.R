# 05_emit.R -- emit the paper's numbers, tables, and figures (CPU only)
#
# Every data-derived value reported in the manuscript is emitted here as a
# LaTeX macro (results/generated/values.tex); the manuscript never contains
# a hand-typed number. Re-running the pipeline regenerates the macros and
# figures, and recompiling the manuscript refreshes every number at once.
#
# Outputs
#   results/generated/values.tex         one \newcommand per reported value
#   results/generated/tab_labels_cross.tex   booktabs body: headline labels
#   results/generated/tab_labels_explore.tex booktabs body: exploratory labels
#   results/generated/tab_retention.tex      booktabs body: retention battery
#   results/figures/*.pdf                the paper's figures
#   embeddings/cache/                    the semanticfa embedding cache
#                                        (restores bit-identical reruns)

source(file.path("R", "00_config.R"))

eval_df <- utils::read.csv(file.path(DIR_RESULTS, "evaluation.csv"),
                           stringsAsFactors = FALSE)
summ    <- jsonlite::read_json(file.path(DIR_RESULTS, "summary.json"))
retn    <- jsonlite::read_json(file.path(DIR_RESULTS, "retention_battery.json"))

pool_dir <- file.path(tools::R_user_dir("semanticfa", "cache"), "pools")
wordlist <- readRDS(file.path(pool_dir, "wordlist.rds"))

# ---- values.tex -------------------------------------------------------------------
vals <- character(0)
val <- function(name, value, src) {
  stopifnot(grepl("^[A-Za-z]+$", name))
  vals <<- c(vals, sprintf("\\newcommand{\\val%s}{%s}   %% %s", name, value, src))
}
fmt_pct <- function(x) formatC(x, format = "f", digits = 1)

n_items_total <- sum(vapply(SCALES$scale,
                     function(s) nrow(read_scale(s)), integer(1)))
val("NScales", nrow(SCALES), "00_config.R: SCALES")
val("NItems", n_items_total, "data/scales/*.csv row counts")
val("NFactorsConfirm", sum(SCALES$k_doc), "00_config.R: sum of documented k")

cfg <- function(name) summ[[name]]
hl <- cfg("cross_fp16")
val("NFactorsExplore",
    sum(vapply(summ$k_explore, function(x) x$k_explore, numeric(1))),
    "summary.json: k_explore totals")

val("PoolCensusBase", format(POOL_CENSUS_BASE, big.mark = ","),
    "pool build manifest: census size")
val("PoolAdditions", format(POOL_ADDITIONS, big.mark = ","),
    "pool build manifest: ontology additions")
val("PoolEligible", format(nrow(wordlist), big.mark = ","),
    "wordlist.rds: eligible candidates")
val("PoolEligiblePct",
    fmt_pct(100 * nrow(wordlist) / (POOL_CENSUS_BASE + POOL_ADDITIONS)),
    "eligible / embedded census")

for (pair in list(c("Cross", "cross_fp16"),  c("CrossIntEight", "cross_int8"),
                  c("Qwen", "qwen_fp16"),    c("ExploreCross", "explore_cross"),
                  c("ExploreQwen", "explore_qwen"))) {
  p <- cfg(pair[2]); if (is.null(p)) next
  val(paste0(pair[1], "TopOneCount"), p$top1_count,
      paste0("summary.json: ", pair[2]))
  val(paste0(pair[1], "TopOnePct"), fmt_pct(p$top1_pct),
      paste0("summary.json: ", pair[2]))
  val(paste0(pair[1], "SetCount"), p$set_count,
      paste0("summary.json: ", pair[2]))
  val(paste0(pair[1], "SetPct"), fmt_pct(p$set_pct),
      paste0("summary.json: ", pair[2]))
  val(paste0(pair[1], "NFactors"), p$n_factors,
      paste0("summary.json: ", pair[2]))
}

val("LooMedian", hl$loo_median, "summary.json: cross_fp16 LOO sets")
val("LooSingletonCount", hl$loo_singleton, "summary.json: cross_fp16")
val("LooSingletonPct", fmt_pct(100 * hl$loo_singleton / hl$n_factors),
    "summary.json: cross_fp16")
val("PurityMedian", formatC(hl$purity_median, format = "f", digits = 2),
    "summary.json: cross_fp16")

val("IntEightChangedCross", summ$int8_vs_fp16_cross$changed,
    "summary.json: int8 vs fp16, naming model")
val("IntEightChangedQwen", summ$int8_vs_fp16_qwen$changed,
    "summary.json: int8 vs fp16, extraction model")
if (!is.null(summ$determinism)) {
  val("DeterminismIdentical", summ$determinism$identical,
      "summary.json: re-embedding determinism")
  val("DeterminismN", summ$determinism$n,
      "summary.json: re-embedding determinism")
}
val("UnreachableConstructs", length(summ$alias_reachability$unreachable),
    "summary.json: constructs with no pool-reachable alias")
val("ReachableTopOneCount", summ$cross_fp16_reachable$top1_count,
    "summary.json: cross_fp16, reachable constructs only")
val("ReachableTopOnePct", fmt_pct(summ$cross_fp16_reachable$top1_pct),
    "summary.json: cross_fp16, reachable constructs only")
val("ReachableSetCount", summ$cross_fp16_reachable$set_count,
    "summary.json: cross_fp16, reachable constructs only")
val("ReachableSetPct", fmt_pct(summ$cross_fp16_reachable$set_pct),
    "summary.json: cross_fp16, reachable constructs only")
val("ReachableNFactors", summ$cross_fp16_reachable$n_factors,
    "summary.json: cross_fp16, reachable constructs only")
val("EnsembleUnanimous", summ$ensemble$unanimous, "summary.json: ensemble")
val("EnsembleMajority", summ$ensemble$majority, "summary.json: ensemble")
val("EnsembleFlips", summ$ensemble$flips, "summary.json: ensemble")
val("EnsembleN", summ$ensemble$n, "summary.json: ensemble")

k_doc_v <- vapply(summ$k_explore, function(x) x$k_doc, numeric(1))
k_exp_v <- vapply(summ$k_explore, function(x) x$k_explore, numeric(1))
val("RetentionAgreeParallel", sum(k_doc_v == k_exp_v),
    "summary.json: scales where parallel analysis k equals documented k")

header <- c(
  "% values.tex -- AUTOGENERATED by osf_factor_naming/R/05_emit.R.",
  "% Do not edit by hand: rerun the analysis pipeline to regenerate.",
  sprintf("%% semanticfa %s | extraction %s | naming %s | pool %s",
          as.character(utils::packageVersion("semanticfa")),
          MODEL_EXTRACT, MODEL_NAME, POOL_PRECISION))
writeLines(c(header, vals), file.path(DIR_GEN, "values.tex"))

# ---- LaTeX tables ------------------------------------------------------------------
tex_escape <- function(x) {
  x <- gsub("\\\\", "\001", x)              # placeholder first, so the brace
  x <- gsub("([&%$#_{}])", "\\\\\\1", x)    # escaping cannot re-escape it
  gsub("\001", "\\\\textbackslash{}", x)
}

label_table <- function(d, path) {
  d <- d[order(match(d$scale, SCALES$scale),
               as.integer(sub("^\\D+", "", d$factor))), ]
  disp_factor <- sub("^MR", "F", d$factor)   # psych names factors MR1, MR2, ...
  lines <- sprintf("%s & %s & %s (%.2f) & %s & %s \\\\",
                   tex_escape(d$scale), tex_escape(disp_factor),
                   tex_escape(d$construct), d$purity,
                   tex_escape(d$label), tex_escape(d$candidates))
  writeLines(lines, path)
}
label_table(eval_df[eval_df$config == "cross_fp16", ],
            file.path(DIR_GEN, "tab_labels_cross.tex"))
label_table(eval_df[eval_df$config == "explore_cross", ],
            file.path(DIR_GEN, "tab_labels_explore.tex"))

retn_lines <- vapply(SCALES$scale, function(s) {
  r <- retn[[s]]
  cols <- c("parallel", "kaiser", "TEFI", "EKC", "MAP", "EGA")
  ks <- vapply(cols, function(m)
    if (!is.null(r[[m]])) as.character(r[[m]]) else "--", character(1))
  sprintf("%s & %s & %s \\\\", tex_escape(s), as.character(r$k_doc),
          paste(ks, collapse = " & "))
}, character(1))
writeLines(retn_lines, file.path(DIR_GEN, "tab_retention.tex"))

# ---- figures ------------------------------------------------------------------------
fig <- function(name, width, height, expr) {
  grDevices::pdf(file.path(DIR_FIGS, name), width = width, height = height,
                 useDingbats = FALSE)
  on.exit(grDevices::dev.off(), add = TRUE)
  force(expr)
}

# Figure: benchmark accuracy by configuration
fig("fig_accuracy.pdf", 7, 4.2, {
  cfgs <- c("cross_fp16", "cross_int8", "qwen_fp16", "explore_cross")
  names_disp <- c("Two-encoder\n(headline)", "Two-encoder\nint8 pool",
                  "Single-encoder\n(extraction model)", "Two-encoder\nexploratory k")
  top1 <- vapply(cfgs, function(c) summ[[c]]$top1_pct, numeric(1))
  sethit <- vapply(cfgs, function(c) summ[[c]]$set_pct, numeric(1))
  par(mar = c(6, 4.5, 1, 1))
  bp <- barplot(rbind(top1, sethit), beside = TRUE, ylim = c(0, 100),
                names.arg = names_disp, cex.names = 0.8,
                col = c("grey25", "grey70"), border = NA,
                ylab = "Factors with a construct-matching label (%)")
  legend("topright", bty = "n", fill = c("grey25", "grey70"), border = NA,
         legend = c("Automatic label", "Leave-one-out candidate set"))
  vals_all <- rbind(top1, sethit)
  text(bp, vals_all + 3, sprintf("%.0f", vals_all), cex = 0.75)
})

# Figure: leave-one-out candidate-set sizes (the label's error bar)
fig("fig_loo_sizes.pdf", 5.5, 4, {
  d <- eval_df[eval_df$config == "cross_fp16", ]
  tab <- table(factor(d$loo_set_size, levels = 1:max(d$loo_set_size)))
  par(mar = c(4.5, 4.5, 1, 1))
  barplot(tab, col = "grey40", border = NA,
          xlab = "Leave-one-out candidate-set size",
          ylab = "Number of factors")
})

# Figure: composition purity vs label correctness
fig("fig_purity.pdf", 5.5, 4.2, {
  d <- eval_df[eval_df$config == "cross_fp16", ]
  set.seed(42)
  par(mar = c(4.5, 4.5, 1, 1))
  x <- jitter(as.numeric(d$top1_hit), amount = 0.08)
  plot(x, d$purity, pch = ifelse(d$top1_hit, 19, 1), cex = 1.1,
       xaxt = "n", xlim = c(-0.4, 1.4), ylim = c(0, 1.02),
       xlab = "", ylab = "Composition purity of the factor")
  axis(1, at = c(0, 1), labels = c("Label misses", "Label matches"))
})

# Figure: documented vs exploratory factor counts
fig("fig_retention.pdf", 6.5, 4.2, {
  par(mar = c(4.5, 6, 1, 1))
  n <- nrow(SCALES)
  plot(NULL, xlim = range(c(k_doc_v, k_exp_v)) + c(-0.5, 0.5),
       ylim = c(0.5, n + 0.5), yaxt = "n",
       xlab = "Number of factors", ylab = "")
  axis(2, at = n:1, labels = SCALES$scale, las = 1, cex.axis = 0.85)
  segments(k_doc_v, n:1, k_exp_v, n:1, col = "grey60")
  points(k_doc_v, n:1, pch = 19)
  points(k_exp_v, n:1, pch = 1)
  legend("bottomright", bty = "n", pch = c(19, 1),
         legend = c("Documented (confirmatory)", "Parallel analysis (exploratory)"))
})

# ---- ship the embedding cache (bit-identical replication without a GPU) -------------
cache_dir <- tools::R_user_dir("semanticfa", "cache")
dest <- file.path(DIR_EMB, "cache")
if (!dir.exists(dest)) dir.create(dest, recursive = TRUE)
for (f in list.files(cache_dir, pattern = "\\.rds$", full.names = TRUE)) {
  file.copy(f, dest, overwrite = TRUE)
}

# ---- session info --------------------------------------------------------------------
writeLines(utils::capture.output(utils::sessionInfo()),
           file.path(DIR_RESULTS, "session_info.txt"))

message("05_emit.R complete: ", length(vals), " macros, 4 figures, 3 tables.")
