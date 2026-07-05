# 03_evaluate.R -- benchmark evaluation of the produced labels (CPU only)
#
# Evaluates every label column produced by 01_fit.R / 02_name.R against the
# frozen ground-truth alias table (data/gt_aliases.csv, committed before the
# analyses were run; see README for the derivation rules R1-R5).
#
# For every factor we compute:
#   dominant_construct  the documented construct that dominates the items
#                       the factor's naming target was built from (primary
#                       loading assignment, dominant-pole side), from the
#                       instrument's own item keying (data/scales/*.csv)
#   purity              the share of those items keyed to that construct
#   top1_hit            does the automatic label match any alias of the
#                       dominant construct? (word-family match, see below)
#   set_hit             does any member of the leave-one-out candidate set
#                       match?
#
# Matching is deterministic: two terms match if their strings are equal or
# their WordNet word families (shipped in the pool's wordlist) are equal.
# No human judgment enters at evaluation time.

source(file.path("R", "00_config.R"))

# ---- word-family lookup (from the shipped pool wordlist) ---------------------
pool_dir <- file.path(tools::R_user_dir("semanticfa", "cache"), "pools")
wordlist <- readRDS(file.path(pool_dir, "wordlist.rds"))
FAMILY <- stats::setNames(wordlist$family, wordlist$word)

family_of <- function(w) {
  w <- tolower(trimws(w))
  f <- unname(FAMILY[w])
  ifelse(is.na(f), w, f)
}

term_match <- function(a, b) {
  a <- tolower(trimws(a)); b <- tolower(trimws(b))
  a == b | family_of(a) == family_of(b) | a == family_of(b) | family_of(a) == b
}

aliases <- utils::read.csv(file.path("data", "gt_aliases.csv"),
                           stringsAsFactors = FALSE)

# ---- alias reachability audit -------------------------------------------------
# A label can only ever be a pool word, so a construct whose aliases are all
# absent from the pool (as word or word family) can never be hit: its ceiling
# is structural, a property of the candidate census rather than the naming
# machinery. This audit records that ceiling; it is reported in the paper,
# and the frozen alias table is NOT amended in response (that would be
# post-hoc fitting).
FAMSET <- unique(wordlist$family)
alias_reachable <- vapply(seq_len(nrow(aliases)), function(i) {
  a <- tolower(trimws(aliases$alias[i]))
  a %in% names(FAMILY) || a %in% FAMSET || family_of(a) %in% FAMSET
}, logical(1))
reach <- aggregate(alias_reachable,
                   by = list(scale = aliases$scale, construct = aliases$construct),
                   FUN = any)
names(reach)[3] <- "reachable"
unreachable <- reach[!reach$reachable, c("scale", "construct")]
if (nrow(unreachable)) {
  message("Constructs with NO pool-reachable alias (structural ceiling): ",
          paste(unreachable$scale, unreachable$construct, collapse = "; "))
}

alias_hit <- function(label, scale, construct) {
  al <- aliases$alias[aliases$scale == scale & aliases$construct == construct]
  if (length(al) == 0L || is.na(label)) return(NA)
  any(vapply(al, function(a) term_match(label, a), logical(1)))
}

# Which construct(s) do a label's aliases identify within a scale?
# Returns the full hit set (a label may match aliases of more than one
# construct, e.g. a shared superordinate term); used for the
# instruction-ensemble "construct flip" check.
constructs_of_label <- function(label, scale) {
  cons <- unique(aliases$construct[aliases$scale == scale])
  cons[vapply(cons, function(cc) isTRUE(alias_hit(label, scale, cc)),
              logical(1))]
}

# ---- dominant construct + purity ----------------------------------------------
# Reproduces the item-selection rule of the naming target (primary loading
# assignment, dominant-pole side) so composition is reported for exactly the
# items the label was computed from.
naming_items <- function(fit) {
  L <- unclass(fit$loadings)
  primary <- apply(abs(L), 1L, which.max)
  lapply(seq_len(ncol(L)), function(j) {
    idx <- which(primary == j)
    if (length(idx) == 0L) idx <- seq_len(nrow(L))
    dom <- sign(L[idx[which.max(abs(L[idx, j]))], j])
    if (dom == 0) dom <- 1
    sel <- idx[sign(L[idx, j]) == dom]
    if (length(sel) == 0L) sel <- idx
    sel
  })
}

dominant_construct <- function(fit, df) {
  sel <- naming_items(fit)
  out <- lapply(sel, function(ix) {
    tab <- sort(table(df$factor[ix]), decreasing = TRUE)
    list(construct = names(tab)[1L],
         purity    = as.numeric(tab[1L]) / sum(tab),
         n_items   = length(ix))
  })
  out
}

# ---- evaluate one label file ---------------------------------------------------
CONFIGS <- c(
  cross_fp16   = "confirmatory_harrier_fp16",
  cross_int8   = "confirmatory_harrier_int8",
  cross_rerun  = "confirmatory_harrier_fp16_rerun",
  cross_nokeeper = "confirmatory_harrier_fp16_nokeeper",
  cross_inst2  = "confirmatory_harrier_fp16_instr2",
  cross_inst3  = "confirmatory_harrier_fp16_instr3",
  cross_inst4  = "confirmatory_harrier_fp16_instr4",
  qwen_fp16    = "confirmatory_qwen_fp16",
  qwen_int8    = "confirmatory_qwen_int8",
  explore_cross = "exploratory_harrier_fp16",
  explore_qwen  = "exploratory_qwen_fp16"
)

rows <- list()
for (s in SCALES$scale) {
  df <- read_scale(s)
  fit_c <- readRDS(file.path(DIR_FITS, paste0(s, "_confirmatory.rds")))
  fit_e <- readRDS(file.path(DIR_FITS, paste0(s, "_exploratory.rds")))
  dom_c <- dominant_construct(fit_c, df)
  dom_e <- dominant_construct(fit_e, df)

  for (cfg in names(CONFIGS)) {
    path <- file.path(DIR_LABELS, paste0(s, "_", CONFIGS[[cfg]], ".json"))
    if (!file.exists(path)) next
    labs <- jsonlite::read_json(path, simplifyVector = FALSE)
    dom  <- if (startsWith(CONFIGS[[cfg]], "exploratory")) dom_e else dom_c

    for (i in seq_along(labs)) {
      li <- labs[[i]]
      cand <- unlist(li$candidates)
      d <- dom[[i]]
      rows[[length(rows) + 1L]] <- data.frame(
        config    = cfg,
        scale     = s,
        factor    = li$factor,
        construct = d$construct,
        purity    = round(d$purity, 3),
        n_items   = d$n_items,
        label     = li$label,
        rule      = li$rule,
        collision_moved = isTRUE(li$collision_moved),
        loo_set_size    = length(cand),
        candidates      = paste(cand, collapse = "; "),
        top1_hit  = isTRUE(alias_hit(li$label, s, d$construct)),
        set_hit   = any(vapply(cand, function(w)
                       isTRUE(alias_hit(w, s, d$construct)), logical(1))),
        stringsAsFactors = FALSE
      )
    }
  }
}
eval_df <- do.call(rbind, rows)
utils::write.csv(eval_df, file.path(DIR_RESULTS, "evaluation.csv"),
                 row.names = FALSE)

# ---- summaries -------------------------------------------------------------------
pct <- function(x) round(100 * mean(x), 1)

summ <- lapply(split(eval_df, eval_df$config), function(d) {
  list(n_factors  = nrow(d),
       top1_count = sum(d$top1_hit),
       top1_pct   = pct(d$top1_hit),
       set_count  = sum(d$set_hit),
       set_pct    = pct(d$set_hit),
       loo_median = stats::median(d$loo_set_size),
       loo_singleton = sum(d$loo_set_size == 1L),
       purity_median = stats::median(d$purity))
})

# Compare two label columns. exact = TRUE compares label strings verbatim
# (the determinism criterion); exact = FALSE counts word-family changes
# (the precision-robustness criterion, where a same-family variant is not
# a construct-level change).
cmp_labels <- function(a, b, exact = FALSE) {
  A <- eval_df[eval_df$config == a, ]; B <- eval_df[eval_df$config == b, ]
  key <- paste(A$scale, A$factor); B <- B[match(key, paste(B$scale, B$factor)), ]
  changed <- if (exact) A$label != B$label else !term_match(A$label, B$label)
  list(n = nrow(A), changed = sum(changed),
       which = paste(A$scale, A$factor, ":", B$label, "->", A$label)[changed])
}
summ$int8_vs_fp16_cross <- cmp_labels("cross_int8", "cross_fp16")
summ$int8_vs_fp16_qwen  <- cmp_labels("qwen_int8",  "qwen_fp16")

# determinism: fresh re-embedding reproduces the labels verbatim?
if (any(eval_df$config == "cross_rerun")) {
  det <- cmp_labels("cross_rerun", "cross_fp16", exact = TRUE)
  summ$determinism <- list(n = det$n, identical = det$n - det$changed,
                           which_changed = det$which)
}

# instruction ensemble: unanimity / majority / construct flips across the
# default instruction (cross_fp16) and the three rewordings
ens_cfgs <- c("cross_fp16", "cross_inst2", "cross_inst3", "cross_inst4")
E <- eval_df[eval_df$config %in% ens_cfgs, ]
E$key <- paste(E$scale, E$factor)
n_per_key <- vapply(split(E, E$key), nrow, integer(1))
stopifnot(all(n_per_key == length(ens_cfgs)))   # every variant present
ens <- lapply(split(E, E$key), function(d) {
  fams <- family_of(d$label)
  # A construct FLIP means two instruction wordings produced labels that
  # identify disjoint, non-empty construct sets: the wording changed which
  # documented construct the label points at (not merely which synonym).
  sets <- lapply(seq_len(nrow(d)),
                 function(i) constructs_of_label(d$label[i], d$scale[i]))
  sets <- Filter(length, sets)
  flip <- length(sets) > 1L &&
          length(Reduce(intersect, sets)) == 0L
  list(unanimous = length(unique(fams)) == 1L,
       majority  = max(table(fams)) >= 3L,
       flip      = flip)
})
summ$ensemble <- list(
  n         = length(ens),
  unanimous = sum(vapply(ens, `[[`, logical(1), "unanimous")),
  majority  = sum(vapply(ens, `[[`, logical(1), "majority")),
  flips     = sum(vapply(ens, `[[`, logical(1), "flip"))
)

# alias reachability (structural ceiling of the census; see audit above)
summ$alias_reachability <- list(
  n_constructs = nrow(reach),
  unreachable  = if (nrow(unreachable))
                   paste(unreachable$scale, unreachable$construct)
                 else character(0)
)

# headline accuracy restricted to factors whose dominant construct has at
# least one pool-reachable alias (the non-structural part of the benchmark)
reach_key <- paste(reach$scale, reach$construct)[reach$reachable]
hl_df <- eval_df[eval_df$config == "cross_fp16", ]
hl_reach <- hl_df[paste(hl_df$scale, hl_df$construct) %in% reach_key, ]
summ$cross_fp16_reachable <- list(
  n_factors  = nrow(hl_reach),
  top1_count = sum(hl_reach$top1_hit),
  top1_pct   = pct(hl_reach$top1_hit),
  set_count  = sum(hl_reach$set_hit),
  set_pct    = pct(hl_reach$set_hit))

# exploratory retention counts (for the confirmatory-vs-exploratory story)
summ$k_explore <- lapply(stats::setNames(SCALES$scale, SCALES$scale), function(s) {
  fit_e <- readRDS(file.path(DIR_FITS, paste0(s, "_exploratory.rds")))
  list(k_doc = k_doc_for(s), k_explore = fit_e$factors)
})

save_json(summ, file.path(DIR_RESULTS, "summary.json"))
message("03_evaluate.R complete: ", nrow(eval_df), " factor x config rows.")
