# Word2vec_Bert_Factor_Analysis.R
# Requirements:
# install.packages(c("psych","tidyverse","text2vec","reticulate","tidytext"))
# Python: pip install sentence-transformers keybert
# Download GloVe: https://nlp.stanford.edu/projects/glove/ (e.g., glove.6B.100d.txt)

library(psych)
library(tidyverse)
library(text2vec)
library(reticulate)
library(tidytext)

# Configure Python virtual environment
use_virtualenv(".venv", required = TRUE)

# ------------------------
# User dataframes assumed: question_df_clean, scale_data_clean
# ------------------------
stopifnot(exists("question_df_clean"))
stopifnot(exists("scale_data_clean"))

items_mat <- as.matrix(scale_data_clean %>% select(starts_with("Q")))
colnames(items_mat) <- colnames(scale_data_clean %>% select(starts_with("Q")))

# ------------------------
# 1) EFA
# ------------------------
suggest_n_factors <- function(x) {
  pa <- fa.parallel(x, fm = "ml", fa = "fa", n.iter = 100, plot = FALSE)
  list(pa = pa, recommended = pa$nfact)
}
pa_res <- suggest_n_factors(items_mat)
n_factors <- pa_res$recommended
efa_res <- fa(items_mat, nfactors = n_factors, fm = "ml", rotate = "oblimin", scores = TRUE)
print(efa_res$loadings)

# ------------------------
# 2) Extract defining items
# ------------------------
get_defining_items_from_fa <- function(fa_obj, top_n = 5, loading_thresh = 0.35) {
  L <- as.matrix(fa_obj$loadings)
  nf <- ncol(L)
  res <- map(1:nf, function(j) {
    tibble(item = rownames(L), loading = L[, j]) %>%
      mutate(abs_loading = abs(loading)) %>%
      arrange(desc(abs_loading)) %>%
      filter(abs_loading >= loading_thresh) %>%
      slice_head(n = top_n)
  })
  names(res) <- paste0("Factor_", 1:nf)
  res
}
defining_items <- get_defining_items_from_fa(efa_res, top_n = 999, loading_thresh = 0.25)

# ------------------------
# 3) Word2Vec Labeling
# ------------------------
# Load pretrained GloVe vectors (adjust path if needed)
# glove <- read.delim("glove.6B.100d.txt", quote = "", comment.char = "", header = FALSE, sep = " ")
# word_vectors <- as.matrix(glove[, -1]); rownames(word_vectors) <- glove[, 1]

# Expanded stop words list
stop_words <- c("i", "a", "the", "of", "to", "and", "when", "for", "on", "it",
                "in", "my", "at", "by", "with", "is", "are", "am", "be", "have",
                "has", "had", "do", "does", "did", "that", "this", "these", "those",
                "will", "would", "could", "should", "can", "may", "might", "must",
                "me", "you", "he", "she", "we", "they", "them", "their", "his",
                "her", "its", "our", "your", "because", "about", "feel", "make",
                "makes", "even", "know", "realize", "think")

# ------------------------
# Helper functions
# ------------------------
create_weighted_text <- function(factor_items, loadings) {
  # Repeat items proportional to their loading weights
  # Higher loading items appear more frequently in the text
  weights <- abs(loadings) / sum(abs(loadings))
  n_repeats <- pmax(1, round(weights * 10))  # Scale to 1-10 repeats

  weighted_text <- map2_chr(factor_items$item_text, n_repeats, function(text, n) {
    paste(rep(text, n), collapse = " ")
  }) %>% paste(collapse = " ")

  weighted_text
}

clean_label <- function(label_text, max_words = 3) {
  # Standardize label formatting
  if (is.na(label_text) || nchar(label_text) == 0) return(NA)

  words <- unlist(strsplit(label_text, "\\s+"))
  words <- words[1:min(length(words), max_words)]
  paste(tools::toTitleCase(tolower(words)), collapse = " ")
}

get_word2vec_label <- function(factor_items, loadings, candidate_words = NULL) {
  # Extract and clean words from all items
  all_text <- paste(factor_items$item_text, collapse = " ")
  words <- unlist(strsplit(tolower(all_text), "\\s+"))

  # Remove stop words and short words
  words <- words[!words %in% stop_words & nchar(words) > 3]

  # Keep only words in vocabulary
  words <- intersect(words, rownames(word_vectors))
  if (length(words) == 0) return(NA)

  # Compute weighted centroid (weight by loadings)
  word_counts <- table(words)
  unique_words <- names(word_counts)
  item_weights <- abs(loadings) / sum(abs(loadings))

  # Create weighted centroid from item embeddings
  item_embeddings <- t(sapply(factor_items$item_text, function(txt) {
    txt_words <- unlist(strsplit(tolower(txt), "\\s+"))
    txt_words <- intersect(txt_words, rownames(word_vectors))
    if (length(txt_words) == 0) return(rep(0, ncol(word_vectors)))
    colMeans(word_vectors[txt_words, , drop = FALSE])
  }))
  weighted_centroid <- colSums(item_embeddings * item_weights)

  # Find best matching word
  search_space <- if (!is.null(candidate_words)) {
    intersect(candidate_words, rownames(word_vectors))
  } else {
    unique_words  # Search within content words from items
  }

  if (length(search_space) == 0) return(NA)
  sims <- word_vectors[search_space, , drop = FALSE] %*% weighted_centroid
  best_word <- search_space[which.max(sims)]
  tools::toTitleCase(best_word)
}

# ------------------------
# 4) BERT Labeling
# ------------------------
sentence_transformers <- import("sentence_transformers")
bert_model <- sentence_transformers$SentenceTransformer('all-MiniLM-L6-v2')

get_bert_label <- function(factor_items, loadings, candidate_labels = NULL) {
  embeddings <- bert_model$encode(factor_items$item_text)

  # Weighted centroid (weight by loadings)
  item_weights <- abs(loadings) / sum(abs(loadings))
  weighted_centroid <- colSums(embeddings * item_weights)

  # If candidate labels provided, use zero-shot classification
  if (!is.null(candidate_labels)) {
    candidate_embs <- bert_model$encode(candidate_labels)
    sims <- as.numeric(candidate_embs %*% weighted_centroid)
    return(candidate_labels[which.max(sims)])
  }

  # Otherwise, find most representative item and extract key words
  sims <- as.numeric(embeddings %*% weighted_centroid)
  best_item <- factor_items$item_text[which.max(sims)]

  # Extract content words
  words <- strsplit(best_item, "\\s+")[[1]]
  words <- words[!tolower(words) %in% stop_words]
  words <- words[nchar(words) > 3]

  if (length(words) == 0) return(best_item)

  # Return top 2-3 most meaningful words (prefer nouns/verbs over adjectives)
  # Simple heuristic: prioritize words that appear later (often more specific)
  if (length(words) <= 2) {
    return(paste(tools::toTitleCase(tolower(words)), collapse = " "))
  }

  # Take last 2 content words (often captures the core concept)
  key_words <- tail(words, 2)
  paste(tools::toTitleCase(tolower(key_words)), collapse = " ")
}

# ------------------------
# 5) TF-IDF Labeling
# ------------------------
get_tfidf_labels <- function(all_defining_items, question_df) {
  # Create corpus: one document per factor
  corpus <- map2_dfr(names(all_defining_items), all_defining_items, function(fname, items_df) {
    items_with_text <- items_df %>%
      left_join(question_df, by = c("item" = "question_id"))

    # Weight text by loadings
    weighted_text <- create_weighted_text(items_with_text, items_df$loading)

    tibble(
      factor = fname,
      text = weighted_text
    )
  })

  # Tokenize and compute TF-IDF
  tfidf_results <- corpus %>%
    unnest_tokens(word, text) %>%
    filter(!word %in% stop_words, nchar(word) > 3) %>%
    count(factor, word) %>%
    bind_tf_idf(word, factor, n) %>%
    group_by(factor) %>%
    arrange(desc(tf_idf)) %>%
    slice_head(n = 3) %>%
    summarise(
      tfidf_label = paste(tools::toTitleCase(word), collapse = " "),
      .groups = "drop"
    )

  tfidf_results
}

# ------------------------
# 6) Contrastive Word2Vec Labeling
# ------------------------
get_contrastive_word2vec_label <- function(factor_items, loadings, all_factor_centroids, current_factor_idx) {
  # Extract and clean words
  all_text <- paste(factor_items$item_text, collapse = " ")
  words <- unlist(strsplit(tolower(all_text), "\\s+"))
  words <- words[!words %in% stop_words & nchar(words) > 3]
  words <- intersect(words, rownames(word_vectors))

  if (length(words) == 0) return(NA)

  unique_words <- unique(words)
  current_centroid <- all_factor_centroids[[current_factor_idx]]

  # Compute contrastive scores
  other_centroids <- all_factor_centroids[-current_factor_idx]

  contrastive_scores <- sapply(unique_words, function(w) {
    word_vec <- word_vectors[w, ]
    sim_current <- sum(word_vec * current_centroid)

    # Average similarity to other factors
    sim_others <- mean(sapply(other_centroids, function(other_cent) {
      sum(word_vec * other_cent)
    }))

    sim_current - sim_others  # Contrastive score
  })

  # Get top words
  top_words <- unique_words[order(contrastive_scores, decreasing = TRUE)[1:min(3, length(unique_words))]]
  paste(tools::toTitleCase(top_words), collapse = " ")
}

# Helper: compute centroids for all factors
compute_all_centroids <- function(all_defining_items, question_df) {
  map(all_defining_items, function(items_df) {
    items_with_text <- items_df %>%
      left_join(question_df, by = c("item" = "question_id"))

    item_weights <- abs(items_df$loading) / sum(abs(items_df$loading))

    # Create weighted centroid from item embeddings
    item_embeddings <- t(sapply(items_with_text$item_text, function(txt) {
      txt_words <- unlist(strsplit(tolower(txt), "\\s+"))
      txt_words <- intersect(txt_words, rownames(word_vectors))
      if (length(txt_words) == 0) return(rep(0, ncol(word_vectors)))
      colMeans(word_vectors[txt_words, , drop = FALSE])
    }))

    colSums(item_embeddings * item_weights)
  })
}

# ------------------------
# 7) KeyBERT Labeling
# ------------------------
get_keybert_label <- function(factor_items, loadings) {
  tryCatch({
    keybert <- import("keybert")
    kw_model <- keybert$KeyBERT()

    # Create weighted text
    weighted_text <- create_weighted_text(factor_items, loadings)

    # Extract keywords
    keywords <- kw_model$extract_keywords(
      weighted_text,
      keyphrase_ngram_range = tuple(1L, 2L),
      top_n = 3L,
      stop_words = "english"
    )

    # Extract just the words (keywords is list of tuples)
    keyword_texts <- sapply(keywords, function(kw) kw[[1]])
    paste(tools::toTitleCase(keyword_texts), collapse = " ")
  }, error = function(e) {
    return(NA)  # KeyBERT not available
  })
}

# ------------------------
# 8) Run all methods
# ------------------------
cat("=== Running Labeling Methods Comparison ===\n\n")

# Pre-compute TF-IDF labels (needs all factors)
cat("Computing TF-IDF labels...\n")
tfidf_labels <- get_tfidf_labels(defining_items, question_df_clean)

# Pre-compute Word2Vec centroids for contrastive method
cat("Computing Word2Vec centroids...\n")
all_centroids <- compute_all_centroids(defining_items, question_df_clean)

# Run methods for each factor
results <- list()
for (i in seq_along(defining_items)) {
  factor_name <- names(defining_items)[i]
  df_items <- defining_items[[i]] %>% left_join(question_df_clean, by = c("item" = "question_id"))
  loadings <- defining_items[[i]]$loading

  cat("\n--- Running methods for", factor_name, "---\n")

  # Method 1: Original Word2Vec
  cat("  - Word2Vec (baseline)...\n")
  w2v_label <- get_word2vec_label(df_items, loadings)

  # Method 2: Contrastive Word2Vec
  cat("  - Contrastive Word2Vec...\n")
  contrastive_w2v_label <- get_contrastive_word2vec_label(df_items, loadings, all_centroids, i)

  # Method 3: BERT
  cat("  - BERT...\n")
  bert_label <- get_bert_label(df_items, loadings)

  # Method 4: TF-IDF (already computed)
  tfidf_label <- tfidf_labels$tfidf_label[tfidf_labels$factor == factor_name]

  # Method 5: KeyBERT
  cat("  - KeyBERT...\n")
  keybert_label <- get_keybert_label(df_items, loadings)

  results[[factor_name]] <- tibble(
    factor = factor_name,
    word2vec = clean_label(w2v_label),
    contrastive_w2v = clean_label(contrastive_w2v_label),
    bert = clean_label(bert_label),
    tfidf = clean_label(tfidf_label),
    keybert = clean_label(keybert_label)
  )
}

# Combine results
final_labels <- bind_rows(results)

cat("\n\n=== LABELING COMPARISON RESULTS ===\n")
print(final_labels, width = Inf)

# Summary statistics
cat("\n=== Method Availability ===\n")
cat("Word2Vec available:", !all(is.na(final_labels$word2vec)), "\n")
cat("Contrastive W2V available:", !all(is.na(final_labels$contrastive_w2v)), "\n")
cat("BERT available:", !all(is.na(final_labels$bert)), "\n")
cat("TF-IDF available:", !all(is.na(final_labels$tfidf)), "\n")
cat("KeyBERT available:", !all(is.na(final_labels$keybert)), "\n")