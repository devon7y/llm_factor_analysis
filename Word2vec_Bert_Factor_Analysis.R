# Strategy1_pipeline_clean.R
# Requirements:
# install.packages(c("psych","tidyverse","httr","jsonlite","glue","text2vec","reticulate"))

library(psych)
library(tidyverse)
library(httr)
library(jsonlite)
library(glue)
library(text2vec)
library(reticulate)

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
# 3) Prompt builder for LLMs
# ------------------------
build_prompt_for_factor <- function(factor_id, top_items_df, fa_obj, question_texts, factor_index) {
  items_text <- top_items_df %>%
    left_join(question_texts, by = c("item" = "question_id")) %>%
    mutate(line = paste0("* ", item, " (loading: ", round(abs(loading), 2), "): ", item_text)) %>%
    pull(line) %>% paste(collapse = "\n")
  avg_loading <- mean(abs(top_items_df$loading))
  prop_var <- round(100 * (fa_obj$Vaccounted["Proportion Var", factor_index]), 1)
  glue(
    "These items load on {factor_id}.\n",
    "Avg abs loading: {round(avg_loading,3)}\n",
    "Variance explained: {prop_var}%\n\n",
    "Items:\n{items_text}\n\n",
    "Provide a concise 1–3 word noun phrase. Return ONLY the phrase.\nLabel:"
  )
}
question_texts <- question_df_clean
prompts_list <- map2(names(defining_items), seq_along(defining_items), function(fnm, idx) {
  df <- defining_items[[fnm]] %>% select(item, loading)
  build_prompt_for_factor(fnm, df, efa_res, question_texts, idx)
})
names(prompts_list) <- names(defining_items)

# ------------------------
# 4) GPT-4 turbo (chat API)
# ------------------------
call_openai_chat <- function(prompt, model = "gpt-4-turbo", max_completion_tokens = 20, temperature = 0) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  body <- list(
    model = model,
    messages = list(list(role = "system", content = "You name psychological factors."),
                    list(role = "user", content = prompt)),
    max_completion_tokens = max_completion_tokens,
    temperature = temperature
  )
  res <- POST("https://api.openai.com/v1/chat/completions",
              add_headers(Authorization = paste("Bearer", api_key),
                          `Content-Type` = "application/json"),
              body = toJSON(body, auto_unbox = TRUE))
  fromJSON(content(res, "text", encoding = "UTF-8"), simplifyVector = TRUE)
}
extract_label_from_chat <- function(chat_out) chat_out$choices$message$content[1]

# ------------------------
# 5) Word2Vec Labeling
# ------------------------
# Load pretrained GloVe vectors (adjust path if needed)
# glove <- read.delim("glove.6B.100d.txt", quote = "", comment.char = "", header = FALSE, sep = " ")
# word_vectors <- as.matrix(glove[, -1]); rownames(word_vectors) <- glove[, 1]

get_word2vec_label <- function(factor_items) {
  words <- unlist(strsplit(tolower(paste(factor_items$item_text, collapse = " ")), "\\s+"))
  words <- intersect(words, rownames(word_vectors))
  if (length(words) == 0) return(NA)
  centroid <- colMeans(word_vectors[words, , drop = FALSE])
  sims <- word_vectors %*% centroid
  best_word <- rownames(word_vectors)[which.max(sims)]
  tools::toTitleCase(best_word)
}

# ------------------------
# 6) BERT Labeling
# ------------------------
sentence_transformers <- import("sentence_transformers")
bert_model <- sentence_transformers$SentenceTransformer('all-MiniLM-L6-v2')

get_bert_label <- function(factor_items) {
  embeddings <- bert_model$encode(factor_items$item_text)
  centroid <- colMeans(embeddings)
  sims <- as.numeric(embeddings %*% centroid)
  best_item <- factor_items$item_text[which.max(sims)]
  words <- strsplit(best_item, "\\s+")[[1]]
  words <- words[!tolower(words) %in% c("i","a","the","of","to","and","when","for","on","it","in","my","at","by","with")]
  words <- words[nchar(words) > 3]
  if (length(words) == 0) return(best_item)
  best_word <- words[which.max(nchar(words))]
  tools::toTitleCase(best_word)
}

# ------------------------
# 7) Run all methods
# ------------------------
results <- list()
for (i in seq_along(prompts_list)) {
  factor_name <- names(prompts_list)[i]
  df_items <- defining_items[[i]] %>% left_join(question_df_clean, by = c("item" = "question_id"))
  prompt <- prompts_list[[i]]
  
  cat("--- Running for", factor_name, "---\n")
  
  # GPT-4 turbo
  out_chat <- call_openai_chat(prompt)
  gpt4_label <- extract_label_from_chat(out_chat)
  
  # Word2Vec
  w2v_label <- get_word2vec_label(df_items)
  
  # BERT
  bert_label <- get_bert_label(df_items)
  
  results[[factor_name]] <- tibble(
    factor = factor_name,
    gpt4_label = gpt4_label,
    word2vec_label = w2v_label,
    bert_label = bert_label
  )
}
final_labels <- bind_rows(results)
print(final_labels)