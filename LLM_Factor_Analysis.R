# Strategy1_pipeline.R
# Requirements:
# install.packages(c("psych","tidyverse","httr","jsonlite","glue"))

library(psych)
library(tidyverse)
library(httr)
library(jsonlite)
library(glue)

# ------------------------
# User dataframes (assumed present in environment)
# question_df_clean: data.frame with columns question_id, item_text
# scale_data_clean: tibble/data.frame, columns Q1..Qk (numeric responses)
# ------------------------

# Quick checks
stopifnot(exists("question_df_clean"))
stopifnot(exists("scale_data_clean"))

# Convert to numeric matrix for factor analysis
items_mat <- as.matrix(scale_data_clean %>% select(starts_with("Q")))
colnames(items_mat) <- colnames(scale_data_clean %>% select(starts_with("Q")))

# ------------------------
# 1) Determine number of factors via parallel analysis
# ------------------------
suggest_n_factors <- function(x) {
  pa <- fa.parallel(x, fm = "ml", fa = "fa", n.iter = 100, plot = FALSE)
  recommended <- pa$nfact
  list(pa = pa, recommended = recommended)
}

pa_res <- suggest_n_factors(items_mat)
n_factors <- pa_res$recommended
message("Parallel analysis suggests ", n_factors, " factors.")

# ------------------------
# 2) Run EFA
# ------------------------
run_efa <- function(x, nfactors) {
  fa_res <- fa(x, nfactors = nfactors, fm = "ml", rotate = "oblimin", scores = TRUE)
  return(fa_res)
}

efa_res <- run_efa(items_mat, n_factors)
print(efa_res$loadings)

# ------------------------
# Helper: extract top items per factor
# ------------------------
get_defining_items_from_fa <- function(fa_obj, top_n = 5, loading_thresh = 0.35) {
  L <- as.matrix(fa_obj$loadings)
  nf <- ncol(L)
  res <- map(1:nf, function(j) {
    loadings_j <- L[, j]
    df <- tibble(item = rownames(L), loading = loadings_j) %>%
      mutate(abs_loading = abs(loading)) %>%
      arrange(desc(abs_loading))
    df_top <- df %>% filter(abs_loading >= loading_thresh) %>% slice_head(n = top_n)
    df_top
  })
  names(res) <- paste0("Factor_", 1:length(res))
  res
}

defining_items <- get_defining_items_from_fa(efa_res, top_n = 999, loading_thresh = 0.25)

# ------------------------
# 3) Build prompt template
# ------------------------
build_prompt_for_factor <- function(factor_id, top_items_df, fa_obj, question_texts, factor_index) {
  items_text <- top_items_df %>%
    left_join(question_texts, by = c("item" = "question_id")) %>%
    mutate(line = paste0("* ", item, " (loading: ", round(abs(loading), 2), "): ", item_text)) %>%
    pull(line) %>%
    paste(collapse = "\n")
  
  avg_loading <- mean(abs(top_items_df$loading))
  prop_var <- round(100 * (fa_obj$Vaccounted["Proportion Var", factor_index]), 1)
  
  prompt <- glue(
    "These questionnaire items load most strongly on {factor_id}.\n",
    "Average absolute loading (top items): {round(avg_loading,3)}\n",
    "Percent variance explained by this factor: {prop_var}%\n\n",
    "Items:\n{items_text}\n\n",
    "Provide a concise, 1–3 word noun phrase that best names the construct.\n",
    "Return ONLY the phrase (no explanation).\nLabel:"
  )
  as.character(prompt)
}

question_texts <- question_df_clean
prompts_list <- map2(
  .x = names(defining_items),
  .y = seq_along(defining_items),
  function(fnm, idx) {
    df <- defining_items[[fnm]] %>% select(item, loading)
    build_prompt_for_factor(factor_id = fnm, top_items_df = df,
                            fa_obj = efa_res, question_texts = question_texts,
                            factor_index = idx)
  }
)
names(prompts_list) <- names(defining_items)

# ------------------------
# 4) LLM call helper – GPT-4 turbo (chat API)
# ------------------------
call_openai_chat <- function(prompt, max_completion_tokens = 20, model = "gpt-4-turbo", temperature = 0) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (nchar(api_key) == 0) stop("Please set OPENAI_API_KEY environment variable.")
  
  body <- list(
    model = model,
    messages = list(
      list(role = "system", content = "You are a helpful assistant."),
      list(role = "user", content = prompt)
    ),
    max_completion_tokens = max_completion_tokens,
    temperature = temperature
  )
  
  res <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    httr::add_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ),
    body = jsonlite::toJSON(body, auto_unbox = TRUE, null = "null")
  )
  
  txt <- httr::content(res, as = "text", encoding = "UTF-8")
  if (httr::http_error(res)) {
    stop(paste0("HTTP ", httr::status_code(res), ": ", txt))
  }
  
  jsonlite::fromJSON(txt, simplifyVector = TRUE)
}

extract_label_from_chat <- function(chat_out) {
  if ("choices" %in% names(chat_out)) {
    # choices is a data.frame with column "message"
    if ("message" %in% names(chat_out$choices)) {
      return(trimws(chat_out$choices$message$content[1]))
    }
  }
  # fallback: dump structure
  print(chat_out)
  stop("Unexpected response structure")
}

# ------------------------
# 5) Generate labels + leave-one-out stability
# ------------------------
generate_labels_for_all_factors <- function(prompts_list, defining_items, question_texts) {
  results <- list()
  for (i in seq_along(prompts_list)) {
    prompt <- prompts_list[[i]]
    factor_name <- names(prompts_list)[i]
    cat("Querying GPT-4 trubo for", factor_name, "...\n")
    
    out <- call_openai_chat(prompt, model = "gpt-4-turbo", max_completion_tokens = 20, temperature = 0)
    base_label <- extract_label_from_chat(out)
    
    # Leave-one-out prompts
    l1o <- defining_items[[i]]$item
    loo_labels <- map_dfr(seq_along(l1o), function(j) {
      reduced_items <- defining_items[[i]]$item[-j]
      reduced_df <- tibble(item = reduced_items) %>%
        left_join(question_texts, by = c("item" = "question_id"))
      
      reduced_prompt <- paste0(
        "These questionnaire items (subset) load on the same factor:\n",
        paste0("* ", reduced_df$item, ": ", reduced_df$item_text, collapse = "\n"),
        "\nProvide a concise 1–3 word noun phrase that best names the construct. Return only the phrase.\nLabel:"
      )
      
      cat(reduced_prompt, "\n\n", sep = "")
      
      out2 <- call_openai_chat(reduced_prompt, model = "gpt-4-turbo", max_completion_tokens = 20, temperature = 0)
      tibble(factor = factor_name,
             prompt_type = paste0("loo_remove_", l1o[j]),
             label = extract_label_from_chat(out2))
    })
    
    results[[factor_name]] <- bind_rows(
      tibble(factor = factor_name, prompt_type = "all_items", label = base_label),
      loo_labels
    )
    Sys.sleep(0.3) # polite delay
  }
  bind_rows(results)
}

# ------------------------
# 6) Run and print results
# ------------------------
labels_df <- generate_labels_for_all_factors(prompts_list, defining_items, question_texts)
print(labels_df)

# ------------------------
# 7) Summarise stability
# ------------------------
labels_summary <- labels_df %>%
  group_by(factor) %>%
  summarise(
    base_label = label[prompt_type == "all_items"][1],
    stability_prop = mean(label[prompt_type != "all_items"] ==
                            label[prompt_type == "all_items"][1]),
    .groups = "drop"
  )
print(labels_summary)
