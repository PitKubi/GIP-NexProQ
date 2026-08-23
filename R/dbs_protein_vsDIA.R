# --- TMT (DBS day-wise) vs DIA (protein matrix) correlation by gene ----------
# Uses log2-transformed values on both sides. TMT summarised as per-gene, per-day medians.
# DIA summarised as per-gene medians across all numeric columns.

# Paths (edit if needed)
tmt_path <- file.path("data", "wide_table_with_bloq.csv")
dia_path <- file.path("data", "report.pr_matrix.tsv")

pkgs <- c("tidyverse", "patchwork", "stringr")
need <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(need)) install.packages(need, repos = "https://cloud.r-project.org")
library(tidyverse); library(patchwork); library(stringr)

# ---------- Helpers -----------------------------------------------------------
guess_gene_col <- function(df) {
  candidates <- c("Genes","PG.Genes","Gene","gene","Gene.names","Gene.Names",
                  "GeneSymbol","Gene_Symbol","Gene.name","protein_id",
                  "Majority.protein.IDs","Protein.Names","Description","Fasta.headers")
  hit <- candidates[candidates %in% names(df)]
  if (length(hit)) return(hit[1])
  hit2 <- names(df)[grepl("gene", names(df), ignore.case = TRUE)]
  if (length(hit2)) return(hit2[1])
  NA_character_
}

extract_gene_from_text <- function(x) {
  x <- as.character(x)
  gn1 <- str_match(x, "GN=([A-Za-z0-9_.-]+)")[,2]
  gn2 <- ifelse(!is.na(gn1), gn1, str_match(x, "\\bgene[:=]([A-Za-z0-9_.-]+)")[,2])
  gn3 <- ifelse(!is.na(gn2), gn2, str_match(x, "\\[([A-Za-z0-9_.-]+)\\]")[,2])
  gn4 <- ifelse(!is.na(gn3), gn3, str_match(x, "\\b([A-Za-z0-9][A-Za-z0-9_.-]{0,15})\\b")[,2])
  toupper(gn4)
}

separate_multi_genes <- function(df, gene_col = "gene") {
  if (!gene_col %in% names(df)) return(df %>% mutate(gene = NA_character_))
  df %>%
    mutate(!!gene_col := gsub("\\s+", "", .data[[gene_col]])) %>%
    separate_rows(!!sym(gene_col), sep = ";|,|\\+")
}

min_pos_eps <- function(v) {
  vv <- suppressWarnings(as.numeric(v))
  m  <- suppressWarnings(min(vv[is.finite(vv) & vv > 0], na.rm = TRUE))
  if (!is.finite(m)) m <- 1e-6
  m/2
}

safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "pearson"))
}

# ---------- Read TMT (wide) ---------------------------------------------------
if (!file.exists(tmt_path)) {
  message("Choose your TMT wide table CSV..."); tmt_path <- file.choose()
}
message("Reading TMT: ", tmt_path)
tmt_raw <- readr::read_csv(tmt_path, guess_max = 1e6, show_col_types = FALSE)

# Measurement & flag columns
meas_regex <- "^(blank2|blank|dbs[123])_d\\d+_\\d+$"
meas_cols  <- names(tmt_raw)[str_detect(names(tmt_raw), meas_regex)]
se_cols    <- names(tmt_raw)[str_ends(names(tmt_raw), "SE")]
meas_cols  <- setdiff(meas_cols, se_cols)

flag_suffix_re <- "_(?i:flag|bloq|is_bloq|below_lod)$"
flag_regex     <- paste0("^(blank2|blank|dbs[123])_d\\d+_\\d+", flag_suffix_re)
flag_cols      <- names(tmt_raw)[str_detect(names(tmt_raw), regex(flag_regex, ignore_case = TRUE))]

if (!length(meas_cols)) stop("No TMT measurement columns matched (e.g. 'dbs1_d5_1').")

# Gene column or derive
tmt_gene_col <- guess_gene_col(tmt_raw)
if (is.na(tmt_gene_col)) {
  desc_cands <- c("Protein.Names","Description","Fasta.headers","Majority.protein.IDs")
  desc_cands <- desc_cands[desc_cands %in% names(tmt_raw)]
  if (length(desc_cands)) {
    tmt_raw <- tmt_raw %>% mutate(gene = extract_gene_from_text(.data[[desc_cands[1]]]))
  } else {
    id_cols <- setdiff(names(tmt_raw), c(meas_cols, se_cols, flag_cols))
    tmt_raw <- tmt_raw %>% unite("protein_id_tmp", all_of(id_cols), sep = "|", remove = FALSE)
    tmt_raw$gene <- extract_gene_from_text(tmt_raw$protein_id_tmp)
  }
} else {
  tmt_raw <- tmt_raw %>% mutate(gene = toupper(.data[[tmt_gene_col]]))
}

# Long TMT, parse day/replicate, keep DBS only, drop blanks
tmt_long <- tmt_raw %>%
  select(gene, all_of(meas_cols), all_of(flag_cols)) %>%
  pivot_longer(cols = all_of(meas_cols), names_to = "sample", values_to = "value") %>%
  tidyr::extract(sample, into = c("group","day","replicate"),
                 regex = "^(blank2|blank|dbs[123])_d(\\d+)_([0-9]+)$",
                 remove = FALSE, convert = TRUE) %>%
  mutate(is_blank = group %in% c("blank","blank2")) %>%
  filter(!is_blank, !is.na(gene), gene != "")

# Attach BLOQ flags (if present) and filter above-LOD
if (length(flag_cols)) {
  flags_long <- tmt_raw %>%
    select(gene, all_of(flag_cols)) %>%
    pivot_longer(cols = all_of(flag_cols), names_to = "flag_col", values_to = "flag_val") %>%
    mutate(sample  = str_replace(flag_col, regex(flag_suffix_re, ignore_case = TRUE), ""),
           flag_val_chr = tolower(as.character(flag_val)),
           flag_num = suppressWarnings(as.numeric(flag_val_chr)),
           is_bloq = case_when(
             flag_val_chr %in% c("true","t","yes","y","1","bloq","below_lod","<lod","<loq") ~ TRUE,
             flag_val_chr %in% c("false","f","no","n","0",">=lod","abovelod")               ~ FALSE,
             !is.na(flag_num) & flag_num > 0 ~ TRUE,
             !is.na(flag_num) & flag_num <= 0 ~ FALSE,
             TRUE ~ NA
           )) %>%
    select(gene, sample, is_bloq)

  tmt_long <- tmt_long %>%
    left_join(flags_long, by = c("gene","sample")) %>%
    filter(is.na(is_bloq) | is_bloq == FALSE)
}

# Numeric + log2
tmt_long <- tmt_long %>%
  mutate(value = suppressWarnings(as.numeric(value)))

eps_tmt <- min_pos_eps(tmt_long$value)
tmt_long <- tmt_long %>% mutate(tmt_log2 = log2(value + eps_tmt))

# Day-wise per-gene medians (DBS1/2/3 pooled within day)
tmt_day_gene <- tmt_long %>%
  group_by(day, gene) %>%
  summarise(tmt_log2_med = median(tmt_log2, na.rm = TRUE), .groups = "drop")

message("TMT: days found = {", paste(sort(unique(tmt_day_gene$day)), collapse = ", "), "}")

# ---------- Read DIA (protein matrix) ----------------------------------------
if (!file.exists(dia_path)) {
  message("Choose your DIA protein matrix (TSV/CSV)..."); dia_path <- file.choose()
}
message("Reading DIA: ", dia_path)
dia_raw <- if (grepl("\\.tsv$", dia_path, ignore.case = TRUE)) {
  readr::read_tsv(dia_path, guess_max = 1e6, show_col_types = FALSE)
} else {
  readr::read_csv(dia_path, guess_max = 1e6, show_col_types = FALSE)
}

# Gene from DIA
dia_gene_col <- guess_gene_col(dia_raw)
if (is.na(dia_gene_col)) {
  desc_cands <- c("Protein.Names","Description","Fasta.headers","Protein.Ids","Protein.Group","ProteinGroup")
  desc_cands <- desc_cands[desc_cands %in% names(dia_raw)]
  if (length(desc_cands)) {
    dia_raw <- dia_raw %>% mutate(gene = extract_gene_from_text(.data[[desc_cands[1]]]))
  } else {
    dia_raw$gene <- extract_gene_from_text(dia_raw[[1]])
  }
} else {
  dia_raw <- dia_raw %>% mutate(gene = toupper(.data[[dia_gene_col]]))
}

# Expand multi-gene rows; choose numeric intensity columns
dia_long_gene <- separate_multi_genes(dia_raw, "gene") %>%
  filter(!is.na(gene), gene != "")

num_cols <- names(dia_long_gene)[sapply(dia_long_gene, is.numeric)]
meta_cols <- c("gene", dia_gene_col, "Protein.Group","Protein.Ids","Protein.Names",
               "Genes","PG.Genes","Q.Value","P.Value","FDR","q.value")
num_cols <- setdiff(num_cols, intersect(num_cols, meta_cols))
if (!length(num_cols)) stop("No numeric intensity columns found in DIA matrix.")

eps_dia <- min_pos_eps(unlist(dia_long_gene[num_cols], use.names = FALSE))
dia_gene <- dia_long_gene %>%
  group_by(gene) %>%
  summarise(dia_log2_med = log2(median(as.numeric(unlist(across(all_of(num_cols)))), na.rm = TRUE) + eps_dia),
            .groups = "drop")

message("DIA genes: ", nrow(dia_gene))

# ---------- Merge per-day and correlate --------------------------------------
days <- sort(unique(tmt_day_gene$day))
plots <- list()
stats <- list()

for (d in days) {
  tmt_d <- tmt_day_gene %>% filter(day == d)
  merged <- tmt_d %>%
    inner_join(dia_gene, by = "gene")

  n <- nrow(merged)
  r <- if (n > 2) safe_cor(merged$tmt_log2_med, merged$dia_log2_med) else NA_real_
  stats[[as.character(d)]] <- tibble(day = d, n_overlap = n, pearson_r = r)

  if (n > 2) {
    p <- ggplot(merged, aes(x = tmt_log2_med, y = dia_log2_med)) +
      geom_point(alpha = 0.6, size = 1) +
      geom_smooth(method = "lm", se = FALSE) +
      labs(
        title = paste0("Day ", d, " — DBS (TMT) vs DIA"),
        subtitle = sprintf("n = %d | r = %.3f", n, r),
        x = "TMT per-gene median log2 (DBS pooled within day)",
        y = "DIA per-gene median log2"
      ) +
      theme_minimal(base_size = 10) +
      theme(plot.title = element_text(size = 10),
            plot.subtitle = element_text(size = 9))
    plots[[length(plots)+1]] <- p
  }
}

stats_df <- bind_rows(stats)
print(stats_df)

if (length(plots)) {
  ncol <- ceiling(sqrt(length(plots)))
  print(wrap_plots(plots, ncol = ncol))
} else {
  message("No day produced at least 3 overlapping genes; no plots drawn.")
}

# ---------- (Optional) overall pooled correlation (all days together) --------
merged_all <- tmt_day_gene %>%
  group_by(gene) %>%
  summarise(tmt_log2_med_overall = median(tmt_log2_med, na.rm = TRUE), .groups = "drop") %>%
  inner_join(dia_gene, by = "gene")

if (nrow(merged_all) > 2) {
  r_all <- safe_cor(merged_all$tmt_log2_med_overall, merged_all$dia_log2_med)
  message(sprintf("Overall pooled (all TMT days median) vs DIA: n=%d, r=%.3f",
                  nrow(merged_all), r_all))
  p_all <- ggplot(merged_all, aes(x = tmt_log2_med_overall, y = dia_log2_med)) +
    geom_point(alpha = 0.6, size = 1) +
    geom_smooth(method = "lm", se = FALSE) +
    labs(title = "All days pooled — DBS(TMT) vs DIA",
         subtitle = sprintf("n = %d | r = %.3f", nrow(merged_all), r_all),
         x = "TMT per-gene median log2 (days pooled)",
         y = "DIA per-gene median log2") +
    theme_minimal(base_size = 10)
  print(p_all)
}
