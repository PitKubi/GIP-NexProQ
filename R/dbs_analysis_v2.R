csv_path <- file.path("data", "wide_table_with_bloq.csv")

# --- DBS analysis (normalized to median-of-sample-medians; BLOQ removed; blanks removed)
#     Correlations use log2; CVs use normalized linear.
#     Includes:
#       - Per-group CV histograms (+ pooled), with mean & median lines and labels
#       - Day-wise CV histograms pooled across DBS1/2/3, dodged by replicate (FIXED labels)
#       - Median CV vs retained-top fraction (keep top 100%..10% by intensity)
#       - Per–day × replicate histogram (DBS pooled) with median line + label in each facet

# Packages
pkgs <- c("tidyverse", "patchwork")
need <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(need)) install.packages(need, repos = "https://cloud.r-project.org")
library(tidyverse)
library(patchwork)

# ======================= Config ==============================================
if (!file.exists(csv_path)) {
  message("Choose your wide table CSV...")
  csv_path <- file.choose()
}
message("Reading: ", csv_path)

lin_cutoff <- NULL                 # HQ subset cutoff (defaults to top 75% if NULL)
kept_fracs <- seq(1.00, 0.10, -0.05)

# ======================= Load & Parse ========================================
df_raw   <- readr::read_csv(csv_path, guess_max = 1e6, show_col_types = FALSE)
all_cols <- colnames(df_raw)

meas_regex <- "^(blank2|blank|dbs[123])_d\\d+_\\d+$"
meas_cols  <- all_cols[stringr::str_detect(all_cols, meas_regex)]
se_cols    <- all_cols[stringr::str_ends(all_cols, "SE")]
meas_cols  <- setdiff(meas_cols, se_cols)
if (length(meas_cols) == 0) stop("No measurement columns matched (e.g. 'dbs1_d5_1').")

flag_suffix_re <- "_(?i:flag|bloq|is_bloq|below_lod)$"
flag_regex     <- paste0("^(blank2|blank|dbs[123])_d\\d+_\\d+", flag_suffix_re)
flag_cols      <- all_cols[stringr::str_detect(all_cols, stringr::regex(flag_regex, ignore_case = TRUE))]

id_cols <- setdiff(all_cols, union(union(meas_cols, se_cols), flag_cols))
if (!length(id_cols)) {
  df_raw <- df_raw %>% mutate(.row_id = row_number())
  id_cols <- ".row_id"
}

df_ids_united <- df_raw %>%
  select(all_of(id_cols)) %>%
  tidyr::unite("protein_id", dplyr::everything(), sep = "|", remove = FALSE)

df_long_meas <- bind_cols(df_ids_united["protein_id"], df_raw[meas_cols]) %>%
  pivot_longer(-protein_id, names_to = "sample", values_to = "value") %>%
  tidyr::extract(sample, into = c("day", "replicate"),
                 regex = "_d(\\d+)_([0-9]+)$", remove = FALSE, convert = TRUE) %>%
  mutate(
    group = stringr::str_extract(sample, "^(blank2|blank|dbs[123])"),
    type  = if_else(stringr::str_detect(group, "^dbs"), "dbs", group)
  )

df_long_flags <- NULL
if (length(flag_cols) > 0) {
  df_long_flags <- bind_cols(df_ids_united["protein_id"], df_raw[flag_cols]) %>%
    pivot_longer(-protein_id, names_to = "flag_col", values_to = "is_bloq_raw") %>%
    mutate(
      sample  = stringr::str_replace(flag_col, stringr::regex(flag_suffix_re, ignore_case = TRUE), ""),
      val_chr = tolower(as.character(is_bloq_raw)),
      num_val = suppressWarnings(as.numeric(val_chr)),
      is_bloq = dplyr::case_when(
        val_chr %in% c("true","t","yes","y","1","bloq","below_lod","<lod","<loq") ~ TRUE,
        val_chr %in% c("false","f","no","n","0",">=lod","abovelod")             ~ FALSE,
        !is.na(num_val) & num_val > 0 ~ TRUE,
        !is.na(num_val) & num_val <= 0 ~ FALSE,
        TRUE ~ NA
      )
    ) %>% select(protein_id, sample, is_bloq)
}

df_long <- df_long_meas %>%
  { if (!is.null(df_long_flags)) left_join(., df_long_flags, by = c("protein_id","sample")) else mutate(., is_bloq = NA) }

all_meas_vec <- suppressWarnings(as.numeric(unlist(df_raw[meas_cols], use.names = FALSE)))
min_pos      <- suppressWarnings(min(all_meas_vec[is.finite(all_meas_vec) & all_meas_vec > 0], na.rm = TRUE))
if (!is.finite(min_pos)) min_pos <- 1e-6

df_long <- df_long %>% mutate(value_lin = if_else(is.finite(value) & value > 0, value, NA_real_))

dbs_clean <- df_long %>%
  filter(group %in% c("dbs1","dbs2","dbs3")) %>%        # remove blanks
  filter(is.na(is_bloq) | is_bloq == FALSE)             # above LOD

message(
  "Days: ", paste(sort(unique(dbs_clean$day)), collapse = ", "),
  " | Max replicate: ", suppressWarnings(max(dbs_clean$replicate, na.rm = TRUE)),
  " | N unique samples: ", dplyr::n_distinct(dbs_clean$sample)
)

# ======================= Normalization (linear) ==============================
sample_meds <- dbs_clean %>%
  group_by(sample) %>%
  summarise(med = median(value_lin, na.rm = TRUE), .groups = "drop")

target_med <- median(sample_meds$med, na.rm = TRUE)

scales <- sample_meds %>%
  mutate(scale = if_else(is.finite(med) & med > 0, target_med/med, 1)) %>%
  select(sample, scale)

dbs_norm <- dbs_clean %>%
  left_join(scales, by = "sample") %>%
  mutate(
    scale     = coalesce(scale, 1),
    value_lin = value_lin * scale
  ) %>%
  select(-scale)

min_pos_norm <- suppressWarnings(min(dbs_norm$value_lin[is.finite(dbs_norm$value_lin) & dbs_norm$value_lin > 0], na.rm = TRUE))
if (!is.finite(min_pos_norm)) min_pos_norm <- 1e-6
eps_norm <- min_pos_norm / 2
dbs_norm <- dbs_norm %>% mutate(value_log2 = log2(value_lin + eps_norm))

# HQ subset (default: top 75% intensities)
if (is.null(lin_cutoff)) lin_cutoff <- quantile(dbs_norm$value_lin, 0.25, na.rm = TRUE)
dbs_norm_hq <- dbs_norm %>% filter(value_lin >= lin_cutoff)

# ======================= Helpers =============================================
safe_cor <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "pearson"))
}

make_cor_grid <- function(dat, title_prefix = "DBS pooled (normalized)") {
  dbs_day_means <- dat %>%
    group_by(protein_id, day) %>%
    summarise(value_log2 = mean(value_log2, na.rm = TRUE), .groups = "drop")

  days <- sort(unique(dbs_day_means$day))
  if (length(days) < 2) return(NULL)

  wide <- dbs_day_means %>%
    pivot_wider(names_from = day, values_from = value_log2, names_prefix = "day")

  make_scatter <- function(df, d_a, d_b) {
    xa <- paste0("day", d_a); xb <- paste0("day", d_b)
    if (!all(c(xa, xb) %in% names(df))) return(NULL)
    r_val <- safe_cor(df[[xa]], df[[xb]])
    ggplot(df, aes_string(x = xa, y = xb)) +
      geom_point(alpha = 0.6, size = 0.9) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      geom_smooth(method = "lm", se = FALSE) +
      labs(
        title    = paste0(title_prefix, ": Day ", d_a, " vs Day ", d_b),
        x        = paste0("Day ", d_a, " (mean log2 intensity)"),
        y        = paste0("Day ", d_b, " (mean log2 intensity)"),
        subtitle = sprintf("Pearson r = %.3f", r_val)
      ) +
      theme_minimal(base_size = 9) +
      theme(
        plot.title    = element_text(size = 9),
        plot.subtitle = element_text(size = 8),
        axis.text     = element_text(size = 8),
        axis.title    = element_text(size = 9)
      )
  }

  pairs <- t(combn(days, 2))
  plots <- apply(pairs, 1, function(p) make_scatter(wide, p[1], p[2]))
  plots <- Filter(Negate(is.null), plots)
  if (!length(plots)) return(NULL)

  nplots <- length(plots)
  ncol   <- ceiling(sqrt(nplots))
  wrap_plots(plots, ncol = ncol)
}

cv_from_data <- function(dat) {
  replicate_cv <- dat %>%
    group_by(group, day, protein_id) %>%
    summarise(
      mean_lin = mean(value_lin, na.rm = TRUE),
      sd_lin   = sd(value_lin,   na.rm = TRUE),
      n_rep    = sum(is.finite(value_lin)),
      .groups  = "drop"
    ) %>%
    mutate(
      cv = ifelse(is.finite(mean_lin) & mean_lin > 0 & is.finite(sd_lin) & n_rep >= 2,
                  100 * sd_lin / mean_lin, NA_real_),
      cv_type = "within_day_replicates"
    ) %>%
    filter(is.finite(cv))

  day_means <- dat %>%
    group_by(group, day, protein_id) %>%
    summarise(mean_lin = mean(value_lin, na.rm = TRUE), .groups = "drop")

  across_day_cv <- day_means %>%
    group_by(group, protein_id) %>%
    summarise(
      mean_of_days = mean(mean_lin, na.rm = TRUE),
      sd_of_days   = sd(mean_lin,   na.rm = TRUE),
      n_days       = sum(is.finite(mean_lin)),
      .groups      = "drop"
    ) %>%
    mutate(
      cv = ifelse(is.finite(mean_of_days) & mean_of_days > 0 & is.finite(sd_of_days) & n_days >= 2,
                  100 * sd_of_days / mean_of_days, NA_real_),
      cv_type = "across_days"
    ) %>%
    filter(is.finite(cv))

  bind_rows(
    replicate_cv %>% select(group, cv, cv_type),
    across_day_cv %>% select(group, cv, cv_type)
  ) %>%
    mutate(cv_type = factor(cv_type,
                            levels = c("within_day_replicates","across_days"),
                            labels = c("Within-day","Across-days")))
}

cv_grid_plots <- function(cv_tbl, title_tag = "normalized, BLOQ removed") {
  cv_pooled <- cv_tbl %>% mutate(group = "DBS pooled")
  dat <- bind_rows(cv_tbl, cv_pooled)

  fill_pal <- c("Within-day" = "#4292c6", "Across-days" = "#fb6a4a")
  line_pal <- c("Within-day" = "#084594", "Across-days" = "#99000D")

  cv_hist_for_group <- function(g) {
    d <- dat %>% filter(group == g)
    if (nrow(d) == 0) return(NULL)

    lab_stats <- d %>%
      group_by(cv_type) %>%
      summarise(
        median_cv = median(cv, na.rm = TRUE),
        mean_cv   = mean(cv,   na.rm = TRUE),
        .groups   = "drop"
      ) %>%
      arrange(cv_type)

    rng  <- range(d$cv, na.rm = TRUE)
    brks <- if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) 30 else seq(rng[1], rng[2], length.out = 51)

    max_count <- function(vec, brks) {
      if (length(brks) == 1) return(length(vec))
      max(hist(vec, breaks = brks, plot = FALSE)$counts, na.rm = TRUE)
    }
    y_max <- max(
      max_count(d$cv[d$cv_type == "Within-day"],  brks),
      max_count(d$cv[d$cv_type == "Across-days"], brks),
      na.rm = TRUE
    )
    if (!is.finite(y_max) || y_max <= 0) y_max <- 1

    x_right <- quantile(d$cv, 0.98, na.rm = TRUE)
    if (!is.finite(x_right)) x_right <- rng[2]
    y_levels <- c(0.92, 0.80)

    lab_df <- lab_stats %>%
      mutate(
        x     = x_right,
        y     = y_max * y_levels[pmin(row_number(), length(y_levels))],
        label = paste0(as.character(cv_type),
                       " mean=",   sprintf("%.1f", mean_cv), "%; ",
                       "median=", sprintf("%.1f", median_cv), "%")
      )

    ggplot(d, aes(x = cv, fill = cv_type)) +
      geom_histogram(position = "dodge", bins = 50, show.legend = FALSE) +
      geom_vline(data = lab_stats, aes(xintercept = mean_cv,   color = cv_type),
                 linetype = "solid",  linewidth = 0.8, show.legend = FALSE) +
      geom_vline(data = lab_stats, aes(xintercept = median_cv, color = cv_type),
                 linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
      geom_label(data = lab_df,
                 aes(x = x, y = y, label = label, color = cv_type),
                 fill = "white", label.size = 0.25, size = 3.0,
                 hjust = 1, show.legend = FALSE) +
      scale_fill_manual(values = fill_pal) +
      scale_color_manual(values = line_pal) +
      labs(title = paste0("CV% — ", g, " (", title_tag, ")"),
           x = "CV (%)", y = "Count") +
      theme_minimal(base_size = 10) +
      theme(legend.position = "none",
            plot.title = element_text(size = 10))
  }

  groups <- c("dbs1","dbs2","dbs3","DBS pooled")
  plots  <- lapply(groups, cv_hist_for_group)
  plots  <- Filter(Negate(is.null), plots)
  if (!length(plots)) return(NULL)
  wrap_plots(plots, ncol = 2)
}

# ---------- Day-wise CV (DBS pooled) — histograms DODGED by replicate (FIXED) ------
cv_daywise_pooled_byrep_hist_grid <- function(dat, title_tag = "normalized") {
  day_rep_cv <- dat %>%
    group_by(day, replicate, protein_id) %>%
    summarise(
      mean_lin = mean(value_lin, na.rm = TRUE),
      sd_lin   = sd(value_lin,   na.rm = TRUE),
      n_obs    = sum(is.finite(value_lin)),
      .groups  = "drop_last"
    ) %>%
    mutate(cv = ifelse(is.finite(mean_lin) & mean_lin > 0 & is.finite(sd_lin) & n_obs >= 2,
                       100 * sd_lin / mean_lin, NA_real_)) %>%
    ungroup() %>%
    filter(is.finite(cv))

  if (!nrow(day_rep_cv)) return(NULL)

  stats_df <- day_rep_cv %>%
    group_by(day, replicate) %>%
    summarise(
      mean_cv   = mean(cv, na.rm = TRUE),
      median_cv = median(cv, na.rm = TRUE),
      .groups   = "drop"
    )

  # Per-day global range, X position for labels, and Y max for dodged hist
  per_day <- day_rep_cv %>%
    group_by(day) %>%
    summarise(
      rng_min = min(cv, na.rm = TRUE),
      rng_max = max(cv, na.rm = TRUE),
      x_right = quantile(cv, 0.98, na.rm = TRUE),
      y_max   = {
        if (is.finite(rng_min) && is.finite(rng_max) && rng_max > rng_min) {
          brks <- seq(rng_min, rng_max, length.out = 51)
          max(hist(cv, breaks = brks, plot = FALSE)$counts, na.rm = TRUE)
        } else length(cv)
      },
      .groups = "drop"
    )

  # Build label rows for each (day, replicate) with staggered Y levels
  labels <- day_rep_cv %>%
    distinct(day, replicate) %>%
    arrange(day, replicate) %>%
    group_by(day) %>%
    mutate(idx = row_number(),
           n_reps = n()) %>%
    ungroup() %>%
    left_join(per_day, by = "day") %>%
    mutate(
      # stagger y positions per replicate within a day
      y = y_max * (ifelse(n_reps > 1,
                          seq(0.92, 0.72, length.out = n_reps)[idx],
                          0.92)),
      x = x_right
    ) %>%
    left_join(stats_df, by = c("day","replicate")) %>%
    mutate(label = paste0("rep ", replicate,
                          "  mean=",   sprintf("%.1f", mean_cv), "%; ",
                          "median=", sprintf("%.1f", median_cv), "%"))

  ggplot(day_rep_cv, aes(x = cv, fill = factor(replicate))) +
    geom_histogram(position = "dodge", bins = 50, alpha = 0.85) +
    geom_vline(data = stats_df,
               aes(xintercept = mean_cv, color = factor(replicate)),
               linewidth = 0.8, linetype = "solid", show.legend = FALSE) +
    geom_vline(data = stats_df,
               aes(xintercept = median_cv, color = factor(replicate)),
               linewidth = 0.8, linetype = "dashed", show.legend = FALSE) +
    geom_label(data = labels,
               aes(x = x, y = y, label = label, color = factor(replicate)),
               fill = "white", label.size = 0.25, size = 3.0,
               hjust = 1, show.legend = FALSE) +
    facet_wrap(~ day, scales = "free_y") +
    labs(title = paste0("Day-wise CV% (DBS pooled; dodged by replicate; ", title_tag, ")"),
         x = "CV (%)", y = "Count", fill = "Replicate", color = "Replicate") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 10))
}

# ---------- Core pooled table used by NEW analyses ---------------------------
cv_pooled_dayrep_table <- function(dat) {
  dat %>%
    group_by(day, replicate, protein_id) %>%
    summarise(
      mean_lin = mean(value_lin, na.rm = TRUE),
      sd_lin   = sd(value_lin,   na.rm = TRUE),
      n_obs    = sum(is.finite(value_lin)),
      int_lin  = median(value_lin,  na.rm = TRUE),
      int_log2 = median(value_log2, na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    mutate(
      cv = ifelse(is.finite(mean_lin) & mean_lin > 0 & is.finite(sd_lin) & n_obs >= 2,
                  100 * sd_lin / mean_lin, NA_real_)
    ) %>%
    filter(is.finite(cv), is.finite(int_log2), is.finite(int_lin))
}

# ---------- Median CV vs retained-top fraction (by intensity) ----------------
median_cv_vs_retained_plot <- function(dat, kept_fracs = seq(1, 0.1, by = -0.05),
                                       title_tag = "normalized") {
  pooled <- cv_pooled_dayrep_table(dat)
  if (!nrow(pooled)) return(NULL)

  qfun <- function(frac) quantile(pooled$int_lin, probs = 1 - frac, na.rm = TRUE)

  res_list <- lapply(kept_fracs, function(fr) {
    thr <- qfun(fr)
    kept <- pooled %>% filter(int_lin >= thr)
    kept %>%
      group_by(day, replicate) %>%
      summarise(median_cv = median(cv, na.rm = TRUE),
                n_proteins = dplyr::n(), .groups = "drop") %>%
      mutate(kept_frac = fr)
  })

  summ <- bind_rows(res_list)
  if (!nrow(summ)) return(NULL)

  ggplot(summ, aes(x = kept_frac * 100, y = median_cv,
                   color = factor(day), group = interaction(day, replicate))) +
    geom_line() +
    geom_point(size = 1) +
    facet_wrap(~ replicate, nrow = 1, scales = "free_y") +
    scale_x_reverse(breaks = seq(100, 10, by = -10)) +
    labs(
      title = paste0("Median CV vs retained top fraction (DBS pooled per day×rep; ", title_tag, ")"),
      x = "Percent of proteins retained (top by intensity)",
      y = "Median CV (%)",
      color = "Day"
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 10))
}

# ---------- Per–day × replicate histograms (DBS pooled) with medians ---------
cv_dayrep_faceted_hist_with_median <- function(dat, title_tag = "normalized") {
  pooled <- cv_pooled_dayrep_table(dat)
  if (!nrow(pooled)) return(NULL)

  stats_df <- pooled %>%
    group_by(day, replicate) %>%
    summarise(
      median_cv = median(cv, na.rm = TRUE),
      mean_cv   = mean(cv,   na.rm = TRUE),
      .groups   = "drop"
    )

  # Compute facet label positions per (day, replicate)
  facets <- expand.grid(day = sort(unique(pooled$day)),
                        replicate = sort(unique(pooled$replicate)))
  build_positions <- function(d, r) {
    dd <- pooled %>% filter(day == d, replicate == r)
    if (!nrow(dd)) return(NULL)
    rng  <- range(dd$cv, na.rm = TRUE)
    brks <- if (is.finite(rng[1]) && is.finite(rng[2]) && rng[2] > rng[1]) seq(rng[1], rng[2], length.out = 51) else 30
    cnts <- if (length(brks) == 1) length(dd$cv) else hist(dd$cv, breaks = brks, plot = FALSE)$counts
    y_max <- ifelse(length(cnts), max(cnts, na.rm = TRUE), 1)
    if (!is.finite(y_max) || y_max <= 0) y_max <- 1
    x_right <- quantile(dd$cv, 0.98, na.rm = TRUE)
    if (!is.finite(x_right)) x_right <- rng[2]
    tibble(day = d, replicate = r, x = x_right, y = y_max * 0.92)
  }
  pos_df <- purrr::map2_dfr(facets$day, facets$replicate, build_positions)

  lab_df <- stats_df %>%
    left_join(pos_df, by = c("day","replicate")) %>%
    mutate(label = paste0("median = ", sprintf("%.1f", median_cv), "%"))

  ggplot(pooled, aes(x = cv)) +
    geom_histogram(bins = 50, fill = "#4292c6", alpha = 0.85) +
    geom_vline(data = stats_df, aes(xintercept = median_cv),
               linetype = "dashed", linewidth = 0.9, color = "#084594") +
    geom_label(data = lab_df,
               aes(x = x, y = y, label = label),
               fill = "white", label.size = 0.25, size = 3.1,
               hjust = 1, show.legend = FALSE) +
    facet_grid(replicate ~ day, scales = "free_y") +
    labs(title = paste0("CV% per facet (DBS pooled) — day × replicate (", title_tag, ")"),
         x = "CV (%)", y = "Count") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 10))
}

# ======================= 1) Correlations across days (grids) =================
p_cor_all <- make_cor_grid(dbs_norm,  "DBS pooled (normalized)")
if (!is.null(p_cor_all)) print(p_cor_all)

p_cor_hq  <- make_cor_grid(dbs_norm_hq, "DBS pooled (normalized, HQ subset)")
if (!is.null(p_cor_hq))  print(p_cor_hq)

# ======================= 2) CV% histograms (per group + pooled) =============
cv_all <- cv_from_data(dbs_norm)
p_cv_all <- cv_grid_plots(cv_all, title_tag = "normalized, BLOQ removed")
if (!is.null(p_cv_all)) print(p_cv_all)

cv_hq <- cv_from_data(dbs_norm_hq)
p_cv_hq <- cv_grid_plots(cv_hq, title_tag = "normalized, HQ subset (intensity cutoff)")
if (!is.null(p_cv_hq)) print(p_cv_hq)

# ======================= 2b) Day-wise CV (DBS pooled) — DODGED by replicate ==
p_cv_day_all <- cv_daywise_pooled_byrep_hist_grid(dbs_norm,  title_tag = "normalized")
if (!is.null(p_cv_day_all)) print(p_cv_day_all)

p_cv_day_hq  <- cv_daywise_pooled_byrep_hist_grid(dbs_norm_hq, title_tag = "normalized, HQ subset")
if (!is.null(p_cv_day_hq)) print(p_cv_day_hq)

# ======================= 2c) Median CV vs retained-top fraction ==============
p_median_cv_curve <- median_cv_vs_retained_plot(dbs_norm, kept_fracs = kept_fracs,
                                                title_tag = "normalized")
if (!is.null(p_median_cv_curve)) print(p_median_cv_curve)

# ======================= 2d) Per–day × replicate histograms (DBS pooled) ====
p_cv_dayrep_facets_all <- cv_dayrep_faceted_hist_with_median(dbs_norm, "normalized")
if (!is.null(p_cv_dayrep_facets_all)) print(p_cv_dayrep_facets_all)

# ======================= (Optional) Boxplots on normalized DBS (log2) ========
df_box <- dbs_norm %>% mutate(day = factor(day, levels = sort(unique(day))))
p_box_all <- ggplot(df_box, aes(x = group, y = value_log2, fill = group)) +
  geom_boxplot(outlier.size = 0.6, width = 0.7) +
  facet_wrap(~ day, nrow = 1, labeller = label_both) +
  labs(title = "DBS over days (log2, normalized; BLOQ removed)",
       x = "Group", y = "log2 intensity") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
print(p_box_all)

# ======================= Done ===============================================
message("Done. Day-wise dodged CV plot fixed; labels placed without pmap().")
