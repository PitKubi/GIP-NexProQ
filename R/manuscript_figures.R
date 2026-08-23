# =============================================================================
# NexProQ manuscript figures — readable, publication-ready regeneration
#
# Reproduces the SAME analysis as dbs_analysis_v2.R / dbs_protein_rankplot.R,
# but renders with large axis fonts and saves via ggsave() at 600 dpi + PDF
# (the old plots/*.png were hand-exported from the RStudio pane at low res,
#  which is why the axes were unreadable).
#
# Outputs (plots/):
#   Fig5AB_CV_histograms_manuscript.{png,pdf}   intra- (A) & inter-assay (B) CV
#   Fig5C_day_correlations_manuscript.{png,pdf}  pairwise day-to-day scatter grid
#   Fig6_protein_rank_manuscript.{png,pdf}       ranked concentration reference
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

data_dir <- "data"
out_dir  <- "plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

wide_csv <- file.path(data_dir, "wide_table_with_bloq.csv")
rank_csv <- file.path(data_dir, "validation cohort - first 10 samples - normalized_wide_table.csv")
stopifnot(file.exists(wide_csv), file.exists(rank_csv))

# ---- Publication theme: large, legible axes --------------------------------
theme_pub <- function(base_size = 16) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title   = element_text(size = base_size + 2, face = "bold", colour = "black"),
      axis.text    = element_text(size = base_size, colour = "black"),
      axis.line    = element_line(linewidth = 0.6, colour = "black"),
      axis.ticks   = element_line(linewidth = 0.6, colour = "black"),
      plot.title   = element_text(size = base_size + 2, face = "bold", hjust = 0),
      plot.subtitle= element_text(size = base_size, colour = "grey25"),
      plot.tag     = element_text(size = base_size + 6, face = "bold"),
      strip.text   = element_text(size = base_size, face = "bold"),
      legend.title = element_text(size = base_size),
      legend.text  = element_text(size = base_size - 1),
      plot.margin  = margin(10, 14, 10, 10)
    )
}

# =============================================================================
# Load & parse (identical logic to dbs_analysis_v2.R)
# =============================================================================
df_raw   <- readr::read_csv(wide_csv, guess_max = 1e6, show_col_types = FALSE)
all_cols <- colnames(df_raw)

meas_regex <- "^(blank2|blank|dbs[123])_d\\d+_\\d+$"
meas_cols  <- all_cols[stringr::str_detect(all_cols, meas_regex)]
se_cols    <- all_cols[stringr::str_ends(all_cols, "SE")]
meas_cols  <- setdiff(meas_cols, se_cols)
stopifnot(length(meas_cols) > 0)

flag_suffix_re <- "_(?i:flag|bloq|is_bloq|below_lod)$"
flag_regex     <- paste0("^(blank2|blank|dbs[123])_d\\d+_\\d+", flag_suffix_re)
flag_cols      <- all_cols[stringr::str_detect(all_cols, stringr::regex(flag_regex, ignore_case = TRUE))]

id_cols <- setdiff(all_cols, union(union(meas_cols, se_cols), flag_cols))
if (!length(id_cols)) { df_raw <- df_raw %>% mutate(.row_id = row_number()); id_cols <- ".row_id" }

df_ids_united <- df_raw %>%
  select(all_of(id_cols)) %>%
  tidyr::unite("protein_id", dplyr::everything(), sep = "|", remove = FALSE)

df_long_meas <- bind_cols(df_ids_united["protein_id"], df_raw[meas_cols]) %>%
  pivot_longer(-protein_id, names_to = "sample", values_to = "value") %>%
  tidyr::extract(sample, into = c("day", "replicate"),
                 regex = "_d(\\d+)_([0-9]+)$", remove = FALSE, convert = TRUE) %>%
  mutate(group = stringr::str_extract(sample, "^(blank2|blank|dbs[123])"),
         type  = if_else(stringr::str_detect(group, "^dbs"), "dbs", group))

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
  { if (!is.null(df_long_flags)) left_join(., df_long_flags, by = c("protein_id","sample")) else mutate(., is_bloq = NA) } %>%
  mutate(value_lin = if_else(is.finite(value) & value > 0, value, NA_real_))

dbs_clean <- df_long %>%
  filter(group %in% c("dbs1","dbs2","dbs3")) %>%   # remove blanks
  filter(is.na(is_bloq) | is_bloq == FALSE)        # above LOD

# ---- Normalization: median-of-sample-medians --------------------------------
sample_meds <- dbs_clean %>% group_by(sample) %>%
  summarise(med = median(value_lin, na.rm = TRUE), .groups = "drop")
target_med <- median(sample_meds$med, na.rm = TRUE)
scales_tbl <- sample_meds %>%
  mutate(scale = if_else(is.finite(med) & med > 0, target_med/med, 1)) %>%
  select(sample, scale)

dbs_norm <- dbs_clean %>%
  left_join(scales_tbl, by = "sample") %>%
  mutate(scale = coalesce(scale, 1), value_lin = value_lin * scale) %>%
  select(-scale)

min_pos_norm <- suppressWarnings(min(dbs_norm$value_lin[is.finite(dbs_norm$value_lin) & dbs_norm$value_lin > 0], na.rm = TRUE))
if (!is.finite(min_pos_norm)) min_pos_norm <- 1e-6
eps_norm <- min_pos_norm / 2
dbs_norm <- dbs_norm %>% mutate(value_log2 = log2(value_lin + eps_norm))

# =============================================================================
# Figure 5A/B — intra- (within-day) & inter-assay (across-day) CV histograms
# =============================================================================
within_day_cv <- dbs_norm %>%
  group_by(group, day, protein_id) %>%
  summarise(mean_lin = mean(value_lin, na.rm = TRUE),
            sd_lin   = sd(value_lin,   na.rm = TRUE),
            n_rep    = sum(is.finite(value_lin)), .groups = "drop") %>%
  mutate(cv = ifelse(is.finite(mean_lin) & mean_lin > 0 & is.finite(sd_lin) & n_rep >= 2,
                     100 * sd_lin / mean_lin, NA_real_)) %>%
  filter(is.finite(cv))

day_means <- dbs_norm %>%
  group_by(group, day, protein_id) %>%
  summarise(mean_lin = mean(value_lin, na.rm = TRUE), .groups = "drop")
across_day_cv <- day_means %>%
  group_by(group, protein_id) %>%
  summarise(mean_of_days = mean(mean_lin, na.rm = TRUE),
            sd_of_days   = sd(mean_lin,   na.rm = TRUE),
            n_days       = sum(is.finite(mean_lin)), .groups = "drop") %>%
  mutate(cv = ifelse(is.finite(mean_of_days) & mean_of_days > 0 & is.finite(sd_of_days) & n_days >= 2,
                     100 * sd_of_days / mean_of_days, NA_real_)) %>%
  filter(is.finite(cv))

# Per 2026-07-09 Vincent call: median only (no mean line/label), panel header
# moved into the x-axis label, and the "n = …; off-scale" subtitle removed for
# a clean publication panel.
cv_hist_panel <- function(cv_vec, bar_fill, line_col, xlab, tag) {
  d <- tibble(cv = cv_vec[is.finite(cv_vec)])
  median_cv <- median(d$cv)
  # cap x display at 99th pct (round to 10) so bars stay legible
  x_cap <- ceiling(quantile(d$cv, 0.99) / 10) * 10
  ggplot(d, aes(x = cv)) +
    geom_histogram(bins = 40, fill = bar_fill, colour = "white", linewidth = 0.2) +
    geom_vline(xintercept = median_cv, colour = line_col, linewidth = 1.0, linetype = "solid") +
    annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.3, size = 5, label.size = 0.3,
             label = sprintf("median = %.1f%%", median_cv)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 8), limits = c(0, x_cap), expand = expansion(mult = c(0, 0.02))) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 6), expand = expansion(mult = c(0, 0.08))) +
    labs(x = xlab, y = "Count", tag = tag) +
    theme_pub(16)
}

# Labels made plex-explicit: both A and B are CROSS-PLEX CVs (across separate
# TMT injections). Intra-day = the 3 replicate injections on one day; inter-day
# = across the 5 days. (Within-plex precision is shown separately, Fig 5D.)
pA <- cv_hist_panel(within_day_cv$cv, "#4292c6", "#084594",
                    "Intra-day CV (%)\n(cross-plex: replicate injections)", "A")
pB <- cv_hist_panel(across_day_cv$cv, "#fb6a4a", "#99000D",
                    "Inter-day CV (%)\n(cross-plex: across days)", "B")
fig5AB <- pA + pB + plot_layout(ncol = 2)

ggsave(file.path(out_dir, "Fig5AB_CV_histograms_manuscript.png"), fig5AB,
       width = 13, height = 5.5, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Fig5AB_CV_histograms_manuscript.pdf"), fig5AB,
       width = 13, height = 5.5)

# =============================================================================
# Figure 5D — replicate CV distributions as VIOLINS
#   Per 2026-07-09 Vincent call: the busy "day x replicate" CV histogram grid
#   (5 days x 3 replicates, from dbs_analysis_v2.R) is condensed into a single
#   violin panel — one violin per replicate, dodged within each day, with an
#   inset boxplot + median. Same replicate-reproducibility story, far more
#   readable than ~15 dodged histograms.
# =============================================================================
day_rep_cv <- dbs_norm %>%
  group_by(day, replicate, protein_id) %>%
  summarise(mean_lin = mean(value_lin, na.rm = TRUE),
            sd_lin   = sd(value_lin,   na.rm = TRUE),
            n_obs    = sum(is.finite(value_lin)), .groups = "drop") %>%
  mutate(cv = ifelse(is.finite(mean_lin) & mean_lin > 0 & is.finite(sd_lin) & n_obs >= 2,
                     100 * sd_lin / mean_lin, NA_real_)) %>%
  filter(is.finite(cv)) %>%
  mutate(day       = factor(paste0("Day ", day), levels = paste0("Day ", sort(unique(day)))),
         replicate = factor(replicate))

# cap the DISPLAY at the 95th pct (density computed on capped data → clean
# violin shapes, no thin high-CV spikes); note the trimmed fraction in caption
rep_cap  <- ceiling(quantile(day_rep_cv$cv, 0.95) / 10) * 10
n_trim   <- sum(day_rep_cv$cv > rep_cap)
pct_trim <- 100 * n_trim / nrow(day_rep_cv)
day_rep_disp <- day_rep_cv %>% filter(cv <= rep_cap)
dodge <- position_dodge(width = 0.85)

# median + IQR as an explicit crossbar per (day, replicate) — precomputed and
# drawn with group = replicate so it dodges reliably alongside the violins
# (an overlaid stat_summary / geom_boxplot collapses to the centre replicate)
rep_box <- day_rep_disp %>%
  group_by(day, replicate) %>%
  summarise(q25 = quantile(cv, .25, na.rm = TRUE),
            med = median(cv, na.rm = TRUE),
            q75 = quantile(cv, .75, na.rm = TRUE), .groups = "drop")

fig5D <- ggplot(day_rep_disp, aes(x = day, y = cv, fill = replicate)) +
  geom_violin(position = dodge, width = 0.8, scale = "width", trim = TRUE,
              colour = "grey30", linewidth = 0.3, alpha = 0.9) +
  geom_crossbar(data = rep_box,
                aes(x = day, y = med, ymin = q25, ymax = q75, group = replicate),
                position = position_dodge(width = 0.85), width = 0.22,
                fill = "white", colour = "grey15", linewidth = 0.35,
                inherit.aes = FALSE) +
  scale_fill_brewer(palette = "Blues", name = "Replicate") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 6),
                     expand = expansion(mult = c(0, 0.03)), limits = c(0, rep_cap)) +
  labs(x = NULL, y = "Within-plex CV (%)  (across DBS spots)",
       caption = sprintf("Display capped at %g%% CV (%.1f%% of values above, off-scale)", rep_cap, pct_trim)) +
  theme_pub(16) +
  theme(plot.caption = element_text(size = 11, colour = "grey35"))

ggsave(file.path(out_dir, "Fig5D_replicate_CV_violin_manuscript.png"), fig5D,
       width = 11, height = 6, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Fig5D_replicate_CV_violin_manuscript.pdf"), fig5D,
       width = 11, height = 6)

# ---- Boxplot variant of the same panel (manuscript alternative) ------------
fig5D_box <- ggplot(day_rep_disp, aes(x = day, y = cv, fill = replicate)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7,
               outlier.size = 0.5, outlier.alpha = 0.25,
               colour = "grey20", linewidth = 0.35) +
  scale_fill_brewer(palette = "Blues", name = "Replicate") +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 6),
                     expand = expansion(mult = c(0, 0.03)), limits = c(0, rep_cap)) +
  labs(x = NULL, y = "Within-plex CV (%)  (across DBS spots)",
       caption = sprintf("Display capped at %g%% CV (%.1f%% of values above, off-scale)", rep_cap, pct_trim)) +
  theme_pub(16) +
  theme(plot.caption = element_text(size = 11, colour = "grey35"))

ggsave(file.path(out_dir, "Fig5D_replicate_CV_boxplot_manuscript.png"), fig5D_box,
       width = 11, height = 6, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Fig5D_replicate_CV_boxplot_manuscript.pdf"), fig5D_box,
       width = 11, height = 6)

# =============================================================================
# Figure 5C — pairwise day-to-day correlation grid (log2, normalized)
# =============================================================================
safe_cor <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y); ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "pearson"))
}

dbs_day_means <- dbs_norm %>%
  group_by(protein_id, day) %>%
  summarise(value_log2 = mean(value_log2, na.rm = TRUE), .groups = "drop")
days <- sort(unique(dbs_day_means$day))
wide <- dbs_day_means %>%
  pivot_wider(names_from = day, values_from = value_log2, names_prefix = "day")

make_scatter <- function(df, d_a, d_b) {
  xa <- paste0("day", d_a); xb <- paste0("day", d_b)
  if (!all(c(xa, xb) %in% names(df))) return(NULL)
  r_val <- safe_cor(df[[xa]], df[[xb]])
  ggplot(df, aes(x = .data[[xa]], y = .data[[xb]])) +
    geom_point(alpha = 0.55, size = 1.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
    geom_smooth(method = "lm", se = FALSE, colour = "#1f6fe5", linewidth = 1.0) +
    annotate("label", x = -Inf, y = Inf, hjust = -0.05, vjust = 1.2, size = 5, label.size = 0.3,
             label = sprintf("r = %.3f", r_val)) +
    labs(x = paste0("Day ", d_a, " (log2)"), y = paste0("Day ", d_b, " (log2)")) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
    coord_equal() +
    theme_pub(15)
}

pairs <- t(combn(days, 2))
plots <- Filter(Negate(is.null), apply(pairs, 1, function(p) make_scatter(wide, p[1], p[2])))
ncol  <- ceiling(sqrt(length(plots)))
nrow  <- ceiling(length(plots) / ncol)
fig5C <- wrap_plots(plots, ncol = ncol)

ggsave(file.path(out_dir, "Fig5C_day_correlations_manuscript.png"), fig5C,
       width = 4.6 * ncol, height = 4.4 * nrow, dpi = 500, bg = "white", limitsize = FALSE)
ggsave(file.path(out_dir, "Fig5C_day_correlations_manuscript.pdf"), fig5C,
       width = 4.6 * ncol, height = 4.4 * nrow, limitsize = FALSE)

# =============================================================================
# Figure 6 — ranked protein concentration reference ranges
#            (logic from dbs_protein_rankplot.R)
# =============================================================================
rdf <- suppressMessages(readr::read_csv(rank_csv, guess_max = 100000, show_col_types = FALSE))
dbs_cols <- names(rdf)[vapply(rdf, is.numeric, logical(1))]
dbs_cols <- dbs_cols[!str_detect(dbs_cols, regex("SE", ignore_case = TRUE))]
stopifnot(length(dbs_cols) >= 2)

rank_stats <- rdf %>%
  select(all_of(dbs_cols)) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, values_to = "val") %>%
  filter(is.finite(val), !is.na(val), val > 0) %>%
  group_by(row_id) %>%
  summarise(n_pos = n(), log_m = mean(log10(val)), log_sd = sd(log10(val)),
            val_min = min(val), val_max = max(val), .groups = "drop") %>%
  filter(n_pos > 1) %>%
  mutate(y_mean = 10^log_m,
         y_min  = pmax(10^(log_m - log_sd), val_min),
         y_max  = pmin(10^(log_m + log_sd), val_max)) %>%
  arrange(y_mean) %>% mutate(rank = row_number())

fig6 <- ggplot(rank_stats, aes(x = rank, y = y_mean)) +
  geom_errorbar(aes(ymin = y_min, ymax = y_max), width = 0, linewidth = 0.4, colour = "grey55") +
  geom_point(size = 1.8, alpha = 0.8) +
  scale_y_log10(labels = label_number(drop0trailing = TRUE, big.mark = ""),
                breaks = scales::breaks_log(n = 7)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10), expand = expansion(mult = c(0.01, 0.03))) +
  labs(x = "Protein rank", y = "Concentration (log scale)") +
  annotation_logticks(sides = "l") +
  theme_pub(16)

ggsave(file.path(out_dir, "Fig6_protein_rank_manuscript.png"), fig6,
       width = 9.5, height = 6, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Fig6_protein_rank_manuscript.pdf"), fig6,
       width = 9.5, height = 6)

message("Done. Saved manuscript figures to: ", out_dir)
