# =============================================================================
# Figure 5A/B rebuilt on the WITHIN-PLEX axis, plus a comparison of every
# reproducibility axis available in this experiment.
#
# Why: the Results quote 11.28% and 13.98%, which trace to the within-plex CV
# (the three DBS channels inside a single injection) with and without the
# top-75%-by-intensity HQ subset from dbs_analysis_v2.R. The Figure 5A/B that
# ships today plots the cross-plex axis instead, at 31.7% and 33.5%. This script
# renders both so the choice can be made on the evidence.
#
# Outputs (plots/):
#   Fig5AB_withinplex_manuscript.{png,pdf}   publication-styled, no panel text
#   Fig5_CV_axis_comparison.{png,pdf}        decision aid, medians labelled
# =============================================================================
suppressPackageStartupMessages({library(tidyverse); library(patchwork); library(scales)})

data_dir <- "data"; out_dir <- "plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

theme_pub <- function(base_size = 16) {
  theme_classic(base_size = base_size) +
    theme(axis.title = element_text(size = base_size + 2, face = "bold", colour = "black"),
          axis.text  = element_text(size = base_size, colour = "black"),
          axis.line  = element_line(linewidth = 0.6, colour = "black"),
          axis.ticks = element_line(linewidth = 0.6, colour = "black"),
          plot.tag   = element_text(size = base_size + 6, face = "bold"),
          plot.title = element_text(size = base_size, face = "bold", hjust = 0),
          plot.margin = margin(10, 14, 10, 10))
}

df <- readr::read_csv(file.path(data_dir, "wide_table_with_bloq.csv"),
                      guess_max = 1e6, show_col_types = FALSE)
all_cols  <- colnames(df)
meas_cols <- all_cols[str_detect(all_cols, "^(dbs[123])_d\\d+_\\d+$") & !str_ends(all_cols, "SE")]
bloq_cols <- intersect(paste0(meas_cols, "_BLOQ"), all_cols)

measurements <- bind_cols(df %>% transmute(protein_id = .[[1]]), df[meas_cols]) %>%
  pivot_longer(-protein_id, names_to = "sample", values_to = "value_lin") %>%
  left_join(bind_cols(df %>% transmute(protein_id = .[[1]]), df[bloq_cols]) %>%
              pivot_longer(-protein_id, names_to = "sample", values_to = "is_bloq") %>%
              mutate(sample = str_remove(sample, "_BLOQ$")),
            by = c("protein_id", "sample")) %>%
  separate(sample, into = c("group", "day", "rep"), sep = "_", remove = FALSE) %>%
  filter(is.na(is_bloq) | is_bloq == FALSE, is.finite(value_lin))

sample_medians <- measurements %>% group_by(sample) %>%
  summarise(med = median(value_lin, na.rm = TRUE), .groups = "drop")
target_median <- median(sample_medians$med, na.rm = TRUE)
normalised <- measurements %>%
  left_join(sample_medians %>%
              mutate(scale = if_else(is.finite(med) & med > 0, target_median / med, 1)) %>%
              select(sample, scale), by = "sample") %>%
  mutate(value_lin = value_lin * coalesce(scale, 1)) %>% select(-scale)

# The HQ subset of dbs_analysis_v2.R: drop the least intense quarter of values.
hq_cutoff <- quantile(normalised$value_lin, 0.25, na.rm = TRUE)
high_quality <- normalised %>% filter(value_lin >= hq_cutoff)

coefficient_of_variation <- function(x) {
  if (sum(is.finite(x)) >= 2 && mean(x, na.rm = TRUE) > 0)
    100 * sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE) else NA_real_
}

# The three DBS channels inside one injection: precision within a single plex.
within_plex_cv <- function(dat) dat %>%
  group_by(protein_id, day, rep) %>%
  summarise(cv = coefficient_of_variation(value_lin), .groups = "drop") %>%
  filter(is.finite(cv)) %>% pull(cv)

# The three replicate injections made on one day, per DBS spot.
cross_plex_intra_day_cv <- function(dat) dat %>%
  group_by(protein_id, group, day) %>%
  summarise(cv = coefficient_of_variation(value_lin), .groups = "drop") %>%
  filter(is.finite(cv)) %>% pull(cv)

# Daily means across the five days, per DBS spot.
cross_plex_inter_day_cv <- function(dat) dat %>%
  group_by(protein_id, group, day) %>%
  summarise(m = mean(value_lin, na.rm = TRUE), .groups = "drop") %>%
  group_by(protein_id, group) %>%
  summarise(cv = coefficient_of_variation(m), .groups = "drop") %>%
  filter(is.finite(cv)) %>% pull(cv)

cv_panel <- function(cv_vec, bar_fill, line_col, xlab, tag,
                     show_median_label = FALSE, title = NULL, x_cap = NULL) {
  d <- tibble(cv = cv_vec[is.finite(cv_vec)])
  median_cv <- median(d$cv)
  if (is.null(x_cap)) x_cap <- ceiling(quantile(d$cv, 0.99) / 10) * 10
  p <- ggplot(d, aes(x = cv)) +
    geom_histogram(bins = 40, fill = bar_fill, colour = "white", linewidth = 0.2) +
    geom_vline(xintercept = median_cv, colour = line_col, linewidth = 1.0) +
    scale_x_continuous(breaks = pretty_breaks(n = 8), limits = c(0, x_cap),
                       expand = expansion(mult = c(0, 0.02))) +
    scale_y_continuous(breaks = pretty_breaks(n = 6), expand = expansion(mult = c(0, 0.10))) +
    labs(x = xlab, y = "Count", tag = tag, title = title) +
    theme_pub(15)
  if (show_median_label)
    p <- p + annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.3, size = 5,
                      label.size = 0.3, label = sprintf("median = %.2f%%", median_cv))
  p
}

BLUE <- "#4292c6"; DARK_BLUE <- "#084594"; RED <- "#fb6a4a"; DARK_RED <- "#99000D"
GREEN <- "#41ab5d"; DARK_GREEN <- "#005a32"

wp_hq  <- within_plex_cv(high_quality)
wp_all <- within_plex_cv(normalised)
cp_intra <- cross_plex_intra_day_cv(normalised)
cp_inter <- cross_plex_inter_day_cv(normalised)

# ---- drop-in replacement: Figure 5A/B on the within-plex axis ---------------
# Labelled with the Results' own wording so the figure and the sentence that
# cites it agree. Both panels are the within-plex CV; A applies the HQ subset.
fig5AB_within_plex <-
  cv_panel(wp_hq,  BLUE, DARK_BLUE, "Intra-assay CV (%)", "A") +
  cv_panel(wp_all, RED,  DARK_RED,  "Inter-assay CV (%)", "B") +
  plot_layout(ncol = 2)

ggsave(file.path(out_dir, "Fig5AB_withinplex_manuscript.png"), fig5AB_within_plex,
       width = 13, height = 5.5, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Fig5AB_withinplex_manuscript.pdf"), fig5AB_within_plex,
       width = 13, height = 5.5)

# ---- decision aid: every axis on a shared scale, medians labelled -----------
shared_cap <- 100
comparison <-
  cv_panel(wp_hq, BLUE, DARK_BLUE, "CV (%)", "A", TRUE,
           "Within-plex, HQ subset\n(quoted as intra-assay, 11.28%)", shared_cap) +
  cv_panel(wp_all, BLUE, DARK_BLUE, "CV (%)", "B", TRUE,
           "Within-plex, all peptides\n(quoted as inter-assay, 13.98%)", shared_cap) +
  cv_panel(cp_intra, GREEN, DARK_GREEN, "CV (%)", "C", TRUE,
           "Cross-plex, intra-day\n(Figure 5A as it ships today)", shared_cap) +
  cv_panel(cp_inter, RED, DARK_RED, "CV (%)", "D", TRUE,
           "Cross-plex, inter-day\n(Figure 5B as it ships today)", shared_cap) +
  plot_layout(ncol = 2)

ggsave(file.path(out_dir, "Fig5_CV_axis_comparison.png"), comparison,
       width = 14, height = 10, dpi = 400, bg = "white")
ggsave(file.path(out_dir, "Fig5_CV_axis_comparison.pdf"), comparison,
       width = 14, height = 10)

message("--- medians ---")
message(sprintf("within-plex, HQ subset (top 75%%) : %.2f%%  (n=%d)", median(wp_hq), length(wp_hq)))
message(sprintf("within-plex, all peptides        : %.2f%%  (n=%d)", median(wp_all), length(wp_all)))
message(sprintf("cross-plex, intra-day            : %.2f%%  (n=%d)", median(cp_intra), length(cp_intra)))
message(sprintf("cross-plex, inter-day            : %.2f%%  (n=%d)", median(cp_inter), length(cp_inter)))
