# =============================================================================
# Supplemental Figure 1 — DIA-PASEF discovery proteomics of pooled DBS
#
# The peptide panel for the targeted assay was selected from a DIA-PASEF
# analysis of pooled DBS run in triplicate. This figure documents that starting
# point: depth per run, reproducibility across the three runs, and the abundance
# range the selected peptides had to span.
#
# Counts here are the ones quoted in the Results: 1037 protein groups over the
# three runs, 982 shared by all three, and 6885 proteotypic peptides quantified
# in all three runs.
#
# No interpretation text is drawn inside the panels; it belongs in the legend.
#
# Input : data/report.pr_matrix.tsv   (DIA-NN precursor matrix, 3 runs)
# Output: plots/SupplementalFigure1_discovery.{png,pdf}
# =============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

data_dir <- "data"
out_dir  <- "plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

theme_pub <- function(base_size = 15) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title = element_text(size = base_size + 2, face = "bold", colour = "black"),
      axis.text  = element_text(size = base_size, colour = "black"),
      axis.line  = element_line(linewidth = 0.6, colour = "black"),
      axis.ticks = element_line(linewidth = 0.6, colour = "black"),
      plot.tag   = element_text(size = base_size + 6, face = "bold"),
      legend.position = "none",
      plot.margin = margin(10, 14, 10, 10)
    )
}

BLUE <- "#4292c6"; DARK_BLUE <- "#084594"; GREEN <- "#41ab5d"

precursors <- readr::read_tsv(file.path(data_dir, "report.pr_matrix.tsv"),
                              show_col_types = FALSE, guess_max = 1e6)

run_columns <- names(precursors)[startsWith(names(precursors), "D:")]
short_run_name <- function(path) {
  base <- sub("\\.d$", "", basename(gsub("\\\\", "/", path)))
  sub("^EV0*([0-9]+)_.*$", "Run \\1", base)
}

long <- precursors %>%
  select(Protein.Group, Stripped.Sequence, Proteotypic, all_of(run_columns)) %>%
  pivot_longer(all_of(run_columns), names_to = "run_path", values_to = "quantity") %>%
  filter(is.finite(quantity), quantity > 0) %>%
  mutate(run = short_run_name(run_path))

# ---- A: depth per run, with the intersection of all three -------------------
per_run <- long %>%
  group_by(run) %>%
  summarise(Proteins = n_distinct(Protein.Group),
            Peptides = n_distinct(Stripped.Sequence), .groups = "drop")

runs_per_protein <- long %>% distinct(run, Protein.Group) %>% count(Protein.Group)
runs_per_peptide <- long %>% distinct(run, Stripped.Sequence) %>% count(Stripped.Sequence)
n_runs <- length(run_columns)

shared <- tibble(
  run = "All three runs",
  Proteins = sum(runs_per_protein$n == n_runs),
  Peptides = sum(runs_per_peptide$n == n_runs)
)

depth <- bind_rows(per_run, shared) %>%
  pivot_longer(c(Proteins, Peptides), names_to = "level", values_to = "count") %>%
  mutate(run = factor(run, levels = c(per_run$run, "All three runs")),
         level = factor(level, levels = c("Proteins", "Peptides")))

panel_a <- ggplot(depth, aes(x = run, y = count, fill = run == "All three runs")) +
  geom_col(width = 0.68, colour = "white") +
  geom_text(aes(label = comma(count)), vjust = -0.45, size = 4.6) +
  facet_wrap(~level, scales = "free_y") +
  scale_fill_manual(values = c(`FALSE` = BLUE, `TRUE` = DARK_BLUE)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)), labels = comma,
                     breaks = pretty_breaks(n = 5)) +
  labs(x = NULL, y = "Identifications", tag = "A") +
  theme_pub(14) +
  theme(strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 30, hjust = 1, size = 12))

# ---- B: how many of the three runs each protein was seen in ----------------
detection <- runs_per_protein %>%
  count(n, name = "proteins") %>%
  mutate(n = factor(n, levels = seq_len(n_runs)))

panel_b <- ggplot(detection, aes(x = n, y = proteins)) +
  geom_col(width = 0.62, fill = GREEN, colour = "white") +
  geom_text(aes(label = comma(proteins)), vjust = -0.45, size = 4.6) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.16)), labels = comma) +
  labs(x = "Runs quantified in", y = "Protein groups", tag = "B") +
  theme_pub(14)

# ---- C: dynamic range covered by the shared proteins ----------------------
shared_proteins <- runs_per_protein %>% filter(n == n_runs) %>% pull(Protein.Group)

ranked <- long %>%
  filter(Protein.Group %in% shared_proteins) %>%
  group_by(Protein.Group, run) %>%
  summarise(run_quantity = sum(quantity), .groups = "drop") %>%
  group_by(Protein.Group) %>%
  summarise(mean_quantity = mean(run_quantity), .groups = "drop") %>%
  arrange(desc(mean_quantity)) %>%
  mutate(rank = row_number())

orders_of_magnitude <- log10(max(ranked$mean_quantity) / min(ranked$mean_quantity))

panel_c <- ggplot(ranked, aes(x = rank, y = mean_quantity)) +
  geom_line(colour = DARK_BLUE, linewidth = 0.9) +
  scale_y_log10(labels = trans_format("log10", math_format(10^.x)),
                breaks = trans_breaks("log10", function(x) 10^x)) +
  annotation_logticks(sides = "l") +
  labs(x = "Protein rank", y = "Mean DIA-PASEF quantity", tag = "C") +
  theme_pub(14)

# ---- D: agreement between the three runs ----------------------------------
protein_by_run <- long %>%
  filter(Protein.Group %in% shared_proteins) %>%
  group_by(Protein.Group, run) %>%
  summarise(run_quantity = sum(quantity), .groups = "drop") %>%
  mutate(log10_quantity = log10(run_quantity))

replicate_cv <- long %>%
  filter(Protein.Group %in% shared_proteins) %>%
  group_by(Protein.Group, run) %>%
  summarise(run_quantity = sum(quantity), .groups = "drop") %>%
  group_by(Protein.Group) %>%
  summarise(cv = 100 * sd(run_quantity) / mean(run_quantity), .groups = "drop") %>%
  filter(is.finite(cv))

median_discovery_cv <- median(replicate_cv$cv)

panel_d <- ggplot(replicate_cv, aes(x = cv)) +
  geom_histogram(bins = 40, fill = BLUE, colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = median_discovery_cv, colour = DARK_BLUE, linewidth = 1.0) +
  scale_x_continuous(breaks = pretty_breaks(n = 8),
                     limits = c(0, ceiling(quantile(replicate_cv$cv, 0.99) / 5) * 5),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = "CV across the three runs (%)", y = "Protein groups", tag = "D") +
  theme_pub(14)

supplemental_figure_1 <- (panel_a + panel_b + plot_layout(widths = c(1.7, 1))) /
  (panel_c + panel_d)

ggsave(file.path(out_dir, "SupplementalFigure1_discovery.png"), supplemental_figure_1,
       width = 13, height = 9.5, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "SupplementalFigure1_discovery.pdf"), supplemental_figure_1,
       width = 13, height = 9.5)

message("--- numbers for the Supplemental Figure 1 legend ---")
message(sprintf("protein groups over all runs : %d", n_distinct(long$Protein.Group)))
message(sprintf("protein groups in all %d runs : %d", n_runs, length(shared_proteins)))
message(sprintf("peptides in all %d runs       : %d", n_runs, sum(runs_per_peptide$n == n_runs)))
# The Results quote proteotypic peptides for which a SINGLE precursor was
# quantified in every run. Pooling precursors across runs instead would let a
# peptide qualify on different precursors in different runs and gives 6894.
proteotypic_in_every_run <- precursors %>%
  filter(Proteotypic == 1) %>%
  filter(if_all(all_of(run_columns), ~ is.finite(.x) & .x > 0)) %>%
  distinct(Stripped.Sequence) %>%
  nrow()
message(sprintf("proteotypic peptides in all %d: %d", n_runs, proteotypic_in_every_run))
message(sprintf("dynamic range (orders of mag): %.1f", orders_of_magnitude))
message(sprintf("median CV across runs (%%)     : %.1f", median_discovery_cv))
