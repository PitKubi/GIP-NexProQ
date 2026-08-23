# --- Ranked protein means with log-SD error bars (clamped), publication-ready with more axis labels ---

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(scales)
})

# Input
csv_path <- file.path("data", "validation cohort - first 10 samples - normalized_wide_table.csv")
stopifnot(file.exists(csv_path))
df <- suppressMessages(readr::read_csv(csv_path, guess_max = 100000))

# Numeric DBS columns excluding "SE"
dbs_cols <- names(df)[vapply(df, is.numeric, logical(1))]
dbs_cols <- dbs_cols[!str_detect(dbs_cols, regex("SE", ignore_case = TRUE))]
if (length(dbs_cols) < 2) stop("Need >= 2 numeric DBS columns after excluding 'SE'.")

# Compute mean ± SD in log10 domain, clamp to observed min/max
stats <- df %>%
  select(all_of(dbs_cols)) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, values_to = "val") %>%
  filter(is.finite(val), !is.na(val), val > 0) %>%
  group_by(row_id) %>%
  summarise(
    n_pos   = n(),
    log_m   = mean(log10(val)),
    log_sd  = sd(log10(val)),
    val_min = min(val),
    val_max = max(val),
    .groups = "drop"
  ) %>%
  filter(n_pos > 1) %>%
  mutate(
    y_mean   = 10^log_m,
    y_min    = pmax(10^(log_m - log_sd), val_min),
    y_max    = pmin(10^(log_m + log_sd), val_max)
  ) %>%
  arrange(y_mean) %>%
  mutate(rank = row_number())

# Publication theme
theme_pub <- theme_classic(base_size = 14) +
  theme(
    axis.title = element_text(size = 16),
    axis.text  = element_text(size = 12, color = "black"),
    axis.line  = element_line(linewidth = 0.6, colour = "black"),
    panel.border = element_rect(fill = NA, colour = "black", linewidth = 0.6),
    plot.title = element_text(size = 16, face = "bold", hjust = 0),
    plot.margin = margin(8, 12, 8, 8)
  )

# Plot
p <- ggplot(stats, aes(x = rank, y = y_mean)) +
  geom_errorbar(aes(ymin = y_min, ymax = y_max), width = 0, linewidth = 0.4) +
  geom_point(size = 1.6, alpha = 0.75) +
  scale_y_log10(
    labels = label_number(accuracy = 0.1),
    breaks = scales::breaks_log(n = 6)   # more tick marks on log scale
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 10)  # ~10 evenly spaced ticks
  ) +
  labs(
    x = "Protein Rank",
    y = "Absolute Protein Conc (log scale)"
  ) +
  coord_cartesian(expand = FALSE) +
  theme_pub

print(p)

# Save
out_dir <- "plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
png_file <- file.path(out_dir, "ranked_means_logSD_clamped_pub.png")
pdf_file <- file.path(out_dir, "ranked_means_logSD_clamped_pub.pdf")

ggsave(png_file, p, width = 7.5, height = 5.0, dpi = 600)
ggsave(pdf_file, p, width = 7.5, height = 5.0, useDingbats = FALSE)
message("Saved:\n  ", png_file, "\n  ", pdf_file)
