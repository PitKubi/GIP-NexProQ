# =============================================================================
# Supplemental table behind Figure 5A, plus the quantified-peptide count check.
#
# Part 1 exports the exact per-sample concentration table the Figure 5A ranked
# protein plot is computed from (dbs_protein_rankplot.R logic: positive finite
# values, peptides quantified in at least 2 of the 10 DBS samples, geometric
# mean and log10 SD with clamped error bars). Output: tables/.
#
# Part 2 recounts how many peptides and proteins are actually quantified in
# the QC reproducibility data (wide_table_with_bloq.csv): 328 peptides from
# 311 proteins, against 330/313 with fitted calibration curves and the 334/318
# the Results text used to claim. Supports the reviewer-comment resolution.
# =============================================================================
suppressPackageStartupMessages({library(tidyverse)})

## ---- Part 1: the table behind Figure 5A (mirrors dbs_protein_rankplot.R) ----
df <- suppressMessages(readr::read_csv(file.path("data","validation cohort - first 10 samples - normalized_wide_table.csv"), guess_max = 100000))
dbs_cols <- names(df)[vapply(df, is.numeric, logical(1))]
dbs_cols <- dbs_cols[!str_detect(dbs_cols, regex("SE|BLOQ", ignore_case = TRUE))]
cat("numeric non-SE cols used by the figure:\n"); print(dbs_cols)

long <- df %>% select(Protein_Peptide, all_of(dbs_cols)) %>%
  pivot_longer(-Protein_Peptide, values_to="val", names_to="samp") %>%
  filter(is.finite(val), !is.na(val), val > 0)

stats <- long %>% group_by(Protein_Peptide) %>%
  summarise(n_pos=n(), log_m=mean(log10(val)), log_sd=sd(log10(val)),
            val_min=min(val), val_max=max(val), .groups="drop") %>%
  filter(n_pos > 1) %>%
  mutate(geo_mean=10^log_m,
         err_lo=pmax(10^(log_m-log_sd), val_min),
         err_hi=pmin(10^(log_m+log_sd), val_max)) %>%
  arrange(geo_mean) %>% mutate(rank=row_number())

prot <- function(x) sub("_[A-Za-z]+$","",x)
cat(sprintf("\nFig5A: %d peptide rows, %d unique proteins, conc %0.4g-%0.4g, median %0.3g, %.1f orders, quantified in all 10: %d\n",
  nrow(stats), n_distinct(prot(stats$Protein_Peptide)),
  min(stats$geo_mean), max(stats$geo_mean), median(stats$geo_mean),
  log10(max(stats$geo_mean)/min(stats$geo_mean)), sum(stats$n_pos==10)))

wide_conc <- df %>% select(Protein_Peptide, all_of(dbs_cols))
out <- stats %>%
  transmute(Protein = prot(Protein_Peptide),
            Peptide = sub("^.*_","",Protein_Peptide),
            Protein_Peptide, Rank = rank, N_samples_quantified = n_pos,
            Geometric_mean_concentration = signif(geo_mean, 4),
            SD_log10 = signif(log_sd, 4),
            Errorbar_lower = signif(err_lo, 4), Errorbar_upper = signif(err_hi, 4)) %>%
  left_join(wide_conc, by="Protein_Peptide") %>%
  arrange(desc(Rank))
dir.create("tables", showWarnings = FALSE)
readr::write_csv(out, file.path("tables", "SupplementalTable_Fig5A_concentrations_10DBS.csv"))
cat("wrote tables/SupplementalTable_Fig5A_concentrations_10DBS.csv\n")

## ---- Part 2: how many peptides/proteins in the Figure 4A/B CV analysis ----
qc <- readr::read_csv(file.path("data","wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
all_cols <- colnames(qc)
meas_cols <- all_cols[str_detect(all_cols, "^(dbs[123])_d\\d+_\\d+$") & !str_ends(all_cols, "SE")]
bloq_cols <- intersect(paste0(meas_cols, "_BLOQ"), all_cols)
measurements <- bind_cols(qc %>% transmute(protein_id=.[[1]]), qc[meas_cols]) %>%
  pivot_longer(-protein_id, names_to="sample", values_to="value_lin") %>%
  left_join(bind_cols(qc %>% transmute(protein_id=.[[1]]), qc[bloq_cols]) %>%
    pivot_longer(-protein_id, names_to="sample", values_to="is_bloq") %>%
    mutate(sample=str_remove(sample,"_BLOQ$")), by=c("protein_id","sample")) %>%
  separate(sample, into=c("group","day","rep"), sep="_", remove=FALSE) %>%
  filter(is.na(is_bloq) | is_bloq==FALSE, is.finite(value_lin))
sm <- measurements %>% group_by(sample) %>% summarise(med=median(value_lin,na.rm=TRUE),.groups="drop")
tm <- median(sm$med, na.rm=TRUE)
norm <- measurements %>% left_join(sm %>% mutate(scale=if_else(is.finite(med)&med>0, tm/med, 1)) %>% select(sample,scale), by="sample") %>%
  mutate(value_lin=value_lin*coalesce(scale,1)) %>% select(-scale)
hq_cut <- quantile(norm$value_lin, 0.25, na.rm=TRUE)
hq <- norm %>% filter(value_lin >= hq_cut)
cvf <- function(x) if (sum(is.finite(x))>=2 && mean(x,na.rm=TRUE)>0) 100*sd(x,na.rm=TRUE)/mean(x,na.rm=TRUE) else NA_real_
wp <- function(dat) dat %>% group_by(protein_id, day, rep) %>% summarise(cv=cvf(value_lin),.groups="drop") %>% filter(is.finite(cv))
wa <- wp(norm); wh <- wp(hq)
cat(sprintf("\nQC data: %d peptide rows total in wide table\n", nrow(qc)))
cat(sprintf("within-plex ALL  (inter-assay panel): median %.2f%% | peptides %d | proteins %d\n",
  median(wa$cv), n_distinct(wa$protein_id), n_distinct(prot(wa$protein_id))))
cat(sprintf("within-plex HQ   (intra-assay panel): median %.2f%% | peptides %d | proteins %d\n",
  median(wh$cv), n_distinct(wh$protein_id), n_distinct(prot(wh$protein_id))))
pep_any_quant <- measurements %>% distinct(protein_id)
cat(sprintf("peptides with ANY above-LOD measurement in QC runs: %d (proteins %d)\n",
  nrow(pep_any_quant), n_distinct(prot(pep_any_quant$protein_id))))
