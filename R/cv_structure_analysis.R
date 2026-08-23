# Grounding analysis for Peter's questions:
#   Q2  what do the 3 replicates mean, and the CV per replicate?
#   Q3  can we lower the combined intra/inter-day CV? subset? batch removal?
# Reuses the exact load+normalize logic from manuscript_figures.R.
suppressPackageStartupMessages({library(tidyverse)})
proj_dir <- "/home/pk/R projects/Science/NexProQ"; data_dir <- file.path(proj_dir,"data")
df_raw <- readr::read_csv(file.path(data_dir,"wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
all_cols <- colnames(df_raw)
meas_regex <- "^(blank2|blank|dbs[123])_d\\d+_\\d+$"
meas_cols <- setdiff(all_cols[str_detect(all_cols,meas_regex)], all_cols[str_ends(all_cols,"SE")])
flag_suffix_re <- "_(?i:flag|bloq|is_bloq|below_lod)$"
flag_cols <- all_cols[str_detect(all_cols, regex(paste0("^(blank2|blank|dbs[123])_d\\d+_\\d+",flag_suffix_re), ignore_case=TRUE))]
id_cols <- setdiff(all_cols, union(union(meas_cols, all_cols[str_ends(all_cols,"SE")]), flag_cols))
if(!length(id_cols)){df_raw<-df_raw%>%mutate(.row_id=row_number());id_cols<-".row_id"}
ids <- df_raw %>% select(all_of(id_cols)) %>% unite("protein_id", everything(), sep="|")
long <- bind_cols(ids, df_raw[meas_cols]) %>%
  pivot_longer(-protein_id, names_to="sample", values_to="value") %>%
  extract(sample, into=c("day","replicate"), regex="_d(\\d+)_([0-9]+)$", remove=FALSE, convert=TRUE) %>%
  mutate(group=str_extract(sample,"^(blank2|blank|dbs[123])")) %>%
  filter(group %in% c("dbs1","dbs2","dbs3")) %>%
  mutate(value_lin=if_else(is.finite(value)&value>0, value, NA_real_))
# median-of-sample-medians normalization (identical to the figure script)
sm <- long %>% group_by(sample) %>% summarise(med=median(value_lin,na.rm=TRUE),.groups="drop")
tgt <- median(sm$med,na.rm=TRUE)
norm <- long %>% left_join(sm,by="sample") %>%
  mutate(value_lin=value_lin*if_else(is.finite(med)&med>0,tgt/med,1)) %>% select(-med)

cvmed <- function(d, ...){ g<-enquos(...)
  d %>% group_by(!!!g, protein_id) %>%
    summarise(m=mean(value_lin,na.rm=TRUE), s=sd(value_lin,na.rm=TRUE), n=sum(is.finite(value_lin)),.groups="drop") %>%
    mutate(cv=ifelse(is.finite(m)&m>0&is.finite(s)&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% pull(cv) %>% median()
}
cat("=== BASELINE median CV by direction (median-normalized) ===\n")
cat(sprintf("Intra-day (across 3 replicate injections; per dbs,day): %.1f%%\n", cvmed(norm, group, day)))
day_means <- norm %>% group_by(group,day,protein_id) %>% summarise(value_lin=mean(value_lin,na.rm=TRUE),.groups="drop")
inter_day <- day_means %>% group_by(group,protein_id) %>% summarise(m=mean(value_lin,na.rm=TRUE),s=sd(value_lin,na.rm=TRUE),n=sum(is.finite(value_lin)),.groups="drop") %>% mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% pull(cv)
cat(sprintf("Inter-day (across 5 days; per dbs):                    %.1f%%\n", median(inter_day)))
cat(sprintf("Inter-spot (across dbs1/2/3; per day,replicate):       %.1f%%\n", cvmed(norm, day, replicate)))

# ---- Q3a: subset by intensity (keep top X%% most intense proteins) ----------
prot_int <- norm %>% group_by(protein_id) %>% summarise(mean_int=mean(value_lin,na.rm=TRUE),.groups="drop")
cat("\n=== INTRA-day median CV vs intensity subset ===\n")
for(f in c(1,0.75,0.5,0.25)){
  keep <- prot_int %>% slice_max(mean_int, prop=f) %>% pull(protein_id)
  cat(sprintf("keep top %3.0f%% by intensity (n=%d): %.1f%%\n", 100*f, length(keep), cvmed(filter(norm,protein_id%in%keep), group, day)))
}

# ---- Q3b: remove batch effects beyond median normalization ------------------
# median norm only rescales each SAMPLE globally. It does NOT remove systematic
# DAY or REPLICATE(channel) offsets that act per-protein. Center each protein's
# log2 values by day (and by replicate) to strip those batch effects, then
# recompute intra-day CV.
norm2 <- norm %>% mutate(l2=log2(value_lin))
# day-centering: subtract per-(protein,group,day)? no—that would remove signal.
# Correct batch removal = subtract the systematic day/replicate MEAN shift shared
# across proteins (a batch factor), estimated per sample as median residual.
# Here: remove per-day and per-replicate global offsets (already mostly gone via
# median norm) AND per-protein day-block means to test the ceiling.
intra_raw <- cvmed(norm, group, day)
# strip replicate(channel) effect: center each (protein,group,day) block is the
# CV itself; instead remove the replicate main-effect per protein across all data
rep_off <- norm2 %>% group_by(protein_id, replicate) %>% summarise(ro=mean(l2,na.rm=TRUE),.groups="drop")
rep_all <- norm2 %>% group_by(protein_id) %>% summarise(mo=mean(l2,na.rm=TRUE),.groups="drop")
norm2 <- norm2 %>% left_join(rep_off,by=c("protein_id","replicate")) %>% left_join(rep_all,by="protein_id") %>%
  mutate(l2_adj=l2-(ro-mo), value_adj=2^l2_adj)
cv_adj <- norm2 %>% group_by(group,day,protein_id) %>%
  summarise(m=mean(value_adj,na.rm=TRUE),s=sd(value_adj,na.rm=TRUE),n=sum(is.finite(value_adj)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% pull(cv)
cat("\n=== BATCH removal (strip replicate/channel main-effect per protein) ===\n")
cat(sprintf("Intra-day CV  baseline: %.1f%%   after removing replicate effect: %.1f%%\n", intra_raw, median(cv_adj)))
