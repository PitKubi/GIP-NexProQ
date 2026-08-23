# Why is the intra-day replicate CV ~32% even after per-plex calibration?
# TMT-informed diagnostics on wide_table_with_bloq.csv (values = concentrations).
suppressPackageStartupMessages({library(tidyverse)})
dd <- "/home/pk/R projects/Science/NexProQ/data"
df <- readr::read_csv(file.path(dd,"wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
ac <- colnames(df)
mc <- ac[str_detect(ac,"^(blank2|blank|dbs[123])_d\\d+_\\d+$") & !str_ends(ac,"SE")]
ids <- df %>% transmute(pid = .[[1]])
long <- bind_cols(ids, df[mc]) %>%
  pivot_longer(-pid, names_to="s", values_to="v") %>%
  extract(s, c("day","rep"), "_d(\\d+)_(\\d+)$", remove=FALSE, convert=TRUE) %>%
  mutate(grp=str_extract(s,"^(blank2|blank|dbs[123])"),
         v=ifelse(is.finite(v)&v>0,v,NA))

# ---- 1. BLANK / interference floor: blank conc vs sample conc ----------------
cat("=== 1. Interference floor (median concentration by channel) ===\n")
long %>% group_by(grp) %>% summarise(median_conc=median(v,na.rm=TRUE),
        detect_rate=mean(is.finite(v)), .groups="drop") %>% print()
samp_med  <- median(long$v[long$grp %in% c("dbs1","dbs2","dbs3")], na.rm=TRUE)
blank_med <- median(long$v[long$grp %in% c("blank","blank2")], na.rm=TRUE)
cat(sprintf("blank/sample median ratio = %.1f%%  (interference as %% of signal)\n\n", 100*blank_med/samp_med))

dbs <- long %>% filter(grp %in% c("dbs1","dbs2","dbs3"))

# ---- 2. Intra-day CV vs concentration DECILE (is it near-LOD driven?) --------
wd <- dbs %>% group_by(grp,day,pid) %>%
  summarise(m=mean(v,na.rm=TRUE), s=sd(v,na.rm=TRUE), n=sum(is.finite(v)), .groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv))
cat("=== 2. Intra-day CV by concentration decile (low->high) ===\n")
wd %>% mutate(dec=ntile(m,10)) %>% group_by(dec) %>%
  summarise(conc_lo=min(m), conc_hi=max(m), median_CV=median(cv), n=n(), .groups="drop") %>%
  mutate(across(c(conc_lo,conc_hi,median_CV), ~round(.,2))) %>% print(n=10)

# ---- 3. Per-DAY systematic effect (is a run/day an outlier?) -----------------
cat("\n=== 3. Median intra-day CV by day and by DBS sample ===\n")
wd %>% group_by(day) %>% summarise(median_CV=round(median(cv),1),.groups="drop") %>% print()
wd %>% group_by(grp) %>% summarise(median_CV=round(median(cv),1),.groups="drop") %>% print()

# ---- 4. Variance partition: within-run(rep) vs between-day vs between-dbs -----
# For each protein, decompose log2 concentration variance across the 3 factors.
cat("\n=== 4. Variance components (log2 conc) averaged over proteins ===\n")
vc <- dbs %>% mutate(l2=log2(v)) %>% filter(is.finite(l2)) %>%
  group_by(pid) %>% filter(n()>=8) %>%
  summarise(
    v_total = var(l2),
    v_rep   = { a<-aggregate(l2~rep, cur_data(), mean); var(a$l2) },
    v_day   = { a<-aggregate(l2~day, cur_data(), mean); var(a$l2) },
    v_dbs   = { a<-aggregate(l2~grp, cur_data(), mean); var(a$l2) },
    .groups="drop")
cat(sprintf("median total var       : %.3f\n", median(vc$v_total,na.rm=TRUE)))
cat(sprintf("median var(replicate)  : %.3f  (%.0f%% of total)\n", median(vc$v_rep,na.rm=TRUE), 100*median(vc$v_rep,na.rm=TRUE)/median(vc$v_total,na.rm=TRUE)))
cat(sprintf("median var(day)        : %.3f  (%.0f%% of total)\n", median(vc$v_day,na.rm=TRUE), 100*median(vc$v_day,na.rm=TRUE)/median(vc$v_total,na.rm=TRUE)))
cat(sprintf("median var(dbs sample) : %.3f  (%.0f%% of total)\n", median(vc$v_dbs,na.rm=TRUE), 100*median(vc$v_dbs,na.rm=TRUE)/median(vc$v_total,na.rm=TRUE)))

# ---- 5. How much data is near the calibration bottom (<= lowest std 0.02) ----
cat("\n=== 5. Fraction of sample values at/below calibration floor ===\n")
cat(sprintf("<= 0.02 (lowest std): %.1f%%   <= 0.1 (2nd lowest): %.1f%%\n",
            100*mean(dbs$v<=0.02,na.rm=TRUE), 100*mean(dbs$v<=0.1,na.rm=TRUE)))
