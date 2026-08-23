# Reconcile the two CV plots + define within-plex vs cross-plex explicitly.
#
# PLEX STRUCTURE: one run/injection = one TMT plex. Within a run the channels
# are standards (126-130N) + dbs1 + dbs2 + dbs3 + blanks.
#   * dbs1/dbs2/dbs3 are 3 channels in the SAME run  -> WITHIN-PLEX (same scan)
#   * rep1/rep2/rep3 are 3 SEPARATE runs same day     -> CROSS-PLEX (across injections)
#   * day1..day5                                        -> CROSS-PLEX (across days)
suppressPackageStartupMessages({library(tidyverse)})
dd <- "data"
df <- readr::read_csv(file.path(dd,"wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
ac <- colnames(df); mc <- ac[str_detect(ac,"^(dbs[123])_d\\d+_\\d+$") & !str_ends(ac,"SE")]
long <- bind_cols(df %>% transmute(pid=.[[1]]), df[mc]) %>%
  pivot_longer(-pid,names_to="s",values_to="v") %>%
  extract(s,c("day","rep"),"_d(\\d+)_(\\d+)$",remove=FALSE,convert=TRUE) %>%
  mutate(grp=str_extract(s,"^dbs[123]"), v=ifelse(is.finite(v)&v>0,v,NA)) %>% filter(is.finite(v))
# per-sample median normalization (same as figures)
sm <- long %>% group_by(s) %>% summarise(md=median(v,na.rm=TRUE),.groups="drop"); tg<-median(sm$md)
long <- long %>% left_join(sm,by="s") %>% mutate(v=v*tg/md) %>% select(-md)

# WITHIN-PLEX CV: across dbs1/2/3 within each (day, rep=run). protein-level int too.
wp <- long %>% group_by(day,rep,pid) %>%
  summarise(m=mean(v),s=sd(v),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv))
# CROSS-PLEX CV: across the 3 replicate runs within each (dbs, day)
xp <- long %>% group_by(grp,day,pid) %>%
  summarise(m=mean(v),s=sd(v),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv))

cat(sprintf("WITHIN-plex  (across dbs1/2/3, same run) median CV: %.1f%%\n", median(wp$cv)))
cat(sprintf("CROSS-plex   (across 3 replicate runs)   median CV: %.1f%%\n", median(xp$cv)))

cat("\n=== CV by intensity decile: does it trend? ===\n")
cat("decile   within-plex CV   cross-plex CV\n")
wpd <- wp %>% mutate(d=ntile(m,10)) %>% group_by(d) %>% summarise(cv=median(cv),.groups="drop")
xpd <- xp %>% mutate(d=ntile(m,10)) %>% group_by(d) %>% summarise(cv=median(cv),.groups="drop")
for(i in 1:10) cat(sprintf("  %2d        %5.1f%%          %5.1f%%\n", i, wpd$cv[i], xpd$cv[i]))

cat("\n=== top-fraction curve (keep top X%% by intensity), median CV ===\n")
cat("kept%   within-plex   cross-plex\n")
for(fr in c(1,.75,.5,.25,.1)){
  tw<-quantile(wp$m,1-fr); tx<-quantile(xp$m,1-fr)
  cat(sprintf(" %3.0f%%     %5.1f%%       %5.1f%%\n", fr*100,
      median(wp$cv[wp$m>=tw]), median(xp$cv[xp$m>=tx])))
}
