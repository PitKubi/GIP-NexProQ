suppressPackageStartupMessages({library(tidyverse)})
dd <- "/home/pk/R projects/Science/NexProQ/data"
df <- readr::read_csv(file.path(dd,"wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
ac <- colnames(df); mc <- ac[str_detect(ac,"^(dbs[123])_d\\d+_\\d+$") & !str_ends(ac,"SE")]
long <- bind_cols(df %>% transmute(pid=.[[1]]), df[mc]) %>%
  pivot_longer(-pid,names_to="s",values_to="v") %>%
  extract(s,c("day","rep"),"_d(\\d+)_(\\d+)$",remove=FALSE,convert=TRUE) %>%
  mutate(grp=str_extract(s,"^dbs[123]"), v=ifelse(is.finite(v)&v>0,v,NA)) %>% filter(is.finite(v))

cvmed <- function(d) d %>% group_by(grp,day,pid) %>%
  summarise(m=mean(v,na.rm=TRUE),s=sd(v,na.rm=TRUE),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% pull(cv) %>% median()
interday <- function(d){ dm<-d %>% group_by(grp,day,pid) %>% summarise(v=mean(v,na.rm=TRUE),.groups="drop")
  dm %>% group_by(grp,pid) %>% summarise(m=mean(v),s=sd(v),n=n(),.groups="drop") %>%
    mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% pull(cv) %>% median() }

cat(sprintf("Baseline: intra-day %.1f%%   inter-day %.1f%%\n", cvmed(long), interday(long)))

# (a) Is the DAY effect a GLOBAL batch (shared across proteins) or protein-specific?
#     Remove a per-(day,rep) GLOBAL median shift shared by all proteins (batch factor),
#     on top of the existing calibration. If day var were global, inter-day CV drops.
gl <- long %>% mutate(l2=log2(v)) %>% group_by(day,rep) %>% mutate(run_off=median(l2,na.rm=TRUE)) %>% ungroup() %>%
  mutate(l2c=l2-run_off+median(l2,na.rm=TRUE), v=2^l2c)
cat(sprintf("After GLOBAL per-run median batch removal: intra %.1f%%  inter %.1f%%\n", cvmed(gl), interday(gl)))

# (b) protein-specific day effect: remove per-(pid,day) block mean shared across reps?
#     (this is the ceiling — shows how much is day-structured vs pure injection noise)
# NOTE: legitimate only as a diagnostic ceiling, not a real correction.
pd <- long %>% mutate(l2=log2(v)) %>% group_by(pid,grp,day) %>% mutate(l2c=l2-mean(l2,na.rm=TRUE)+ (function(x) mean(x))(l2)) %>% ungroup()
# intra-day CV is unaffected by removing per-day-block mean; report inter-day ceiling
pdm <- long %>% mutate(l2=log2(v)) %>% group_by(pid,grp) %>% mutate(l2c=l2-mean(l2)) # remove protein grand mean only
# multi-peptide averaging potential
long2 <- long %>% mutate(gene=str_extract(pid,"^[^_]+"))
npep <- long2 %>% distinct(gene,pid) %>% count(gene)
cat(sprintf("\nPeptides per gene: %d genes total; %.0f%% have >1 peptide (median %d, max %d)\n",
  nrow(npep), 100*mean(npep$n>1), median(npep$n), max(npep$n)))

# (c) protein-level = average peptides per gene within each sample, then CV
prot <- long2 %>% group_by(gene,grp,day,rep) %>% summarise(v=mean(v,na.rm=TRUE),.groups="drop") %>%
  rename(pid=gene)
cat(sprintf("Protein-level (avg peptides/gene): intra-day %.1f%%   inter-day %.1f%%\n", cvmed(prot), interday(prot)))

# (d) restrict to genes with >=2 peptides, compare peptide-level vs protein-level
multi <- npep %>% filter(n>=2) %>% pull(gene)
cat(sprintf("  genes with >=2 peptides (n=%d): peptide-level intra %.1f%% -> protein-level intra %.1f%%\n",
  length(multi),
  cvmed(long2 %>% filter(gene%in%multi) %>% select(pid,grp,day,rep,v)),
  cvmed(prot %>% filter(pid%in%multi))))
