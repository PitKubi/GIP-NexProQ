# =============================================================================
# Within-plex vs cross-plex CV — the figure that explains the reproducibility.
#   A) CV distributions: within-plex (across dbs1/2/3, same scan) vs cross-plex
#      (across replicate injections). Median-labeled.
#   B) Median CV vs protein abundance: within-plex trends DOWN (ion statistics),
#      cross-plex stays FLAT (run-to-run co-isolation interference).
# =============================================================================
suppressPackageStartupMessages({library(tidyverse); library(patchwork); library(scales)})
data_dir <- "data"; out <- "plots"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
df <- readr::read_csv(file.path(data_dir,"wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
ac <- colnames(df); mc <- ac[str_detect(ac,"^(dbs[123])_d\\d+_\\d+$") & !str_ends(ac,"SE")]
long <- bind_cols(df %>% transmute(pid=.[[1]]), df[mc]) %>%
  pivot_longer(-pid,names_to="s",values_to="v") %>%
  extract(s,c("day","rep"),"_d(\\d+)_(\\d+)$",remove=FALSE,convert=TRUE) %>%
  mutate(grp=str_extract(s,"^dbs[123]"), v=ifelse(is.finite(v)&v>0,v,NA)) %>% filter(is.finite(v))
sm <- long %>% group_by(s) %>% summarise(md=median(v),.groups="drop"); tg<-median(sm$md)
long <- long %>% left_join(sm,by="s") %>% mutate(v=v*tg/md) %>% select(-md)

theme_pub <- function(bs=16) theme_classic(base_size=bs) +
  theme(axis.title=element_text(size=bs+2,face="bold",colour="black"),
        axis.text=element_text(size=bs,colour="black"),
        axis.line=element_line(linewidth=0.6),axis.ticks=element_line(linewidth=0.6),
        plot.tag=element_text(size=bs+6,face="bold"),
        legend.position=c(0.98,0.98), legend.justification=c(1,1),
        legend.title=element_blank(), legend.text=element_text(size=bs-1),
        legend.background=element_rect(fill=alpha("white",0.8),colour=NA))
WP <- "#2171b5"; XP <- "#cb181d"

wp <- long %>% group_by(day,rep,pid) %>% summarise(m=mean(v),s=sd(v),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% mutate(kind="Within-plex\n(across DBS, same scan)")
xp <- long %>% group_by(grp,day,pid) %>% summarise(m=mean(v),s=sd(v),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv)) %>% mutate(kind="Cross-plex\n(across injections)")
both <- bind_rows(wp,xp) %>% mutate(kind=factor(kind,levels=c("Within-plex\n(across DBS, same scan)","Cross-plex\n(across injections)")))
med <- both %>% group_by(kind) %>% summarise(md=median(cv),.groups="drop")

# ---- Panel A: distributions -------------------------------------------------
cap <- 60
pA <- ggplot(filter(both,cv<=cap), aes(kind, cv, fill=kind)) +
  geom_violin(scale="width",trim=TRUE,colour="grey30",linewidth=0.3,alpha=0.9,show.legend=FALSE) +
  geom_boxplot(width=0.14,outlier.shape=NA,fill="white",linewidth=0.35,show.legend=FALSE) +
  geom_text(data=med,aes(y=md,label=sprintf("%.0f%%",md)),nudge_x=0.34,size=5.2,fontface="bold") +
  scale_fill_manual(values=c(WP,XP)) +
  scale_y_continuous(limits=c(0,cap),breaks=pretty_breaks(6),expand=expansion(mult=c(0,0.03))) +
  labs(x=NULL,y="CV (%)",tag="A") + theme_pub(16) + theme(legend.position="none")

# ---- Panel B: median CV vs abundance ---------------------------------------
bin_curve <- function(d,lab){ d %>% mutate(b=ntile(m,12)) %>% group_by(b) %>%
    summarise(x=median(m),cv=median(cv),lo=quantile(cv,.25),hi=quantile(cv,.75),.groups="drop") %>% mutate(kind=lab) }
curve <- bind_rows(bin_curve(wp,"Within-plex (across DBS, same scan)"),
                   bin_curve(xp,"Cross-plex (across injections)")) %>%
  mutate(kind=factor(kind,levels=c("Within-plex (across DBS, same scan)","Cross-plex (across injections)")))
pB <- ggplot(curve, aes(x,cv,colour=kind,fill=kind)) +
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.15,colour=NA) +
  geom_line(linewidth=1.1) + geom_point(size=2.2) +
  scale_colour_manual(values=c(WP,XP)) + scale_fill_manual(values=c(WP,XP)) +
  scale_x_log10(labels=label_number(drop0trailing=TRUE)) + annotation_logticks(sides="b") +
  scale_y_continuous(limits=c(0,NA),breaks=pretty_breaks(6),expand=expansion(mult=c(0,0.05))) +
  labs(x="Protein abundance (concentration, log scale)",y="Median CV (%)",tag="B") +
  theme_pub(16)

fig <- pA + pB + plot_layout(widths=c(1,1.35))
ggsave(file.path(out,"Fig_within_vs_cross_plex.png"), fig, width=14, height=6, dpi=600, bg="white")
ggsave(file.path(out,"Fig_within_vs_cross_plex.pdf"), fig, width=14, height=6)
message("saved Fig_within_vs_cross_plex")
