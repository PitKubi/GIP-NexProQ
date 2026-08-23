# =============================================================================
# MAIN FIGURE — TMT reproducibility: within-plex vs cross-plex
#   A) CV distributions for the 3 reproducibility axes (within-plex, cross-plex
#      intra-day, cross-plex inter-day), medians labelled.
#   B) Median CV vs protein abundance — within-plex falls (ion statistics),
#      cross-plex stays flat (co-isolation interference). The reviewer answer.
#   C) Variance partition — where the total variance actually lives.
# =============================================================================
suppressPackageStartupMessages({library(tidyverse); library(patchwork); library(scales)})
proj <- "/home/pk/R projects/Science/NexProQ"; out <- file.path(proj,"plots")
df <- readr::read_csv(file.path(proj,"data","wide_table_with_bloq.csv"), guess_max=1e6, show_col_types=FALSE)
ac <- colnames(df); mc <- ac[str_detect(ac,"^(dbs[123])_d\\d+_\\d+$") & !str_ends(ac,"SE")]
long <- bind_cols(df %>% transmute(pid=.[[1]]), df[mc]) %>%
  pivot_longer(-pid,names_to="s",values_to="v") %>%
  extract(s,c("day","rep"),"_d(\\d+)_(\\d+)$",remove=FALSE,convert=TRUE) %>%
  mutate(grp=str_extract(s,"^dbs[123]"), v=ifelse(is.finite(v)&v>0,v,NA)) %>% filter(is.finite(v))
sm <- long %>% group_by(s) %>% summarise(md=median(v),.groups="drop"); tg<-median(sm$md)
long <- long %>% left_join(sm,by="s") %>% mutate(v=v*tg/md) %>% select(-md)

theme_pub <- function(bs=15) theme_classic(base_size=bs) +
  theme(axis.title=element_text(size=bs+2,face="bold",colour="black"),
        axis.text=element_text(size=bs,colour="black"),
        axis.line=element_line(linewidth=0.6),axis.ticks=element_line(linewidth=0.6),
        plot.tag=element_text(size=bs+7,face="bold"),
        legend.text=element_text(size=bs-1), legend.title=element_blank(),
        legend.background=element_rect(fill=alpha("white",0.75),colour=NA))
WP<-"#2171b5"; XI<-"#fb6a4a"; XD<-"#a50f15"

cvtab <- function(d,...) d %>% group_by(...,pid) %>%
  summarise(m=mean(v),s=sd(v),n=sum(is.finite(v)),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv))
wp <- cvtab(long, day, rep)                                   # within-plex (across DBS)
xi <- cvtab(long, grp, day)                                   # cross-plex intra-day (across reps)
dm <- long %>% group_by(grp,day,pid) %>% summarise(v=mean(v),.groups="drop")
xd <- dm %>% group_by(grp,pid) %>% summarise(m=mean(v),s=sd(v),n=n(),.groups="drop") %>%
  mutate(cv=ifelse(m>0&n>=2,100*s/m,NA)) %>% filter(is.finite(cv))               # cross-plex inter-day

L <- c("Within-plex\n(across DBS,\nsame scan)","Cross-plex\nintra-day\n(replicate inj.)","Cross-plex\ninter-day\n(across days)")
both <- bind_rows(mutate(wp,k=L[1]), mutate(xi,k=L[2]), mutate(xd,k=L[3])) %>% mutate(k=factor(k,levels=L))
med <- both %>% group_by(k) %>% summarise(md=median(cv),.groups="drop")

# ---- A: distributions -------------------------------------------------------
cap<-60
pA <- ggplot(filter(both,cv<=cap), aes(k,cv,fill=k)) +
  geom_violin(scale="width",trim=TRUE,colour="grey30",linewidth=0.3,alpha=0.9) +
  geom_boxplot(width=0.13,outlier.shape=NA,fill="white",linewidth=0.35) +
  geom_text(data=med,aes(y=md,label=sprintf("%.0f%%",md)),nudge_x=0.36,size=5,fontface="bold") +
  scale_fill_manual(values=c(WP,XI,XD),guide="none") +
  scale_y_continuous(limits=c(0,cap),breaks=pretty_breaks(6),expand=expansion(mult=c(0,0.03))) +
  labs(x=NULL,y="CV (%)",tag="A") + theme_pub(15)

# ---- B: CV vs abundance -----------------------------------------------------
binc <- function(d,lab) d %>% mutate(b=ntile(m,12)) %>% group_by(b) %>%
  summarise(x=median(m),cv=median(cv),lo=quantile(cv,.25),hi=quantile(cv,.75),.groups="drop") %>% mutate(k=lab)
cur <- bind_rows(binc(wp,"Within-plex (same scan)"), binc(xi,"Cross-plex (across injections)")) %>%
  mutate(k=factor(k,levels=c("Within-plex (same scan)","Cross-plex (across injections)")))
pB <- ggplot(cur,aes(x,cv,colour=k,fill=k)) +
  geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.15,colour=NA) +
  geom_line(linewidth=1.1)+geom_point(size=2.2) +
  scale_colour_manual(values=c(WP,XI))+scale_fill_manual(values=c(WP,XI)) +
  scale_x_log10(labels=label_number(drop0trailing=TRUE))+annotation_logticks(sides="b") +
  scale_y_continuous(limits=c(0,NA),breaks=pretty_breaks(6),expand=expansion(mult=c(0,0.05))) +
  labs(x="Protein abundance (conc., log scale)",y="Median CV (%)",tag="B") +
  theme_pub(15)+theme(legend.position=c(0.97,0.05),legend.justification=c(1,0))

# ---- C: variance partition --------------------------------------------------
vc <- long %>% mutate(l2=log2(v)) %>% group_by(pid) %>% filter(n()>=8) %>%
  summarise(v_total=var(l2),
            v_rep={a<-aggregate(l2~rep,cur_data(),mean);var(a$l2)},
            v_day={a<-aggregate(l2~day,cur_data(),mean);var(a$l2)},
            v_dbs={a<-aggregate(l2~grp,cur_data(),mean);var(a$l2)},.groups="drop")
vt<-median(vc$v_total,na.rm=TRUE)
part <- tibble(comp=factor(c("Day\n(cross-plex)","Replicate inj.\n(cross-plex)","DBS sample\n(within-plex)","Residual\n(injection noise)"),
                           levels=c("Residual\n(injection noise)","DBS sample\n(within-plex)","Replicate inj.\n(cross-plex)","Day\n(cross-plex)")),
               pct=c(100*median(vc$v_day,na.rm=TRUE)/vt, 100*median(vc$v_rep,na.rm=TRUE)/vt,
                     100*median(vc$v_dbs,na.rm=TRUE)/vt, NA))
part$pct[4] <- 100-sum(part$pct[1:3])
pC <- ggplot(part,aes(pct,comp,fill=comp)) + geom_col(width=0.7,colour="grey25",linewidth=0.3) +
  geom_text(aes(label=sprintf("%.0f%%",pct)),hjust=-0.15,size=4.6,fontface="bold") +
  scale_fill_manual(values=c("grey70","#2171b5","#fb6a4a","#a50f15"),guide="none") +
  scale_x_continuous(limits=c(0,72),breaks=seq(0,60,20),expand=expansion(mult=c(0,0.02))) +
  labs(x="Share of total variance (%)",y=NULL,tag="C") + theme_pub(15)

fig <- (pA | pB) / pC + plot_layout(heights=c(1.35,1))
ggsave(file.path(out,"Fig_TMT_reproducibility_main.png"), fig, width=14, height=11, dpi=600, bg="white")
ggsave(file.path(out,"Fig_TMT_reproducibility_main.pdf"), fig, width=14, height=11)
message("saved Fig_TMT_reproducibility_main")
