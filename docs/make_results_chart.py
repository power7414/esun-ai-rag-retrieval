import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cats=["FAQ","Insurance","Finance","Overall"]
base=[90,80,44,71.33]; mid=[96,90,70,85.33]; final=[96,90,80,88.67]
fig,ax=plt.subplots(figsize=(10,5.2),dpi=160)
x=range(len(cats)); w=0.26
cols=["#9aa3ad","#7fb8a4","#1f7a5a"]
labs=["Organizer baseline (BM25)","Unified pipeline, chunk hybrid","Unified pipeline, final (dict hybrid)"]
for i,(vals,c,l) in enumerate(zip([base,mid,final],cols,labs)):
    xs=[k+(i-1)*w for k in x]
    b=ax.bar(xs,vals,w,color=c,label=l)
    for xx,v in zip(xs,vals):
        ax.text(xx,v+1.2,f"{v:g}",ha="center",va="bottom",fontsize=9,color="#222")
ax.set_xticks(list(x)); ax.set_xticklabels(cats,fontsize=11)
ax.set_ylim(0,108); ax.set_ylabel("Precision@1 (%)")
ax.set_title("E.SUN RAG retrieval — measured on the official 150-question sample (2026-11-07)",fontsize=12,loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.yaxis.grid(True,color="#e6e6e6"); ax.set_axisbelow(True)
ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,-0.08),ncol=3,fontsize=9)
fig.tight_layout(); fig.savefig("results.png",facecolor="white")
