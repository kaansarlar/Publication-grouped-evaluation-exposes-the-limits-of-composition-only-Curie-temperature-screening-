#!/usr/bin/env python3
"""Create manuscript Figures 6 and 7 from the frozen rule-analysis outputs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="0.91",linewidth=.7,zorder=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-dir",type=Path,default=ROOT/"results/rule_reassessment")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reproduced/figures")
    args=parser.parse_args()
    source=args.input_dir
    args.output_dir.mkdir(parents=True,exist_ok=True)

    shap=pd.read_csv(source/"shap_rank_stability.csv")
    repeats=pd.read_csv(source/"repeat_level_metrics.csv")
    external=pd.read_csv(source/"external_v1_v2_metrics.csv")
    selections=pd.read_csv(source/"outer_fold_rule_selections.csv")

    fig,axes=plt.subplots(2,2,figsize=(11.2,8.2))
    s=shap.sort_values("mean_rank",ascending=False)
    axes[0,0].barh(np.arange(len(s)),s["mean_rank"],xerr=s["sd_rank"],color="#4378A8",alpha=.9,capsize=2)
    axes[0,0].set_yticks(np.arange(len(s)),s["feature"].str.replace("_"," "),fontsize=8)
    axes[0,0].invert_xaxis(); axes[0,0].set_xlabel("Mean SHAP rank (1 = highest)")
    axes[0,0].set_title("a  Publication-grouped SHAP stability",loc="left",fontsize=10)
    style(axes[0,0])

    order=["v1_rederived","v2_nested","v2_nested_legacy_f1"]
    labels=["Fixed-family\nbaseline","Nested\nMCC-first","Nested\nF1-first"]
    metric_names=["balanced_accuracy","MCC","recall","specificity"]
    colors=["#808080","#2E6F9E","#4C8C6B","#C47A2C"]
    x=np.arange(3); width=.19
    for j,(metric,color) in enumerate(zip(metric_names,colors)):
        means=[]; stds=[]
        for rule in order:
            v=repeats.loc[repeats.rule.eq(rule),metric]
            means.append(v.mean()); stds.append(v.std())
        axes[0,1].bar(x+(j-1.5)*width,means,width,yerr=stds,label=metric.replace("balanced_accuracy","balanced accuracy"),color=color,capsize=2,zorder=3)
    axes[0,1].set_xticks(x,labels); axes[0,1].set_ylim(0,1); axes[0,1].legend(frameon=False,fontsize=7,ncol=2)
    axes[0,1].set_title("b  Repeated nested development performance",loc="left",fontsize=10)
    style(axes[0,1])

    ext_names=["Historical 32 records, v1 locked","Historical 32 records, v2 frozen","Targeted five-paper primary challenge, v1","Targeted five-paper primary challenge, v2"]
    e=external.set_index("set").loc[ext_names]
    xx=np.arange(4)
    axes[1,0].bar(xx-.18,e["balanced_accuracy"],.36,color="#2E6F9E",label="Balanced accuracy")
    axes[1,0].bar(xx+.18,e["MCC"],.36,color="#C47A2C",label="MCC")
    axes[1,0].axhline(0,color="0.35",lw=.8)
    axes[1,0].set_xticks(xx,["Historical\nbaseline","Historical\ncandidate","Stress test\nbaseline","Stress test\ncandidate"])
    axes[1,0].set_ylim(-.3,1); axes[1,0].legend(frameon=False,fontsize=8)
    axes[1,0].set_title("c  Structured literature comparisons",loc="left",fontsize=10)
    style(axes[1,0])

    f1sel=selections[selections.objective.eq("legacy_f1")].copy()
    short={"mag_moment_mean":"magmom","avg_d_valence_electrons":"d-valence","melting_point_mean":"Tm mean",
           "electronegativity_range":"χ range","dHmix":"ΔHmix","dSmix":"ΔSmix","sigma":"δ"}
    f1sel["selection"]=f1sel.apply(lambda r:" + ".join(short.get(x,x) for x in r["features"].split(";"))+f" | P{r['q_low']:g}–{r['q_high']:g}",axis=1)
    counts=f1sel["selection"].value_counts().head(6).sort_values()
    axes[1,1].barh(np.arange(len(counts)),counts.values,color="#6E9F75")
    axes[1,1].set_yticks(np.arange(len(counts)),counts.index,fontsize=7)
    axes[1,1].set_xlabel("Selections across 100 outer folds")
    axes[1,1].set_title("d  Selected rule instability",loc="left",fontsize=10)
    style(axes[1,1])
    fig.tight_layout()
    fig.savefig(args.output_dir/"Fig6_nested_rule_validation.png",dpi=350,bbox_inches="tight")
    fig.savefig(args.output_dir/"Fig6_nested_rule_validation.pdf",bbox_inches="tight")
    plt.close(fig)

    old=pd.read_csv(source/"historical_32_v1_v2_predictions.csv")
    new=pd.read_csv(source/"targeted_challenge_v1_v2_predictions.csv")
    primary=new[new.cohort.eq("primary")].copy().sort_values("TC_K",na_position="last")
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    plot=old.sort_values("TC_K").reset_index(drop=True); x=np.arange(len(plot))
    axes[0].scatter(x,plot.TC_K,c=plot["pass::v2_250_350"].map({True:"#2f7d5a",False:"#b24a48"}),s=36)
    axes[0].axhspan(250,350,color="#e8c66a",alpha=.2); axes[0].set_xlabel("Historical audit record")
    axes[0].set_ylabel("Reported transition temperature (K)"); axes[0].set_title("a  Full-development candidate: historical audit",loc="left",fontsize=10); style(axes[0])
    x=np.arange(len(primary)); y=primary.TC_K.fillna(370)
    axes[1].scatter(x,y,c=primary["pass::v2_250_350"].map({True:"#2f7d5a",False:"#b24a48"}),s=46)
    axes[1].axhspan(250,350,color="#e8c66a",alpha=.2); axes[1].set_xticks(x,primary.record_id,rotation=70,ha="right",fontsize=7)
    axes[1].set_ylabel("Reported transition temperature (K)"); axes[1].set_title("b  Full-development candidate: targeted stress test",loc="left",fontsize=10); style(axes[1])
    fig.tight_layout(); fig.savefig(args.output_dir/"Fig7_candidate_rule_outcomes.png",dpi=350,bbox_inches="tight"); fig.savefig(args.output_dir/"Fig7_candidate_rule_outcomes.pdf",bbox_inches="tight"); plt.close(fig)

if __name__=="__main__": main()
