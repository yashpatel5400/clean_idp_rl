import copy
import os
import json
import random
import rdkit
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
plt.rcParams['text.usetex'] = True
plt.rcParams["axes.grid"] = False

if __name__ == "__main__":
    method_names = ["etkdg", "rl", "diff"]
    metrics_per_method = {
        "mean_coverage_recalls": {},
        "mean_coverage_precisions": {},
    }
    for method_name in method_names:
        results_path = f"/home/yppatel/conformer-rl/examples/data/geom_metrics/{method_name}"    
        metric_fns = os.listdir(results_path)

        coverage_recalls, amr_recalls, coverage_precisions, amr_precisions = [], [], [], []
        for metric_fn in metric_fns:
            with open(os.path.join(results_path, metric_fn), "rb") as f:
                metrics = pickle.load(f)
                if metrics is None:
                    continue
                coverage_recall, amr_recall, coverage_precision, amr_precision = metrics
                
                coverage_recalls.append(coverage_recall)
                amr_recalls.append(amr_recall)
                coverage_precisions.append(coverage_precision)
                amr_precisions.append(amr_precision)
                
        thresholds = np.arange(0, 2.5, .125)
        coverage_recalls = np.array(coverage_recalls)
        coverage_precisions = np.array(coverage_precisions)

        metrics_per_method["mean_coverage_recalls"][method_name] = np.mean(coverage_recalls, axis=0)
        metrics_per_method["mean_coverage_precisions"][method_name] = np.mean(coverage_precisions, axis=0)

    print(metrics_per_method)

    for metric in metrics_per_method:
        title = " \ ".join(metric.split("_")).capitalize()
        plt.title(r"$\mathrm{" + title + r"}$")
        plt.xlabel(r"$\mathrm{Threshold} \ \delta \ (\mathrm{\AA})$")
        plt.ylabel(r"$\mathrm{Coverage \ (\%)}$")
        for method in metrics_per_method[metric]:
            plt.plot(thresholds, metrics_per_method[metric][method], label=method)
        plt.legend()
        plt.savefig(f"{metric}.png")
        plt.clf()
    
    metrics = {
        "AMR Recall": amr_recalls,
        "AMR Precision": amr_precisions,
    }

    for metric in metrics:
        print(f"{metric} (mean): {np.mean(metrics[metric])}")
        print(f"{metric} (median): {np.median(metrics[metric])}")