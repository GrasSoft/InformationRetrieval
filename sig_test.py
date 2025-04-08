import os
import sys
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Significance testing for reformulated queries.")
parser.add_argument("dataset", type=str, help="Dataset name (e.g., arguana, msmarco_passages, trec_covid)")
args = parser.parse_args()

dataset = args.dataset
sizes = ["small"] #, "medium", "large"]
output_path = f"significance_test_{dataset}_DENSE_misspelled.txt"

# --- Significance Testing ---
with open(output_path, "w") as outfile:
    for size in sizes:
        baseline_path = f"./baseline/trec-covid-misspelled_DENSE_perquery.csv"

        if not os.path.exists(baseline_path):
            outfile.write(f"\nBaseline file missing for size '{size}'\n")
            continue

        baseline = pd.read_csv(baseline_path)
        baseline['qid'] = baseline['qid'].astype(str)

        mod_dir = f"./modified/trec-covid-misspelled"
        if not os.path.exists(mod_dir):
            outfile.write(f"\nNo modified results directory for size '{size}'\n")
            continue

        outfile.write(f"\n==============================\n")
        outfile.write(f"Dataset: {dataset} | Size: {size}\n")
        outfile.write(f"==============================\n\n")

        for filename in os.listdir(mod_dir):
            if not filename.endswith("DENSE_perquery.csv"):
                continue

            outfile.write(f"\nResults for {filename}:\n")
            mod_path = os.path.join(mod_dir, filename)

            try:
                modified = pd.read_csv(mod_path)
                modified['qid'] = modified['qid'].astype(str)
            except Exception as e:
                outfile.write(f"  Failed to read {filename}: {e}\n")
                continue

            try:
                merged = baseline.merge(modified, on=["qid", "measure"], suffixes=("_base", "_mod"))
                metrics = merged["measure"].unique()

                for metric in metrics:
                    sub = merged[merged["measure"] == metric]
                    base_vals = sub["value_base"]
                    mod_vals = sub["value_mod"]

                    if len(base_vals) < 2:
                        outfile.write(f"  Skipping {metric}: not enough samples.\n")
                        continue

                    t_stat, t_p = ttest_rel(base_vals, mod_vals)
                    try:
                        w_stat, w_p = wilcoxon(base_vals, mod_vals)
                    except ValueError:
                        w_stat, w_p = None, None

                    outfile.write(f"  Metric: {metric}\n")
                    outfile.write(f"    T-test:    stat = {t_stat:.4f}, p = {t_p:.4f}\n")
                    if w_stat is not None:
                        outfile.write(f"    Wilcoxon:  stat = {w_stat:.4f}, p = {w_p:.4f}\n")
                    else:
                        outfile.write(f"    Wilcoxon:  not valid (no variation)\n")

            except Exception as e:
                outfile.write(f"  Error processing {filename}: {e}\n")

print(f"Significance testing complete. Results written to: {output_path}")

