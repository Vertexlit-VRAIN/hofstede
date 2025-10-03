
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_csv", default=str(Path(__file__).resolve().parents[1]/"data"/"example_models_results.csv"))
    ap.add_argument("--hofstede_csv", default=str(Path(__file__).resolve().parents[1]/"data"/"hofstede_reference.csv"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parents[1]/"outputs"))
    ap.add_argument("--figures_dir", default=str(Path(__file__).resolve().parents[1]/"figures"))
    ap.add_argument("--weights_json", default="")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    compute = base/"src"/"compute_metrics.py"
    plot = base/"src"/"plot_figures.py"
    desc = base/"src"/"stats_descriptives.py"
    anova = base/"src"/"anova_regression.py"
    maps = base/"src"/"make_maps.py"

    cmd = [sys.executable, str(compute), "--models_csv", args.models_csv, "--hofstede_csv", args.hofstede_csv, "--out_dir", args.out_dir]
    if args.weights_json:
        cmd += ["--weights_json", args.weights_json]
    run(cmd)

    run([sys.executable, str(desc), "--models_csv", args.models_csv, "--out_dir", args.out_dir])

    run([sys.executable, str(plot),
         "--model_profiles_csv", str(Path(args.out_dir)/"model_profiles_6D.csv"),
         "--metrics_csv", str(Path(args.out_dir)/"metrics_per_model_country.csv"),
         "--deltas_csv", str(Path(args.out_dir)/"per_dimension_deltas.csv"),
         "--hofstede_csv", args.hofstede_csv,
         "--out_dir", args.figures_dir])

    run([sys.executable, str(anova),
         "--metrics_csv", str(Path(args.out_dir)/"metrics_per_model_country.csv"),
         "--deltas_csv", str(Path(args.out_dir)/"per_dimension_deltas.csv"),
         "--out_dir", args.out_dir])

    run([sys.executable, str(maps),
         "--profiles_csv", str(Path(args.out_dir)/"model_profiles_6D.csv"),
         "--hofstede_csv", args.hofstede_csv,
         "--out_dir", str(Path(args.figures_dir)/"maps")])

    print("Done.")
