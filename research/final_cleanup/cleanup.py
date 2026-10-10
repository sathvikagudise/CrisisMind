"""Guarded cleanup: removes runtime junk + old-era experiments, protects final
research/source artifacts. Use `python cleanup.py` (dry-run) or `... apply`.

Nothing in PROTECT is ever touched. Deletions are logged to CLEANUP_DELETED.log.
"""
import os, sys, shutil, json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APPLY = "apply" in sys.argv

# ---- PROTECTED (never delete anything under these) --------------------------
PROTECT_DIRS = [
    "research/final_experiment",
    "backend/outputs/checkpoints/final_exp",
    "research/data_cleaning",
    "research/paper_assets",
    "research/Conference",
    "research/references",
    "research/assets",
    "project_audit",
    "testing",
    "dataset_analysis/CrisisMMD_v2.0",
    "dataset_analysis/CrisisMMD_v2.0_clean",
    "dataset_analysis/reports",
    "backend/analysis_engine", "backend/analysis_center", "backend/api",
    "backend/batch_processing", "backend/config", "backend/dashboard",
    "backend/datasets", "backend/disaster_intelligence", "backend/explainability",
    "backend/models", "backend/reporting", "backend/tests", "backend/training",
    "frontend/src", "frontend/public",
]
PROTECT_EXTS = {".py"}
PROTECT_FILES = {
    "research/README.md", "research/IEEE_ASSET_CHECKLIST.md",
    "research/00_Project_Overview.md", "research/01_Dataset_Analysis.md",
    "research/02_Image_Baseline.md", "research/03_Text_Baseline.md",
    "research/04_Multimodal_Fusion.md", "research/05_Experimental_Results.md",
    "research/06_Project_Architecture.md", "research/07_Research_Log.md",
    "research/08_Novelty_and_Contribution.md",
    "run_fusion.py", "run_image_baseline.py", "run_text_baseline.py",
    "run_text_eval.py", "dataset_audit.py",
    "crisismindsample1.jpg", "sample image3 .jpg",
    "backend/outputs/checkpoints/best_model.pth",
    "backend/outputs/checkpoints/best_text_model.pth",
    "backend/outputs/checkpoints/best_gated_fusion_v3.pth",
    "backend/outputs/checkpoints/best_concat_fusion.pth",
}

def is_protected(rel):
    rel = rel.replace("\\", "/")
    for p in PROTECT_DIRS:
        if rel == p or rel.startswith(p + "/"):
            return True
    if rel in PROTECT_FILES:
        return True
    return False

# ---- DELETE TARGETS ---------------------------------------------------------
DELETE_DIRS = [
    "research/optimization/P2_baseline", "research/optimization/P3_Focal_g0.5",
    "research/optimization/P3_Focal_g1.0", "research/optimization/P3_Focal_g2.0",
    "research/optimization/P3_WCE_g0.5", "research/optimization/P3_WCE_g1.0",
    "research/optimization/P3_WCE_g2.0", "research/optimization/P4_augmented",
    "research/optimization/P5_A_frozen", "research/optimization/P5_B_effnet",
    "research/optimization/P5_C_distilbert", "research/optimization/P5_D_both",
    "research/optimization/P7_e2e",
    "backend/outputs/checkpoints/v2", "backend/outputs/checkpoints/v3",
    "backend/outputs/checkpoints/v3_clean",
    "experiments",
    "dataset_analysis/CrisisMMD_v2.0_original",
    "testing/.venv",
    "frontend/dist",
    ".pytest_cache", ".freebuff",
    "research/final_cleanup/__pycache__",
]

DELETE_FILES = [
    "backend.log", "frontend.log", "test_server.log", "NUL",
    "FINAL_RELEASE_REPORT.md", "FINAL_VALIDATION_REPORT.md",
    "MODEL_AUDIT_REPORT.md", "MODEL_IMPROVEMENT_REPORT.md",
    "ROOT_CAUSE_ANALYSIS.md", "dataset_summary_report.md",
    "backend/backend.log", "frontend/frontend.log", "frontend/tsconfig.tsbuildinfo",
    "backend/outputs/checkpoints/best_gated_fusion.pth",
    "backend/outputs/checkpoints/best_gated_fusion_v2.pth",
    "research/optimization/FINAL_REPORT.json", "research/optimization/P6_grid_results.csv",
    "research/optimization/P7_e2e_finetune.py", "research/optimization/run_all_phases.py",
    "research/optimization/run_remaining.py", "research/optimization/features_frozen.pt",
    "research/final_experiment/img.pid",
    "dataset_analysis/CrisisMMD_v2.0/crisismmd_datasplit_all.zip",
    # final_cleanup scratch (keep .md reports + deployment evidence)
    "research/final_cleanup/scan_repo.py", "research/final_cleanup/scan_detail.py",
    "research/final_cleanup/scan_detail2.py", "research/final_cleanup/classify_repo.py",
    "research/final_cleanup/audit_report_gen.py",
    "research/final_cleanup/BACKEND_TREE.txt", "research/final_cleanup/DATASET_TREE.txt",
    "research/final_cleanup/EXPERIMENTS_TREE.txt", "research/final_cleanup/FRONTEND_TREE.txt",
    "research/final_cleanup/PROJECT_AUDIT_TREE.txt", "research/final_cleanup/RESEARCH_TREE.txt",
    "research/final_cleanup/TESTING_TREE.txt",
    "research/final_cleanup/FILE_INVENTORY.csv", "research/final_cleanup/FILE_CLASSIFICATION.csv",
    "research/final_cleanup/_demo_launch.py", "research/final_cleanup/_demo_backend.log",
    "research/final_cleanup/_demo_frontend.log", "research/final_cleanup/_deploy_backend.log",
]

CLEAR_CONTENTS = [
    "backend/outputs/explanations", "backend/outputs/cached_features",
    "backend/outputs/plots", "backend/outputs/metrics", "backend/outputs/reports",
]

# ---- GLOBAL PATTERN JUNK ----------------------------------------------------
def _rel(full):
    try:
        return os.path.relpath(full, ROOT).replace("\\", "/")
    except ValueError:
        return None


def find_junk():
    junk = []
    for r, d, f in os.walk(ROOT):
        rel_r = _rel(r)
        if rel_r is None:
            d[:] = []
            continue
        if rel_r.startswith(("frontend/node_modules", "backend/.venv", "testing/.venv",
                             "frontend/dist")):
            d[:] = []
            continue
        for x in f:
            full = os.path.join(r, x)
            rel = _rel(full)
            if rel is None or is_protected(rel):
                continue
            if (x.endswith((".pyc", ".log", ".pid")) or x == ".DS_Store"
                    or x.startswith("._") or x.endswith(".tsbuildinfo")):
                junk.append(full)
        for dd in list(d):
            full = os.path.join(r, dd)
            rel = _rel(full)
            if dd == "__pycache__" and rel is not None and not is_protected(rel):
                junk.append(full)
                d.remove(dd)
    return junk


def size_of(p):
    if os.path.isfile(p):
        try: return os.path.getsize(p)
        except OSError: return 0
    t = 0
    for r, _, f in os.walk(p):
        for x in f:
            try: t += os.path.getsize(os.path.join(r, x))
            except OSError: pass
    return t


def main():
    plan = []
    for d in DELETE_DIRS:
        p = os.path.join(ROOT, d)
        if os.path.exists(p) and not is_protected(d):
            plan.append((p, size_of(p), "dir"))
    for f in DELETE_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p) and not is_protected(f):
            plan.append((p, size_of(p), "file"))
    for c in CLEAR_CONTENTS:
        p = os.path.join(ROOT, c)
        if os.path.isdir(p):
            for e in os.listdir(p):
                fp = os.path.join(p, e)
                if not is_protected(os.path.relpath(fp, ROOT).replace("\\", "/")):
                    plan.append((fp, size_of(fp), "file|dir"))
    for p in find_junk():
        plan.append((p, size_of(p), "junk"))

    # dedupe (junk walk may repeat clear_contents entries)
    seen = set()
    uniq = []
    for p, s, k in plan:
        if p in seen: continue
        seen.add(p); uniq.append((p, s, k))

    total = sum(s for _, s, _ in uniq)
    print(f"{'APPLY' if APPLY else 'DRY-RUN'}: {len(uniq)} items, "
          f"{round(total/1048576,1)} MB total\n")
    for p, s, k in sorted(uniq):
        rel = _rel(p) or os.path.basename(p)
        print(f"  [{k:8s}] {round(s/1048576,2):>9.2f} MB  {rel}")

    if not APPLY:
        print("\n(dry-run — nothing deleted). Re-run with `apply`.")
        return

    log = []
    for p, s, k in sorted(uniq, key=lambda t: -len(t[0])):
        try:
            if os.path.basename(p) == "NUL":
                os.remove("\\\\?\\" + os.path.abspath(p))
            elif os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            log.append({"path": _rel(p) or os.path.basename(p), "bytes": s, "type": k})
        except Exception as e:
            print("  FAILED:", p, e)
    with open(os.path.join(ROOT, "research", "final_cleanup", "CLEANUP_DELETED.log"),
              "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"\nDELETED {len(log)} items, {round(sum(l['bytes'] for l in log)/1048576,1)} MB")


if __name__ == "__main__":
    main()