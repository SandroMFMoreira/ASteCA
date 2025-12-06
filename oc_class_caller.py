# multi_streamlit_launcher.py
import subprocess
import socket
import time
import os
from pathlib import Path

APP_PATH = "oc_class_app.py"  # path to your Streamlit app script
BASE_PORT = 8520
MAX_PORT = 9999

def find_free_port(start, max_port=9999):
    port = start
    while port <= max_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free port found in range")

def ensure_csv_exists(csv_file):
    """Create the CSV file with the expected header if it doesn't exist."""
    p = Path(csv_file)
    if not p.exists():
        # create parent dir if missing
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        header = "cluster,age_classif,av_classif\n"
        p.write_text(header, encoding="utf-8")
        print(f"Created CSV file: {csv_file}")

def launch_instance(pdf_folder, csv_file, k, port=None, title=None, headless=True):
    # normalize paths
    pdf_folder = str(Path(pdf_folder))
    csv_file = str(Path(csv_file))

    if not Path(pdf_folder).exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    # create csv if missing (previous behaviour you wanted)
    ensure_csv_exists(csv_file)

    # pick a free port starting from BASE_PORT + k (so each instance tends to use a different region)
    if port is None:
        start_port = BASE_PORT
        port = start_port + k

    # build command
    cmd = [
        "streamlit", "run", APP_PATH,
        "--server.port", str(port),
        "--server.headless", "true" if headless else "false",
        "--",
        "--pdf_folder", str(Path(pdf_folder).absolute()),
        "--csv_file", str(Path(csv_file).absolute())
    ]
    if title:
        cmd += ["--title", title]

    # Launch process
    env = os.environ.copy()
    stdout_f = open(f"streamlit_{port}.out", "ab")
    stderr_f = open(f"streamlit_{port}.err", "ab")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    # small delay so the server starts binding ports
    time.sleep(5)
    return proc, port

if __name__ == "__main__":
    # ====== configure the instances you want to launch ======
    instances = [
        #{"pdf_folder": "./Sandro_parsec_UBVRI_cmd_ccd_kde/", "csv_file": "classif_parsec_Gaia_cmd_ccd.csv", "title": "Parsec"},
        {"pdf_folder": "./Dias_parsec_UBVRI_cmd_ccd_kde/", "csv_file": "classif_parsec_Gaia_cmd_ccd.csv", "title": "Parsec"},
        # {"pdf_folder": "./Sandro_baraffe_Gaia_cmd_ccd_kde/", "csv_file": "classif_baraffe_Gaia_cmd_ccd.csv", "title": "Baraffe"},
        # {"pdf_folder": "./Sandro_mixed_UBVRI_cmd_ccd_kde/", "csv_file": "classif_mixed_UBVRI_cmd_ccd.csv",
        #  "title": "Mixed"}
        #{"pdf_folder": "./Dias_baraffe_UBVRI_cmd_av_fixed/", "csv_file": "classif_baraffe_UBVRI_cmd_av_fixed.csv", "title": "Baraffe"},
        #{"pdf_folder": "./Dias_baraffe_UBVRI_cmd_av_fixed_bins_fixed/", "csv_file": "classif_baraffe_UBVRI_cmd_av_fixed_bins_fixed.csv",
        # "title": "Baraffe Bins fixed"},
        #{"pdf_folder": "Dias_baraffe_Gaia_cmd_av_fixed_bins_fixed", "csv_file": "classif_baraffe_Gaia_cmd_av_fixed_bins_fixed.csv",}
    ]
    procs = []
    try:
        for k, cfg in enumerate(instances):
            pdf_plots_folder = os.path.join(cfg["pdf_folder"], "Plots")
            csv_path = os.path.join(cfg["pdf_folder"], cfg["csv_file"])
            p, port = launch_instance(pdf_plots_folder, csv_path, k, title=cfg.get("title"), headless=True)
            print(f"Launched {cfg.get('title','app')} on http://127.0.0.1:{port}  (PID {p.pid}) — CSV: {csv_path}")
            procs.append((p, port, cfg))
        print("\nAll started. To stop them, kill the printed PIDs or press Ctrl-C here.\n")
        # wait until user interrupts
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping launched Streamlit instances...")
        for p, port, cfg in procs:
            try:
                p.terminate()
                print(f"Terminated PID {p.pid} (port {port})")
            except Exception as e:
                print("Error terminating:", e)
    except Exception as e:
        print("Launcher error:", e)
