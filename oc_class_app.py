import os
import streamlit as st
import pandas as pd
import base64
import argparse
import sys

# parse args passed after '--' in the `streamlit run` command
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--pdf_folder", dest="pdf_folder", required=True)
parser.add_argument("--csv_file", dest="csv_file", required=True)
parser.add_argument("--title", dest="title", default="OC Classifier")
# parse_known_args is safer because Streamlit adds its own args
args, _ = parser.parse_known_args()
PDF_FOLDER = args.pdf_folder
CSV_FILE = args.csv_file
APP_TITLE = args.title

# then later use APP_TITLE for the page title/header, e.g.
st.set_page_config(layout="wide", page_title=APP_TITLE)
st.title(APP_TITLE)

CLASS_KEYS = ["A", "B", "C"]

# === PAGE CONFIG ===
st.set_page_config(layout="wide", page_title="OC Classifier")

# === Helpers to load/create dataframe ===
def load_or_init_df(csv_path):
    if os.path.exists(csv_path):
        try:
            df_load = pd.read_csv(csv_path)
            # ensure required columns exist
            if not {"cluster", "age_classif", "av_classif"}.issubset(df_load.columns):
                # create clean structure keeping 'cluster' if present
                if "cluster" in df_load.columns:
                    df = pd.DataFrame({"cluster": df_load["cluster"].astype(str).tolist(),
                                       "age_classif": [None]*len(df_load),
                                       "av_classif": [None]*len(df_load)})
                else:
                    df = pd.DataFrame(columns=["cluster", "age_classif", "av_classif"])
            else:
                df = df_load[["cluster", "age_classif", "av_classif"]].copy()
        except Exception:
            df = pd.DataFrame(columns=["cluster", "age_classif", "av_classif"])
    else:
        df = pd.DataFrame(columns=["cluster", "age_classif", "av_classif"])
    return df

def persist_df(df, path=CSV_FILE):
    df.to_csv(path, index=False)

# === Initialize session_state df ===
if "df" not in st.session_state:
    st.session_state["df"] = load_or_init_df(CSV_FILE)
    st.session_state["last_saved"] = ""

# === Collect PDFs (alphabetical) ===
pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")])
clusters = [os.path.splitext(f)[0] for f in pdf_files]

# ensure all clusters exist in df
df = st.session_state["df"]
for c in clusters:
    if c not in df["cluster"].values:
        df = pd.concat([df, pd.DataFrame([[c, None, None]], columns=["cluster", "age_classif", "av_classif"])], ignore_index=True)
# keep df ordered by cluster to stay consistent
df = df.drop_duplicates(subset=["cluster"], keep="first").reset_index(drop=True)
st.session_state["df"] = df

# === Navigation state ===
if "cluster_idx" not in st.session_state:
    st.session_state["cluster_idx"] = 0

# Top navigation row with unique labels (exact match in JS)
col_nav_left, col_nav_center, col_nav_right = st.columns([1, 8, 1])
with col_nav_left:
    if st.button("NAV_PREV_BTN"):
        if st.session_state.cluster_idx > 0:
            st.session_state.cluster_idx -= 1
with col_nav_center:
    if clusters:
        st.markdown(f"**Cluster {st.session_state.cluster_idx + 1} / {len(clusters)}**  —  **{clusters[st.session_state.cluster_idx]}**")
    else:
        st.markdown("**No PDFs found**")
with col_nav_right:
    if st.button("NAV_NEXT_BTN"):
        if st.session_state.cluster_idx < len(clusters) - 1:
            st.session_state.cluster_idx += 1

# if no clusters present, show message and exit
if not clusters:
    st.warning(f"No PDF files found in `{PDF_FOLDER}`. Place files named `cluster_name.pdf` in that folder.")
    st.stop()

# current cluster and pdf path
cluster_idx = st.session_state.cluster_idx
cluster = clusters[cluster_idx]
pdf_path = os.path.join(PDF_FOLDER, f"{cluster}.pdf")

# === Layout: PDF left, controls right ===
col1, col2 = st.columns([7, 1])  # big left column for PDF

with col1:
    st.subheader(f"{cluster}  (index {cluster_idx})")
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1200" '
            f'type="application/pdf"></iframe>'
        )
        st.components.v1.html(pdf_display, height=800, width=1400)
    except FileNotFoundError:
        st.error(f"PDF not found: {pdf_path}")

with col2:
    st.subheader("Classifications")

    # show current stored classifications
    df = st.session_state["df"]
    cur_row = df.loc[df["cluster"] == cluster]
    if len(cur_row) == 0:
        cur_age = None
        cur_av = None
    else:
        cur_age = cur_row.iloc[0]["age_classif"]
        cur_av = cur_row.iloc[0]["av_classif"]

    st.markdown("**Age classification**")
    # Age buttons: Age: A / B / C
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Age: A"):
            # immediate save
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "age_classif"] = "A"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — age_classif: A"
            # refresh cur_* variables for display
            cur_age = "A"
    with c2:
        if st.button("Age: B"):
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "age_classif"] = "B"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — age_classif: B"
            cur_age = "B"
    with c3:

        if st.button("Age: C"):
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "age_classif"] = "C"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — age_classif: C"
            cur_age = "C"

    st.markdown(f"Current age_classif: **{cur_age if cur_age is not None else '—'}**")
    st.markdown("---")

    st.markdown("**AV classification**")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("AV: 1"):
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "av_classif"] = "1"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — av_classif: 1"
            cur_av = "1"
    with a2:
        if st.button("AV: 2"):
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "av_classif"] = "2"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — av_classif: 2"
            cur_av = "2"
    with a3:
        if st.button("AV: 3"):
            st.session_state["df"].loc[st.session_state["df"]["cluster"] == cluster, "av_classif"] = "3"
            persist_df(st.session_state["df"], CSV_FILE)
            st.session_state["last_saved"] = f"Saved {cluster} — av_classif: 3"
            cur_av = "3"

    st.markdown(f"Current av_classif: **{cur_av if cur_av is not None else '—'}**")

    st.markdown("---")
    # small informative text
    st.markdown("Use Left / Right arrow keys to navigate clusters (page must be focused).")
    st.caption("Click anywhere on the page (or PDF) once to give the browser focus, then use ← / →.")
    # show last saved message
    if st.session_state.get("last_saved"):
        st.success(st.session_state["last_saved"])

# === JS: exact-match nav buttons for keyboard arrows (no fuzzy matching) ===
js = """
<script>
(function(){
  function findNavButtonsExact() {
    const parentDoc = window.parent.document;
    const buttons = parentDoc.querySelectorAll('button');
    let prev = null;
    let next = null;
    buttons.forEach(b=>{
      const txt = (b.innerText || b.textContent || "").trim();
      if (txt === "NAV_PREV_BTN") prev = b;
      if (txt === "NAV_NEXT_BTN") next = b;
    });
    return {prev, next};
  }

  let wired = false;
  const poll = setInterval(()=>{
    const {prev, next} = findNavButtonsExact();
    if ((prev || next) && !wired) {
      wired = true;
      window.parent.window.addEventListener('keydown', function(e){
        // ignore if focus is on an input/textarea or a content editable element
        const active = document.activeElement;
        const tag = active && active.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || (active && active.isContentEditable)) return;
        if (e.key === 'ArrowLeft') {
          if (prev) prev.click();
        } else if (e.key === 'ArrowRight') {
          if (next) next.click();
        }
      }, false);
      clearInterval(poll);
    }
  }, 200);
})();
</script>
"""
# invisible component
st.components.v1.html(js, height=1)
