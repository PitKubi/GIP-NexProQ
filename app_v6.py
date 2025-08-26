import streamlit as st
import pandas as pd
import io
import img2pdf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.linear_model import TheilSenRegressor



# Helper function to abbreviate sample names for plotting
def abbreviate_name(name, max_length=15):
    return name if len(name) <= max_length else name[:max_length] + "..."


# --- Start Over Button ---
if st.sidebar.button("Start Over"):
    # Preserve the page_menu key and force it to "Generate Template"
    keys_to_keep = ["page_menu"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state["page_menu"] = "Generate Template"
    st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)


# --- Persistent State Initialization ---
if "template_input_data" not in st.session_state:
    st.session_state["template_input_data"] = None
if "process_input_data" not in st.session_state:
    st.session_state["process_input_data"] = None
if "process_template_data" not in st.session_state:
    st.session_state["process_template_data"] = None
if "edited_template" not in st.session_state:
    st.session_state["edited_template"] = None
if "sample_names" not in st.session_state:
    st.session_state["sample_names"] = {}
if "experiment_group_mapping" not in st.session_state:
    st.session_state["experiment_group_mapping"] = {}

# --- Page Selection ---
menu = st.sidebar.radio("Select Operation",
                        ["Generate Template", "Calculate Concentrations", "Data Refinement"],
                        key="page_menu")

st.title("GIP - NexProQ: Data extraction and analysis pipeline")

# ------------------
# Generate Template
# ------------------
if menu == "Generate Template":
    st.header("Generate Template")
    if st.session_state.get("template_committed", False):
        st.write("Committed Template:")
        st.dataframe(st.session_state["edited_template"])
        st.download_button("Download Template CSV",
                           data=st.session_state["edited_template"].to_csv(index=False).encode('utf-8'),
                           file_name="template.csv",
                           mime="text/csv")
        st.info("The template is already committed. To modify it, please click 'Start Over'.")
    else:
        uploaded_file = st.file_uploader("Upload Input CSV", type=["csv"], key="template_input")
        if uploaded_file is not None:
            st.session_state["template_input_data"] = uploaded_file.getvalue()
            try:
                df = pd.read_csv(io.BytesIO(st.session_state["template_input_data"]))
            except Exception as e:
                st.error(f"Error reading the file: {e}")
            else:
                # Always use the large cohort template
                df['Fragment Ion'] = df['Fragment Ion'].astype(str)
                # Only keep rows where 'Fragment Ion' contains "TMT"
                df_tmt = df[df['Fragment Ion'].str.contains("TMT", na=False)]
                # Get unique rows based on 'Replicate' and 'Fragment Ion'
                template_df = df_tmt[['Replicate', 'Fragment Ion']].drop_duplicates().copy()
                # Rename columns:
                #   use the original 'Replicate' as 'Original Sample'
                #   rename 'Fragment Ion' to 'TMT Channel'
                template_df = template_df.rename(
                    columns={'Replicate': 'Original Sample', 'Fragment Ion': 'TMT Channel'})
                # Add empty columns for user editing: 'Sample Name', 'Replicate', and 'conc'
                template_df["Sample Name"] = ""
                template_df["Replicate"] = ""
                template_df["conc"] = ""
                # Reorder the columns as specified
                template_df = template_df[["Original Sample", "TMT Channel", "Sample Name", "Replicate", "conc"]]

                # Ask if auto-population of concentrations is desired.
                auto_populate = st.radio("Auto-populate concentrations for same TMT channels across all samples?",
                                         options=["No", "Yes"], index=0, key="auto_populate")
                if auto_populate == "Yes":
                    # Get a table of unique TMT channels
                    unique_tmt = template_df["TMT Channel"].unique()
                    auto_df = pd.DataFrame({
                        "TMT Channel": unique_tmt,
                        "conc": [""] * len(unique_tmt)
                    })
                    st.write("Enter concentrations corresponding to each TMT Channel (leave blank to skip):")
                    try:
                        edited_auto = st.data_editor(auto_df, key="auto_editor")
                    except Exception as e:
                        st.warning("Your Streamlit version does not support st.data_editor. Using text inputs instead.")
                        conc_values = []
                        for idx, row in auto_df.iterrows():
                            conc_val = st.text_input(f"Concentration for TMT Channel {row['TMT Channel']}:",
                                                     value=row["conc"], key=f"auto_conc_{idx}")
                            conc_values.append(conc_val)
                        auto_df["conc"] = conc_values
                        edited_auto = auto_df.copy()
                    # Now update the main template_df:
                    auto_mapping = dict(zip(edited_auto["TMT Channel"], edited_auto["conc"]))
                    template_df["conc"] = template_df["TMT Channel"].map(auto_mapping)

                st.write( "**Enter known data for samples in the template below, blanks will be autopopulated:**")
                st.write("###**Important for conc column**: Leave unknown concentrations empty or write delete to disregard measurement")
                # Use data_editor if available; otherwise, fall back to text inputs.
                try:
                    edited_template = st.data_editor(
                        template_df,
                        key="template_editor",
                        column_config={
                            "TMT Channel": st.column_config.TextColumn(disabled=True)
                        }
                    )
                except Exception as e:
                    st.warning("Your Streamlit version does not support st.data_editor. Using text inputs instead.")
                    edited_sample_names = []
                    edited_replicates = []
                    edited_concs = []
                    for idx, row in template_df.iterrows():
                        new_sample = st.text_input(
                            f"Sample Name for Original Sample {row['Original Sample']} - TMT {row['TMT Channel']}:",
                            value=row['Sample Name'], key=f"sample_{idx}")
                        new_replicate = st.text_input(
                            f"Replicate for Original Sample {row['Original Sample']} - TMT {row['TMT Channel']}:",
                            value=row['Replicate'], key=f"replicate_{idx}")
                        new_conc = st.text_input(
                            f"Concentration for Original Sample {row['Original Sample']} - TMT {row['TMT Channel']}:",
                            value=row['conc'], key=f"conc_{idx}")
                        edited_sample_names.append(new_sample)
                        edited_replicates.append(new_replicate)
                        edited_concs.append(new_conc)
                    template_df["Sample Name"] = edited_sample_names
                    template_df["Replicate"] = edited_replicates
                    template_df["conc"] = edited_concs
                    edited_template = template_df.copy()

                st.download_button("Download Template CSV",
                                   data=edited_template.to_csv(index=False).encode('utf-8'),
                                   file_name="template.csv",
                                   mime="text/csv")
                if st.button("Use Data for Processing"):
                    # For rows with a concentration value (known samples), leave Sample Name and Replicate empty.
                    def default_sample_name(row):
                        if row["conc"].strip() != "":
                            return ""
                        if pd.isna(row["Sample Name"]) or row["Sample Name"].strip() == "":
                            return row["Original Sample"]
                        return row["Sample Name"]

                    def default_replicate(row):
                        if row["conc"].strip() != "":
                            return ""
                        if pd.isna(row["Replicate"]) or row["Replicate"].strip() == "":
                            return "1"
                        return row["Replicate"]

                    edited_template["Sample Name"] = edited_template.apply(default_sample_name, axis=1)
                    edited_template["Replicate"] = edited_template.apply(default_replicate, axis=1)

                    st.session_state["edited_template"] = edited_template.copy()
                    st.session_state["template_committed"] = True
                    st.session_state["process_input_data"] = uploaded_file.getvalue()
                    st.session_state["process_template_data"] = edited_template.to_csv(index=False).encode('utf-8')
                    st.success("Data forwarded to processing. Please switch to the Calculate Concentrations page.")


        else:
            st.info("Please upload an input CSV to generate a template.")

elif menu == "Calculate Concentrations":
    st.header("Calculate Concentrations")

    # Only show the file upload prompt if data has not been loaded yet.
    if st.session_state.get("process_input_data") is None:
        st.info("If you already have a template, please upload both the template CSV and the raw CSV for processing.")
        uploaded_csv = st.file_uploader("Upload Raw CSV for processing", type=["csv"], key="processed_csv_upload")
        uploaded_template = st.file_uploader("Upload Template CSV", type=["csv"], key="processed_template_upload")

        if st.button("Start Process"):
            if uploaded_csv is not None and uploaded_template is not None:
                try:
                    raw_df = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
                    template_df = pd.read_csv(uploaded_template)
                    st.session_state["process_input_data"] = uploaded_csv.getvalue()
                    st.session_state["edited_template"] = template_df.copy()
                    st.success("Files uploaded successfully! Starting calculations...")
                except Exception as e:
                    st.error("Error reading uploaded files: " + str(e))
            else:
                st.error("Please upload both the raw CSV and the template CSV.")
        if st.session_state.get("process_input_data") is None:
            st.stop()


    # If we reach here, the raw CSV and template are already loaded.
    df = pd.read_csv(io.BytesIO(st.session_state["process_input_data"]))
    df["Transition Result"] = pd.to_numeric(df["Transition Result"], errors="coerce")

    ###############################################
    # Step 1: Precursor Filtering and ID Creation #
    ###############################################

    # First, try to find proper precursor rows
    df_precursor = df[df["Fragment Ion"].str.lower() == "precursor"].copy()

    if not df_precursor.empty:
        # Use only "precursor" fragment ion rows and pick best per peptide
        df_precursor["Precursor"] = df_precursor["Precursor"].astype(str)
        df_precursor["ID"] = (
                df_precursor["Replicate"].astype(str) + "_" +
                df_precursor["Protein"].astype(str) + "_" +
                df_precursor["Peptide"].astype(str) + "_" +
                df_precursor["Precursor"]
        )
        idx_max = df_precursor.groupby(["Replicate", "Protein", "Peptide"])["Transition Result"].idxmax()
        df_filtered_precursor = df_precursor.loc[idx_max].reset_index(drop=True)
        precursor_ids = df_filtered_precursor["ID"].unique()
        st.info(f"✅ Using {len(precursor_ids)} precursor IDs from 'Fragment Ion == precursor'")
    else:
        # Fallback: use all rows and build ID directly, no filtering
        st.warning("⚠️ No 'precursor' fragment ions found. Falling back to all rows for ID generation.")
        df["Precursor"] = df["Precursor"].astype(str)
        df["ID"] = (
                df["Replicate"].astype(str) + "_" +
                df["Protein"].astype(str) + "_" +
                df["Peptide"].astype(str) + "_" +
                df["Precursor"]
        )
        precursor_ids = df["ID"].unique()

    #####################################################
    # Step 2: Filter TMT Rows Using the Precursor IDs    #
    #####################################################
    df_tmt = df[df["Fragment Ion"].str.contains("TMT", na=False)].copy()
    df_tmt["Precursor"] = df_tmt["Precursor"].astype(str)
    df_tmt["ID"] = (
            df_tmt["Replicate"].astype(str) + "_" +
            df_tmt["Protein"].astype(str) + "_" +
            df_tmt["Peptide"].astype(str) + "_" +
            df_tmt["Precursor"].astype(str)
    )
    df_tmt_filtered = df_tmt[df_tmt["ID"].isin(precursor_ids)].copy()

    ##############################################################
    # Step 3: Merge the TMT Data with the User-Edited Template    #
    ##############################################################
    if st.session_state.get("edited_template") is None:
        st.error("Template not available. Please generate or upload the template first.")
    else:
        template_df = st.session_state["edited_template"]
        merged_df = pd.merge(
            df_tmt_filtered,
            template_df,
            how="left",
            left_on=["Replicate", "Fragment Ion"],
            right_on=["Original Sample", "TMT Channel"],
            suffixes=("", "_template")
        )
        # Exclude rows where concentration is marked for deletion and replace empty with unknown:
        # Convert the 'conc' column to strings (handling NaNs by filling them with an empty string)
        merged_df["conc"] = merged_df["conc"].fillna("").astype(str)
        merged_df = merged_df[merged_df["conc"].str.lower() != "delete"]
        merged_df["conc"] = merged_df["conc"].replace("", "unknown")
        merged_df["Sample_TMT"] = merged_df["Sample Name"].fillna("") + "_" + merged_df["Fragment Ion"].fillna("")
        #st.write("Merged Data with Template Information:")
        # st.dataframe(merged_df)
        # st.download_button(
        #     label="Download Merged Data CSV",
        #     data=merged_df.to_csv(index=False).encode('utf-8'),
        #     file_name="merged_data.csv",
        #     mime="text/csv"
        # )
        # --- New Filtering Step ---
        # Remove IDs where more than 2 known concentrations have Transition Results of 0 or NaN
        # Get only known concentration rows
        known_df = merged_df[merged_df["conc"] != "unknown"].copy()

        # Group by ID and check:
        # - at least 2 rows
        # - at least one Transition Result > 0
        valid_ids = known_df.groupby("ID").filter(
            lambda g: len(g) >= 2 and (g["Transition Result"] > 0).sum() >= 2
        )["ID"].unique()

        # Filter main merged_df to only valid IDs
        merged_df = merged_df[merged_df["ID"].isin(valid_ids)]


    # ... Continue with your downstream processing (calibration, tables, regression plots, etc.) ...

    st.subheader("LOD / Baseline options")

    use_blanks = st.checkbox(
        "Treat samples whose name contains 'blank' as blanks and compute LOD from them (median Transition Result)",
        value=True,
        help="If checked, all rows where Sample Name OR Original Sample contains 'blank' (case-insensitive) are pooled per-ID to compute an LOD baseline (median Transition Result)."
    )

    lod_override_text = st.text_input(
        "Optional: Override LOD baseline (Transition Result units) for ALL IDs (leave empty to use blanks or 0)",
        value="",
        help="If set, this numeric value overrides the blank-derived baseline for all IDs."
    )
    lod_override = None
    try:
        lod_override = float(lod_override_text) if lod_override_text.strip() != "" else None
    except:
        st.warning("LOD override must be numeric; ignoring override.")

    lod_policy = st.selectbox(
        "When a calibration point (known concentration) is below the LOD:",
        ["Drop that point", "Clamp the lowest x to 0 and keep"],
        index=0,
        help="Drop: remove those points from the fit. Clamp: if the lowest point is below LOD, set its Transition Result to 0 and keep it."
    )

    # UI
    unknown_lod_action = st.selectbox(
        "For UNKNOWN samples below LOD:",
        ["Keep (just flag BLOQ)", "Set calc_conc to 0", "Drop from wide table"],
        index=0
    )


    # ---------------------------
    # Step A: Calibration Function
    # ---------------------------
    def calibrate_group(group):
        """
        Per-ID calibration (ID = Replicate + Protein + Peptide + Precursor)
        - LOD baseline from pooled blanks (or global override)
        - Weighted least squares with 1/x^2 weights for known points
        - Below-LOD handling (drop or clamp lowest to zero)
        - Predict unknowns with SE
        """
        # Determine LOD baseline for this group
        baseline = None

        # Identify blanks (either Sample Name or Original Sample contains "blank")
        is_blank = (
                group.get("Sample Name", pd.Series("", index=group.index)).astype(str).str.contains("blank", case=False,
                                                                                                    na=False)
                | group.get("Original Sample", pd.Series("", index=group.index)).astype(str).str.contains("blank",
                                                                                                          case=False,
                                                                                                          na=False)
        )
        if use_blanks and is_blank.any():
            # median Transition Result among blanks in THIS group
            baseline = group.loc[is_blank, "Transition Result"].median(skipna=True)

        # User override wins if provided
        if lod_override is not None:
            baseline = lod_override

        # If still None (no blanks and no override), set to 0
        if baseline is None or pd.isna(baseline):
            baseline = 0.0

        # Flag all rows w.r.t. LOD baseline
        group["lod_baseline"] = baseline
        group["below_LOD"] = group["Transition Result"] < baseline

        # Known points for fitting
        known = group[group["conc"] != "unknown"].copy()
        if len(known) < 2:
            # Not enough known points to calibrate
            group["calc_conc"] = np.nan
            group["calc_se"] = np.nan
            group["slope"] = np.nan
            group["intercept"] = np.nan
            group["R2"] = np.nan
            return group

        # Parse known concentrations to float
        try:
            known["conc_float"] = known["conc"].astype(float)
        except Exception as e:
            st.error(f"Error converting known concentration values to float: {e}")
            group["calc_conc"] = np.nan
            group["calc_se"] = np.nan
            group["slope"] = np.nan
            group["intercept"] = np.nan
            group["R2"] = np.nan
            return group

        # Apply LOD policy to known points
        known_for_fit = known.copy()
        known_for_fit["clamped_to_zero"] = False

        if lod_policy == "Drop that point":
            known_for_fit = known_for_fit[~(known_for_fit["Transition Result"] < baseline)]
            if len(known_for_fit) < 2:
                group["calc_conc"] = np.nan
                group["calc_se"] = np.nan
                group["slope"] = np.nan
                group["intercept"] = np.nan
                group["R2"] = np.nan
                return group
        else:
            # Clamp the single lowest below-LOD known x to 0; drop other below-LOD known points
            below_mask = known_for_fit["Transition Result"] < baseline
            if below_mask.any():
                # index of the minimum x among below-LOD points
                min_idx = known_for_fit.loc[below_mask, "Transition Result"].idxmin()
                # clamp that one to zero
                known_for_fit.loc[min_idx, "Transition Result"] = 0.0
                known_for_fit.loc[min_idx, "clamped_to_zero"] = True
                # drop all other below-LOD known points
                drop_others = below_mask & (known_for_fit.index != min_idx)
                known_for_fit = known_for_fit.loc[~drop_others]
            if len(known_for_fit) < 2:
                group[["calc_conc", "calc_se", "slope", "intercept", "R2"]] = np.nan
                return group

        # Prepare WLS (1/x^2) design
        x = known_for_fit["Transition Result"].astype(float).values
        y = known_for_fit["conc_float"].values
        eps = 1e-12
        w = 1.0 / (np.maximum(x, eps) ** 2)

        # Fit weighted line: y = m x + b
        # numpy.polyfit supports weights; it applies them to y (WLS).
        try:
            m, b = np.polyfit(x, y, 1, w=w)
        except Exception as e:
            st.error(f"WLS fit failed: {e}")
            group["calc_conc"] = np.nan
            group["calc_se"] = np.nan
            group["slope"] = np.nan
            group["intercept"] = np.nan
            group["R2"] = np.nan
            return group

        # Weighted residuals and R^2
        y_hat = m * x + b
        resid = y - y_hat
        # Weighted SSE and SST
        ss_res = np.sum(w * resid ** 2)
        y_wmean = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - y_wmean) ** 2)
        R2w = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Estimate sigma^2 with DOF = n - 2
        n = len(y)
        dof = max(n - 2, 1)
        sigma2 = ss_res / dof

        # For prediction SE we need (X^T W X)^-1
        X = np.vstack([x, np.ones_like(x)]).T  # columns: [x, 1]
        # Build (X^T W X)
        XtWX = (X.T * w) @ X
        try:
            XtWX_inv = np.linalg.pinv(XtWX)
        except Exception:
            XtWX_inv = np.linalg.pinv(XtWX + 1e-12 * np.eye(2))

        # Predict unknowns (and keep known rows' calc_* empty)
        def predict_row(row):
            if row["conc"] == "unknown":
                x0 = float(row["Transition Result"])
                y0 = m * x0 + b
                v = np.array([x0, 1.0])
                var_pred = sigma2 * (1.0 + v @ XtWX_inv @ v.T)  # prediction variance
                se = float(np.sqrt(max(var_pred, 0.0)))
                return pd.Series({"calc_conc": y0, "calc_se": se})
            else:
                return pd.Series({"calc_conc": np.nan, "calc_se": np.nan})

        preds = group.apply(predict_row, axis=1)
        group["calc_conc"] = preds["calc_conc"]
        group["calc_se"] = preds["calc_se"]
        group["slope"] = m
        group["intercept"] = b
        group["R2"] = R2w
        return group


    # Apply calibration per unique ID (ID = Replicate + Protein + Peptide + Precursor)
    merged_calibrated = merged_df.groupby("ID").apply(calibrate_group).reset_index(drop=True)
    if unknown_lod_action != "Keep (just flag BLOQ)":
        mask_est_below = (merged_calibrated["conc"] == "unknown") & (merged_calibrated["below_LOD"])
        if unknown_lod_action == "Set calc_conc to 0":
            merged_calibrated.loc[mask_est_below, "calc_conc"] = 0.0
        else:  # "Drop from wide table"
            merged_calibrated = merged_calibrated.loc[~mask_est_below]

    ###############################################
    # Step B: Display and Download the Long Format Table
    ###############################################
    # Create a copy for display purposes so that for known calibration rows the sample mapping is blanked out.
    df_long = merged_calibrated.copy()
    mask = df_long["conc"] != "unknown"
    df_long.loc[mask, "Sample Name"] = ""
    if "Replicate_template" in df_long.columns:
        df_long.loc[mask, "Replicate_template"] = ""

    st.write("### Long Format Table (Including calculated values for Unknowns):")
    #st.dataframe(df_long)
    st.download_button(
        label="Download Long Format Table CSV",
        data=df_long.to_csv(index=False).encode('utf-8'),
        file_name="long_format_table.csv",
        mime="text/csv"
    )

    # ---------------------------
    # Step C: Create Wide Table Using Template's Sample Name + Replicate
    # ---------------------------


    # Create a combined identifier for Protein and Peptide.
    merged_calibrated["Protein_Peptide"] = (
            merged_calibrated["Protein"].astype(str) + "_" +
            merged_calibrated["Peptide"].astype(str)
    )
    # Use the user-specified Sample Name and Replicate from the template
    merged_calibrated["Sample_Rep"] = (
            merged_calibrated["Sample Name"].astype(str) + "_" +
            merged_calibrated["Replicate_template"].astype(str)
    )

    # Pivot into wide format: one table for estimated concentrations...
    conc_wide = merged_calibrated.pivot_table(
        index="Protein_Peptide",
        columns="Sample_Rep",
        values="calc_conc",
        aggfunc='first'
    )
    # ...and one for associated standard errors.
    se_wide = merged_calibrated.pivot_table(
        index="Protein_Peptide",
        columns="Sample_Rep",
        values="calc_se",
        aggfunc='first'
    )

    # Combine the two so that each Sample_Rep gets two adjacent columns.
    wide_table = pd.DataFrame(index=conc_wide.index)
    for sample in conc_wide.columns:
        wide_table[sample] = conc_wide[sample]
        wide_table[sample + "_SE"] = se_wide[sample]

    # Add a BLOQ (below LOD) sheet next to each Sample_Rep
    # We consider BLOQ for unknown (estimated) rows; for known rows we leave it empty.
    bloq_long = merged_calibrated.copy()
    bloq_long = bloq_long[bloq_long["conc"] == "unknown"][["Protein_Peptide", "Sample_Rep", "below_LOD"]]

    bloq_wide = bloq_long.pivot_table(
        index="Protein_Peptide",
        columns="Sample_Rep",
        values="below_LOD",
        aggfunc="first"
    )

    # Stitch BLOQ flags into the wide table, adjacent to each sample column
    wide_with_flags = pd.DataFrame(index=wide_table.index)
    for col in [c for c in wide_table.columns if not c.endswith("_SE")]:
        wide_with_flags[col] = wide_table[col]
        wide_with_flags[col + "_SE"] = wide_table.get(col + "_SE", np.nan)
        # Boolean flag (True=below LOD); if missing, set False
        if bloq_wide is not None and col in getattr(bloq_wide, "columns", []):
            wide_with_flags[col + "_BLOQ"] = bloq_wide[col].fillna(False).astype(bool)
        else:
            wide_with_flags[col + "_BLOQ"] = False

    st.write("### Wide Table: Estimated Concentrations, SE, and BLOQ (Below LOD) flags")
    st.dataframe(wide_with_flags)
    st.download_button(
        label="Download Wide Table (+BLOQ) CSV",
        data=wide_with_flags.to_csv().encode('utf-8'),
        file_name="wide_table_with_bloq.csv",
        mime="text/csv"
    )

    # Keep in session
    st.session_state["wide_table"] = wide_with_flags.copy()

    # st.write("### Wide Table: Estimated Concentrations and Standard Errors")
    # st.dataframe(wide_table)
    # st.download_button(
    #     label="Download Wide Table CSV",
    #     data=wide_table.to_csv().encode('utf-8'),
    #     file_name="wide_table.csv",
    #     mime="text/csv"
    # )
    # st.session_state["wide_table"] = wide_table.copy()
    #####Second optional deconvoluted wide table
    # --- New Toggle for Separate TMT Channels ---
    if st.checkbox("Show separate wide table by TMT Channel", key="toggle_separate_tmt"):
        # Create a new identifier that includes Sample Name, Replicate, and TMT Channel
        merged_calibrated["Sample_Rep_TMT"] = (
                merged_calibrated["Sample Name"].astype(str) + "_" +
                merged_calibrated["Replicate_template"].astype(str) + "_" +
                merged_calibrated["TMT Channel"].astype(str)
        )

        # Pivot table for estimated concentrations using the new identifier as columns
        conc_wide_sep = merged_calibrated.pivot_table(
            index="Protein_Peptide",
            columns="Sample_Rep_TMT",
            values="calc_conc",
            aggfunc='first'
        )

        # Pivot table for the associated standard errors
        se_wide_sep = merged_calibrated.pivot_table(
            index="Protein_Peptide",
            columns="Sample_Rep_TMT",
            values="calc_se",
            aggfunc='first'
        )

        # Combine the concentration and SE tables so each TMT channel gets two adjacent columns
        wide_table_sep = pd.DataFrame(index=conc_wide_sep.index)
        for col in conc_wide_sep.columns:
            wide_table_sep[col] = conc_wide_sep[col]
            wide_table_sep[col + "_SE"] = se_wide_sep[col]

        st.write("### Wide Table with Separate TMT Channels")
        st.dataframe(wide_table_sep)
        st.download_button(
            label="Download Separate Wide Table CSV",
            data=wide_table_sep.to_csv().encode('utf-8'),
            file_name="wide_table_separate_tmt.csv",
            mime="text/csv"
        )

    # ---------------------------
    # Interactive Regression Plotting Section
    # ---------------------------
    def plot_regression_for_id(protein, id_value, data):
        """
        Plot regression for a given protein (Protein_Peptide) and calibration group (ID).
        Shortens sample names via abbreviate_name and colors points by sample.
        """
        fig = go.Figure()

        # Separate known calibration points and unknown (estimated) values.
        known = data[data["conc"] != "unknown"]
        unknown = data[data["conc"] == "unknown"]

        if not known.empty:
            slope = known["slope"].iloc[0]
            intercept = known["intercept"].iloc[0]
            R2 = known["R2"].iloc[0]

            fig.add_trace(go.Scatter(
                x=known["Transition Result"],
                y=known["conc"].astype(float),
                mode='markers',
                marker=dict(color='blue', size=10),
                name="Known Calibration"
            ))

            x_min = known["Transition Result"].min()
            x_max = known["Transition Result"].max()
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = slope * x_vals + intercept
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='blue'),
                name=f"WLS (1/x²): y = {slope:.3e}·x + {intercept:.3f}  (R²={R2:.3f})"
            ))
        else:
            fig.add_annotation(text="No known calibration points", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.5)

        # For unknown points, assign colors by sample.
        if not unknown.empty:
            color_palette = ["red", "green", "orange", "purple", "brown", "magenta", "cyan", "olive", "teal", "navy"]
            sample_colors = {}
            color_idx = 0
            unique_samples = unknown["Sample_Rep"].dropna().unique()
            for sample in unique_samples:
                short_sample = abbreviate_name(sample, max_length=15)
                if short_sample not in sample_colors:
                    sample_colors[short_sample] = color_palette[color_idx % len(color_palette)]
                    color_idx += 1

            for sample in unique_samples:
                short_sample = abbreviate_name(sample, max_length=15)
                sample_data = unknown[unknown["Sample_Rep"] == sample]
                fig.add_trace(go.Scatter(
                    x=sample_data["Transition Result"],
                    y=sample_data["calc_conc"],
                    error_y=dict(
                        type='data',
                        array=sample_data["calc_se"],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, color=sample_colors.get(short_sample, 'black')),
                    name=f"{short_sample} Estimated"
                ))
        else:
            fig.add_annotation(text="No unknown values", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.45)

        fig.update_layout(
            title=f"Regression for {protein}",
            xaxis_title="Transition Result",
            yaxis_title="Relative Concentration",
            legend_title="Data Type"
        )

        # inside plot_regression_for_id, after computing known/unknown:
        lod = data["lod_baseline"].dropna().iloc[0] if "lod_baseline" in data and data[
            "lod_baseline"].notna().any() else None
        if lod is not None and not known.empty:
            fig.add_shape(
                type="line",
                x0=lod, x1=lod,
                y0=min(0, known["conc"].astype(float).min()),
                y1=max(known["conc"].astype(float).max(), (unknown["calc_conc"].max() if not unknown.empty else 0)),
                line=dict(dash="dot"),
                name="LOD"
            )
            fig.add_annotation(x=lod, y=0, text="LOD", showarrow=False, yshift=10)

        return fig


    # --- Interactive Plot Display ---
    selected_protein = st.selectbox("Select a Protein_Peptide",
                                    options=sorted(merged_calibrated["Protein_Peptide"].unique()))
    subset = merged_calibrated[merged_calibrated["Protein_Peptide"] == selected_protein]
    unique_ids = list(subset["ID"].unique())

    st.write("### Regression Plots (Interactive)")
    for current_id in unique_ids:
        id_data = subset[subset["ID"] == current_id]
        fig = plot_regression_for_id(selected_protein, current_id, id_data)
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{current_id}")
    # --- PDF Download Section for All Regression Plots Across All Proteins ---
    if st.button("Generate All Regression Plots as PDFs"):
        pdf_images = []
        # Loop over all unique Protein_Peptide values.
        all_proteins = sorted(merged_calibrated["Protein_Peptide"].unique())
        for protein in all_proteins:
            protein_subset = merged_calibrated[merged_calibrated["Protein_Peptide"] == protein]
            unique_ids = list(protein_subset["ID"].unique())
            for current_id in unique_ids:
                id_data = protein_subset[protein_subset["ID"] == current_id]
                fig = plot_regression_for_id(protein, current_id, id_data)
                # Capture the figure as PNG bytes (requires Kaleido)
                png_bytes = fig.to_image(format="png")
                pdf_images.append(png_bytes)
        try:
            pdf_bytes = img2pdf.convert(pdf_images)
            st.download_button(
                label="Download PDF of Regression Plots",
                data=pdf_bytes,
                file_name="regression_plots.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error creating PDF: {e}")



elif menu == "Data Refinement":
    st.header("Data Refinement")

    # Ensure the wide table is available (it should have been stored in session_state during "Calculate Concentrations")
    if "wide_table" not in st.session_state:
        st.error("Wide table not available. Please run the concentration calculations first.")
        st.stop()
    else:
        wide_table = st.session_state["wide_table"]

        # Identify concentration columns (exclude columns ending with '_SE')
        conc_cols = [col for col in wide_table.columns if not col.endswith("_SE")]

        # Calculate the median for each concentration column (skip NaN values)
        medians = wide_table[conc_cols].median(skipna=True)
        st.write("Sample Medians:", medians)

        # Calculate the overall median of these medians
        overall_median = medians.mean()
        st.write("Overall Mean of Medians:", overall_median)

        # Determine normalization factors for each column:
        # Factor = overall_median / column_median
        norm_factors = overall_median / medians
        st.write("Normalization Factors:", norm_factors)

        # Apply normalization: multiply each concentration column by its corresponding factor
        norm_wide_table = wide_table.copy()
        for col in conc_cols:
            norm_wide_table[col] = norm_wide_table[col] * norm_factors[col]

        st.subheader("Normalized Wide Table")
        st.dataframe(norm_wide_table)
        st.download_button(
            label="Download Normalized Wide Table CSV",
            data=norm_wide_table.to_csv(index=True, index_label="Protein_Peptide").encode('utf-8'),
            file_name="normalized_wide_table.csv",
            mime="text/csv"
        )

        st.subheader("Histograms of Normalized Concentrations")
        # Plot histograms for each Sample_Rep (concentration) column
        for col in conc_cols:
            fig = go.Figure(data=[go.Histogram(x=norm_wide_table[col].dropna())])
            fig.update_layout(
                title=f"Histogram for {col}",
                xaxis_title=col,
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)




    def plot_pairwise_interactive_correlations(df, columns):
        """
        Create interactive pairwise scatter plots with robust regression lines and Spearman ρ
        for every unique pair of the specified columns.
        """
        # Check if there are at least two columns to form a pair
        if len(columns) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough concentration columns for pairwise correlation plots.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Generate all unique pairs
        pairs = [(columns[i], columns[j]) for i in range(len(columns)) for j in range(i + 1, len(columns))]
        n_plots = len(pairs)

        # If no pairs are generated (shouldn't happen if len(columns) >= 2), handle it
        if n_plots == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No pairs to plot.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Define grid layout (adjust n_cols as needed)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{x} vs {y} (x vs y)" for x, y in pairs]
        )

        # Loop through each pair and add traces
        for idx, (xcol, ycol) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            subset = df[[xcol, ycol]].dropna()
            if len(subset) < 2:
                fig.add_annotation(
                    text="Not enough data", row=row, col=col, showarrow=False
                )
                continue

            # Scatter trace
            fig.add_trace(
                go.Scatter(
                    x=subset[xcol],
                    y=subset[ycol],
                    mode='markers',
                    marker=dict(color='blue'),
                    name=f"{xcol} vs {ycol}"
                ),
                row=row, col=col
            )

            # Robust regression using Theil-Sen estimator
            x_vals = subset[xcol].values.reshape(-1, 1)
            y_vals = subset[ycol].values
            model = TheilSenRegressor(random_state=42).fit(x_vals, y_vals)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))

            # Regression line trace
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name="Robust Regression"
                ),
                row=row, col=col
            )

            # Calculate Spearman correlation
            rho, _ = spearmanr(subset[xcol], subset[ycol])
            # Annotation for correlation
            fig.add_annotation(
                text=f"ρ = {rho:.2f}",
                x=0.05, y=0.95,
                xref=f"x{idx + 1 if idx > 0 else ''} domain",
                yref=f"y{idx + 1 if idx > 0 else ''} domain",
                showarrow=False,
                font=dict(color="red", size=12),
                row=row, col=col
            )

        fig.update_layout(
            height=400 * n_rows,
            width=500 * n_cols,
            showlegend=False,
            title_text="Pairwise Correlation Plots with Robust Regression"
        )
        return fig


    # --- Inside your Data Refinement block, after normalization ---
    # Identify concentration columns excluding 'Protein_Peptide' and columns with the '_SE' suffix.
    conc_cols = [col for col in norm_wide_table.columns if col != "Protein_Peptide" and not col.endswith("_SE")]

    st.subheader("Interactive Pairwise Correlation Plots of Normalized Concentrations")
    fig = plot_pairwise_interactive_correlations(norm_wide_table, conc_cols)
    st.plotly_chart(fig, use_container_width=True)
