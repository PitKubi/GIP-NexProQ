import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import img2pdf
from scipy.stats import t
from plotly.subplots import make_subplots  # required for subplots


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
        st.stop()  # Stop further execution until the files are loaded.

    # If we reach here, the raw CSV and template are already loaded.
    df = pd.read_csv(io.BytesIO(st.session_state["process_input_data"]))
    df["Transition Result"] = pd.to_numeric(df["Transition Result"], errors="coerce")


    ###############################################
    # Step 1: Precursor Filtering and ID Creation #
    ###############################################
    df_precursor = df[df["Fragment Ion"].str.lower() == "precursor"].copy()
    df_precursor["Precursor"] = df_precursor["Precursor"].astype(str)
    df_precursor["ID"] = (
            df_precursor["Replicate"].astype(str) + "_" +
            df_precursor["Protein"].astype(str) + "_" +
            df_precursor["Peptide"].astype(str) + "_" +
            df_precursor["Precursor"].astype(str)
    )
    idx_max = df_precursor.groupby(["Replicate", "Protein", "Peptide"])["Transition Result"].idxmax()
    df_filtered_precursor = df_precursor.loc[idx_max].reset_index(drop=True)
    precursor_ids = df_filtered_precursor["ID"].unique()

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
        valid_ids = merged_df[merged_df["conc"] != "unknown"] \
            .groupby("ID")["Transition Result"] \
            .apply(lambda x: (x.isna() | (x == 0)).sum() <= 2)

        # Get the IDs that meet the condition
        valid_ids = valid_ids[valid_ids].index

        # Filter merged_df to keep only rows with valid IDs
        merged_df = merged_df[merged_df["ID"].isin(valid_ids)]


    # ... Continue with your downstream processing (calibration, tables, regression plots, etc.) ...




    # ---------------------------
    # Step A: Calibration Function
    # ---------------------------
    def calibrate_group(group):
        # Compute regression parameters using only rows with known concentration (conc != "unknown")
        known = group[group["conc"] != "unknown"].copy()
        n = len(known)
        if n < 2:
            group["calc_conc"] = np.nan
            group["calc_se"] = np.nan
            group["slope"] = np.nan
            group["intercept"] = np.nan
            group["R2"] = np.nan
            return group
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

        # Perform linear regression on the known points
        x = known["Transition Result"].astype(float)
        y = known["conc_float"]
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else np.sum(residuals ** 2) / n
        x_mean = x.mean()
        Sxx = np.sum((x - x_mean) ** 2)
        R2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2) if np.sum((y - y.mean()) ** 2) > 0 else 1.0

        # For each row, if the concentration is unknown, predict it using the regression;
        # if known, leave the calculated values empty.
        def predict_with_se(row):
            if row["conc"] == "unknown":
                x0 = float(row["Transition Result"])
                pred = slope * x0 + intercept
                se = np.sqrt(mse * (1 + 1 / n + ((x0 - x_mean) ** 2 / Sxx))) if Sxx > 0 else np.sqrt(mse * (1 + 1 / n))
                return pd.Series({"calc_conc": pred, "calc_se": se})
            else:
                return pd.Series({"calc_conc": np.nan, "calc_se": np.nan})

        preds = group.apply(predict_with_se, axis=1)
        group["calc_conc"] = preds["calc_conc"]
        group["calc_se"] = preds["calc_se"]
        group["slope"] = slope
        group["intercept"] = intercept
        group["R2"] = R2
        return group


    # Apply calibration per unique ID (ID = Replicate + Protein + Peptide + Precursor)
    merged_calibrated = merged_df.groupby("ID").apply(calibrate_group).reset_index(drop=True)

    ###############################################
    # Step B: Display and Download the Long Format Table
    ###############################################
    # Create a copy for display purposes so that for known calibration rows the sample mapping is blanked out.
    df_long = merged_calibrated.copy()
    mask = df_long["conc"] != "unknown"
    df_long.loc[mask, "Sample Name"] = ""
    df_long.loc[mask, "Replicate_template"] = ""  # Adjust the column name if needed

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

    st.write("### Wide Table: Estimated Concentrations and Standard Errors")
    st.dataframe(wide_table)
    st.download_button(
        label="Download Wide Table CSV",
        data=wide_table.to_csv().encode('utf-8'),
        file_name="wide_table.csv",
        mime="text/csv"
    )
    st.session_state["wide_table"] = wide_table.copy()


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
                name=f"Fit: y = {slope:.2e}x + ({intercept:.2f}) (R²={R2:.1f})"
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

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr
    from sklearn.linear_model import TheilSenRegressor


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
