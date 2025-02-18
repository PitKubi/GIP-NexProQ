import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from scipy.stats import t



# Helper function to abbreviate sample names
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
menu = st.sidebar.radio("Select Operation", ["Generate Template", "Calculate Concentrations"], key="page_menu")

st.title("GIP - NexProQ: Data extraction and analysis pipeline")

# ------------------
# Generate Template
# ------------------
if menu == "Generate Template":
    st.header("Generate Template")

    # If a committed template exists, just display it.
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
            # Save the uploaded file so it persists.
            st.session_state["template_input_data"] = uploaded_file.getvalue()
            try:
                df = pd.read_csv(io.BytesIO(st.session_state["template_input_data"]))
            except Exception as e:
                st.error(f"Error reading the file: {e}")
            else:
                df['Fragment Ion'] = df['Fragment Ion'].astype(str)
                df_tmt = df[df['Fragment Ion'].str.startswith('TMT', na=False)]
                unique_labels = sorted(df_tmt['Fragment Ion'].unique())
                template_df = pd.DataFrame({'Label': unique_labels, 'conc': [""] * len(unique_labels)})
                st.write("Edit the concentration values below:")
                try:
                    edited_template = st.data_editor(template_df, key="template_editor")
                except Exception as e:
                    st.warning("Your Streamlit version does not support st.data_editor. Using text inputs instead.")
                    edited_concs = []
                    for idx, row in template_df.iterrows():
                        new_val = st.text_input(f"Concentration for {row['Label']}:", value=row['conc'],
                                                key=f"conc_{idx}")
                        edited_concs.append(new_val)
                    template_df['conc'] = edited_concs
                    edited_template = template_df.copy()

                st.download_button("Download Template CSV",
                                   data=edited_template.to_csv(index=False).encode('utf-8'),
                                   file_name="template.csv",
                                   mime="text/csv")
                # Button to commit and forward data for processing.
                if st.button("Use Data for Processing"):
                    st.session_state["edited_template"] = edited_template.copy()
                    st.session_state["template_committed"] = True
                    st.session_state["process_input_data"] = uploaded_file.getvalue()
                    st.session_state["process_template_data"] = edited_template.to_csv(index=False).encode('utf-8')
                    st.success("Data forwarded to processing. Please switch to the Calculate Concentrations page.")
        else:
            st.info("Please upload an input CSV to generate a template.")

# ------------------
# Calculate Concentrations
# ------------------
elif menu == "Calculate Concentrations":
    st.header("Calculate Concentrations")

    st.subheader("Step 1. Upload Your Data (Only if not using data from 'Generate Template')")
    st.markdown("Both an **Input CSV** and a **Template CSV** are required.")

    # Use file uploaders with keys; if files are already in session_state, they will persist.
    uploaded_input = st.file_uploader("Upload Input CSV", type=["csv"], key="process_input")
    uploaded_template = st.file_uploader("Upload Template CSV", type=["csv"], key="process_template")

    if uploaded_input is not None:
        st.session_state["process_input_data"] = uploaded_input.getvalue()
    if uploaded_template is not None:
        st.session_state["process_template_data"] = uploaded_template.getvalue()

    # If an edited template was saved from Generate Template, use it.
    if st.session_state["edited_template"] is not None:
        st.info("Using saved template from Generate Template page.")
        template_df = st.session_state["edited_template"]
    else:
        # Otherwise, require the user to upload a template file.
        if st.session_state["process_template_data"] is None:
            st.error("Please upload a Template CSV.")
            st.stop()
        try:
            template_df = pd.read_csv(io.BytesIO(st.session_state["process_template_data"]))
        except Exception as e:
            st.error(f"Error reading the Template CSV: {e}")
            st.stop()

    # Process the input file.
    if st.session_state["process_input_data"] is None:
        st.error("Please upload an Input CSV.")
        st.stop()
    try:
        df = pd.read_csv(io.BytesIO(st.session_state["process_input_data"]))
    except Exception as e:
        st.error(f"Error reading the Input CSV: {e}")
        st.stop()

    required_cols = ['Replicate', 'Protein', 'Peptide', 'Precursor', 'Fragment Ion', 'Transition Result']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Error: Column '{col}' not found in the Input CSV.")
            st.stop()
        else:
            df[col] = df[col].astype(str)

    df['ID'] = df['Replicate'] + '_' + df['Protein'] + '_' + df['Peptide'] + '_' + df['Precursor']
    filtered_df = df[df['Fragment Ion'].str.startswith('TMT', na=False)].copy()

    template_df['conc'] = pd.to_numeric(template_df['conc'], errors='coerce')
    conc_mapping = dict(zip(template_df['Label'], template_df['conc']))
    filtered_df['conc'] = filtered_df['Fragment Ion'].map(conc_mapping)

    st.write("Processing your data...")


    st.subheader("Step 2. Group or rename TMT channels containing unknown concentrations, if desired:")
    # ===== Experiment Grouping for Unknown Channels =====
    unknown_channels = template_df.loc[template_df['conc'].isnull(), 'Label'].tolist()
    if unknown_channels:
        unknown_df = pd.DataFrame({'Channel': unknown_channels})
        unknown_df['Experiment Group'] = unknown_df['Channel']
        if "experiment_group_mapping" in st.session_state and st.session_state["experiment_group_mapping"]:
            mapping = st.session_state["experiment_group_mapping"]
            unknown_df['Experiment Group'] = unknown_df['Channel'].apply(lambda ch: mapping.get(ch, ch))
        with st.expander("Group Unknown TMT Channels into Experiment Groups", expanded=False):
            edited_unknown_df = st.data_editor(unknown_df, key="unknown_groups", use_container_width=True)
            if st.button("Apply Grouping Changes", key="apply_grouping"):
                new_mapping = dict(zip(edited_unknown_df['Channel'], edited_unknown_df['Experiment Group']))
                new_mapping = {ch: grp for ch, grp in new_mapping.items() if grp.strip() and grp.strip().lower() != "delete"}
                st.session_state["experiment_group_mapping"] = new_mapping
                st.session_state["grouping_committed"] = True
            if st.session_state.get("grouping_committed", False):
                st.success("Experiment grouping updated.")
        experiment_group_mapping = st.session_state.get("experiment_group_mapping",
                                                        dict(zip(edited_unknown_df['Channel'],
                                                                 edited_unknown_df['Experiment Group'])))
    else:
        experiment_group_mapping = {}
        st.session_state["experiment_group_mapping"] = experiment_group_mapping

    st.subheader("Step 3. Rename Samples/Replicates if needed:")
    # --- Sample Renaming ---
    unique_reps = sorted(filtered_df['Replicate'].unique())
    if "sample_names" not in st.session_state:
        st.session_state["sample_names"] = {rep: rep for rep in unique_reps}
    st.write("Rename samples (each must be unique):")
    for rep in unique_reps:
        new_name = st.text_input(f"New name for sample '{rep}':", value=st.session_state["sample_names"].get(rep, rep),
                                 key=f"rename_sample_{rep}")
        st.session_state["sample_names"][rep] = new_name
    sample_names = st.session_state["sample_names"]
    if len(set(sample_names.values())) < len(sample_names):
        st.error("Duplicate sample names detected. Please assign unique sample names to each replicate.")
        st.stop()
    st.subheader("Step 4. (De-)Select TMT channels to calculate calibration curves:")
    # ===== Replicate-Specific Calibration Point Selection =====
    replicate_inclusions = {}
    for rep in unique_reps:
        rep_df = filtered_df[filtered_df['Replicate'] == rep].copy()
        rep_df['conc_numeric'] = pd.to_numeric(rep_df['conc'], errors='coerce')
        available_labels = sorted(rep_df.loc[rep_df['conc_numeric'].notnull(), 'Fragment Ion'].unique())
        default_selection = st.session_state.get("replicate_inclusions", {}).get(sample_names[rep], available_labels)
        selection = st.multiselect(
            f"Select TMT labels to include for sample {sample_names[rep]} (calibration points)",
            options=available_labels,
            default=default_selection,
            key=f"include_{rep}"
        )
        replicate_inclusions[sample_names[rep]] = selection
    st.session_state["replicate_inclusions"] = replicate_inclusions

    # --- Regression Processing (Per Replicate, Protein, Peptide) ---
    wide_data = {}      # Final wide-format table keyed by Protein_Peptide.
    regression_info = {}  # Regression details keyed by (SampleName_Protein_Peptide).

    for key, group in filtered_df.groupby(['Replicate', 'Protein', 'Peptide']):
        rep, prot, pep = key
        protein_peptide = f"{prot}_{pep}"
        sample_name = sample_names[rep]
        rep_key = f"{sample_name}_{protein_peptide}"
        group = group.copy()
        group['conc_numeric'] = pd.to_numeric(group['conc'], errors='coerce')
        known = group[group['conc_numeric'].notnull()].copy()
        known = known[known['Fragment Ion'].isin(replicate_inclusions.get(sample_name, []))]
        unknown = group[group['conc_numeric'].isnull()].copy()
        if known.empty or len(known) < 2:
            continue

        known = known.sort_values(by='conc_numeric')
        x = known['conc_numeric'].values
        y = known['Transition Result'].astype(float).values

        slope = np.sum(x * y) / np.sum(x ** 2)
        residuals = y - slope * x
        n = len(x)
        se = np.sqrt(np.sum(residuals ** 2) / (n - 1)) if n > 1 else np.nan

        exp_group_values = {}
        for _, row in unknown.iterrows():
            try:
                trans_result = float(row['Transition Result'])
            except ValueError:
                trans_result = np.nan
            estimated_conc = trans_result / slope if slope != 0 else np.nan
            se_percent = (se / estimated_conc) * 100 if (estimated_conc not in [0, np.nan] and not np.isnan(estimated_conc)) else np.nan
            channel = row['Fragment Ion']
            group_label = experiment_group_mapping.get(channel, channel)
            if group_label and group_label.lower() != "delete":
                if group_label not in exp_group_values:
                    exp_group_values[group_label] = []
                exp_group_values[group_label].append(estimated_conc)
            else:
                sample_col = f"{sample_name}_{channel}"
                se_col = f"SE_{sample_name}_{channel}"
                if protein_peptide not in wide_data:
                    wide_data[protein_peptide] = {}
                wide_data[protein_peptide][sample_col] = estimated_conc
                wide_data[protein_peptide][se_col] = se_percent

        # For each experiment group, compute aggregated (median) value and combined SE.
        for grp, vals in exp_group_values.items():
            if len(vals) == 0:
                continue
            median_val = np.median(vals)
            if len(vals) > 1:
                combined_se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            else:
                combined_se = se / slope if slope != 0 else np.nan
            exp_col = f"{sample_name}_Exp_{grp}"
            se_exp_col = f"SE_{sample_name}_Exp_{grp}"
            if protein_peptide not in wide_data:
                wide_data[protein_peptide] = {}
            wide_data[protein_peptide][exp_col] = median_val
            wide_data[protein_peptide][se_exp_col] = combined_se

        # Save regression details for this replicate sample.
        # Save regression details for this replicate sample.
        if rep_key not in regression_info:
            regression_info[rep_key] = {
                'slope': slope,
                'se': se,
                'known_x': x,
                'known_y': y,
                'unknown_points': []
            }
        for _, row in unknown.iterrows():
            try:
                trans_result = float(row['Transition Result'])
            except ValueError:
                trans_result = np.nan
            estimated_conc = trans_result / slope if slope != 0 else np.nan
            conc_error = se / slope if slope != 0 else np.nan
            # --- New: Use the user-specified group for labeling ---
            channel = row['Fragment Ion']
            group_label = experiment_group_mapping.get(channel, channel)
            # Build a base label using the sample name and the group label.
            sample_label = f"{sample_name}_{group_label}"
            regression_info[rep_key]['unknown_points'].append({
                'sample': sample_label,  # base label; numbering will be added later if needed
                'channel': channel,  # store the original channel for reference
                'group': group_label,  # store the group label
                'estimated_conc': estimated_conc,
                'conc_error': conc_error,
                'transition_result': trans_result
            })

    wide_df = pd.DataFrame.from_dict(wide_data, orient='index')
    wide_df.index.name = 'Protein_Peptide'
    wide_df.reset_index(inplace=True)

    if wide_df.empty:
        st.info("No valid regression groups found. Please verify that your template CSV provides valid known concentrations for at least two rows per group, and that you have selected calibration points for each replicate.")
    else:
        st.subheader("Step 5. See Processed Data and visualizations below:")
        # Editable Column Headers inside a collapsible expander (collapsed by default)
        with st.expander("Edit Column Headers", expanded=False):
            new_names = {}
            for col in wide_df.columns:
                if col != "Protein_Peptide":
                    new_names[col] = st.text_input(f"New name for '{col}':", value=col, key=f"rename_{col}")
                else:
                    new_names[col] = col
        wide_df_edited = wide_df.rename(columns=new_names)
        st.dataframe(wide_df_edited)

        wide_filename = st.text_input("Enter file name for the processed wide CSV:", value="wide_output.csv", key="wide_filename")
        wide_csv = wide_df_edited.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Processed Wide CSV", data=wide_csv, file_name=wide_filename, mime="text/csv")

    # --- Regression Plot Section ---
    if regression_info:
        st.subheader("Regression Plot Viewer")
        selected_rep_key = st.selectbox("Select a replicate sample to view its regression curve", list(regression_info.keys()))
        details = regression_info[selected_rep_key]
        # --- Update unknown point labels with numbering using underscore ---
        group_counts = {}
        for pt in details['unknown_points']:
            grp = pt.get('group', pt['sample'])
            if grp in group_counts:
                group_counts[grp] += 1
                pt['sample'] = f"{grp}_{group_counts[grp]}"
            else:
                group_counts[grp] = 1
                pt['sample'] = f"{grp}_1"

        slope = details['slope']
        se = details['se']
        known_x = details['known_x']
        known_y = details['known_y']

        x_fit = np.linspace(min(known_x), max(known_x), 100)
        y_fit = slope * x_fit

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=known_x,
            y=known_y,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Known Data'
        ))
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name='Regression Line'
        ))
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, pt in enumerate(details['unknown_points']):
            fig.add_trace(go.Scatter(
                x=[pt['estimated_conc']],
                y=[pt['transition_result']],
                mode='markers',
                marker=dict(size=10, color=colors[i % len(colors)]),
                name=pt['sample'],
                error_y=dict(
                    type='constant',
                    value=pt.get('conc_error', se / slope if slope != 0 else np.nan),
                    visible=True
                ),
                hovertemplate=f"Sample: {pt['sample']}<br>Est. Conc: {pt['estimated_conc']:.3f} (± {pt.get('conc_error', se / slope if slope != 0 else np.nan):.3f})<br>Transition: {pt['transition_result']:.3f}"
            ))
        fig.update_layout(
            title=selected_rep_key,
            xaxis_title="Concentration",
            yaxis_title="Transition Result",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig)

        try:
            fig_image = fig.to_image(format="png")
            st.download_button(
                label="Download Figure Image",
                data=fig_image,
                file_name=f"{selected_rep_key}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error("Error generating image for download. Ensure that kaleido is installed.")

        rows = []
        rep_id = selected_rep_key.split('_')[0]
        prot_pep = "_".join(selected_rep_key.split('_')[1:])
        for pt in details['unknown_points']:
            rows.append({
                "Replicate": rep_id,
                "Protein_Peptide": prot_pep,
                "Sample": pt['sample'],
                "Estimated Conc": pt['estimated_conc'],
                "Estimated Conc Error": pt.get('conc_error', se / slope if slope != 0 else np.nan),
                "Transition Result": pt['transition_result'],
                "Slope": slope
            })
        regression_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        st.download_button(
            label="Download Regression CSV (User-Friendly Format) for this replicate sample",
            data=regression_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_rep_key}_regression.csv",
            mime="text/csv"
        )


    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import t

    # --- QC Analysis Section ---
    if regression_info:
        st.subheader("QC Analysis")
        with st.expander("View QC Metrics and Plots", expanded=True):

            ##########################################################
            # 1. Custom Percentage Error Distributions (IQR-only)
            ##########################################################
            sample_perc_data = []
            y_positions_perc = {}
            pos = 0
            for col in wide_df.columns:
                if col != "Protein_Peptide" and not col.startswith("SE_"):
                    se_col = "SE_" + col
                    if se_col in wide_df.columns:
                        measured = pd.to_numeric(wide_df[col], errors='coerce')
                        se_vals = pd.to_numeric(wide_df[se_col], errors='coerce')
                        valid = measured.notnull() & se_vals.notnull() & (measured != 0)
                        if valid.sum() > 0:
                            perc_errors = (se_vals[valid] / measured[valid]) * 100
                            q1 = np.percentile(perc_errors, 25)
                            med = np.percentile(perc_errors, 50)
                            q3 = np.percentile(perc_errors, 75)
                            sample_perc_data.append((col, q1, med, q3))
                            y_positions_perc[col] = pos
                            pos += 1

            fig_perc = go.Figure()
            for (col, q1, med, q3) in sample_perc_data:
                y = y_positions_perc[col]
                # Draw a rectangle representing the IQR.
                fig_perc.add_shape(
                    type="rect",
                    xref="x", yref="y",
                    x0=q1, x1=q3,
                    y0=y - 0.3, y1=y + 0.3,
                    fillcolor="rgba(100,200,100,0.5)",
                    line=dict(width=0),
                    layer="below"
                )
                # Mark the median.
                fig_perc.add_trace(go.Scatter(
                    x=[med],
                    y=[y],
                    mode="markers",
                    marker=dict(color="black", size=10),
                    showlegend=False,
                    hovertemplate=f"{col}<br>Median: {med:.2f}%"
                ))
            fig_perc.update_yaxes(
                tickvals=list(y_positions_perc.values()),
                ticktext=[abbreviate_name(name) for name in list(y_positions_perc.keys())]
            )
            fig_perc.update_layout(
                title="Percentage Error Distributions (IQR-only)",
                xaxis_title="Percentage Error (%)",
                height=300
            )
            st.plotly_chart(fig_perc)

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import numpy as np
            from scipy.stats import t

            ##################################################################
            # Overall Regression Plot with Measured Concentration Distributions
            # (Custom IQR-only Boxplots with Known Calibration Medians, 95% CI,
            #  and an Annotation Showing the Regression Equation with SE)
            ##################################################################

            # --- Process Known Calibration Data ---
            all_known_x = np.concatenate([info['known_x'] for info in regression_info.values()])
            all_known_y = np.concatenate([info['known_y'] for info in regression_info.values()])

            if len(all_known_x) > 0:
                # Group known data by rounding the calibration concentration to 2 decimals.
                rounded_known_x = np.round(all_known_x, 2)
                unique_known = np.unique(rounded_known_x)
                median_known_x = []
                median_known_y = []
                for val in unique_known:
                    indices = np.where(rounded_known_x == val)[0]
                    median_known_x.append(val)
                    median_known_y.append(np.median(all_known_y[indices]))
                median_known_x = np.array(median_known_x)
                median_known_y = np.array(median_known_y)
            else:
                median_known_x = np.array([])
                median_known_y = np.array([])

            n_med = len(median_known_x)
            if n_med > 1:
                # Compute a regression through the origin using the median calibration points.
                slope_med = np.sum(median_known_x * median_known_y) / np.sum(median_known_x ** 2)
                residuals_med = median_known_y - slope_med * median_known_x
                sigma2_med = np.sum(residuals_med ** 2) / (n_med - 1)
                # Standard error of the slope for regression through the origin:
                se_slope_med = np.sqrt(sigma2_med / np.sum(median_known_x ** 2))

                # Compute the 95% CI for each calibration point (error at x is given by):
                #    error = t_val * ( x * sqrt(sigma2_med / sum(median_known_x**2)) )
                t_val = t.ppf(0.975, n_med - 1)
                error_array = t_val * (median_known_x * np.sqrt(sigma2_med / np.sum(median_known_x ** 2)))

                # --- Determine X-axis Range ---
                # Maximum known calibration concentration from the median data:
                max_known = np.max(median_known_x)

                # Build IQR data for each unknown sample (from wide_df):
                sample_iqr_data = []
                y_positions = {}
                pos = 0
                for col in wide_df.columns:
                    if col != "Protein_Peptide" and not col.startswith("SE_"):
                        vals = pd.to_numeric(wide_df[col], errors='coerce').dropna()
                        if not vals.empty:
                            q1 = np.percentile(vals, 25)
                            med = np.percentile(vals, 50)
                            q3 = np.percentile(vals, 75)
                            sample_iqr_data.append((col, q1, med, q3))
                            y_positions[col] = pos
                            pos += 1
                # Determine the maximum unknown value based on the Q3 of each sample.
                if sample_iqr_data:
                    max_unknown = max(q3 for (_, _, _, q3) in sample_iqr_data)
                else:
                    max_unknown = 0

                max_x_val = max(max_known, max_unknown)
                shared_x_range = [0, max_x_val * 1.05]  # 5% padding

                # --- Create Regression Curve Data (Top Panel) ---
                # Use the known calibration range (0 to max_known) for the regression line.
                top_x_range = np.linspace(0, max_known, 100)
                y_fit = slope_med * top_x_range
                # 95% CI for the regression line (through the origin):
                se_pred = top_x_range * np.sqrt(sigma2_med / np.sum(median_known_x ** 2))
                y_upper = y_fit + t_val * se_pred
                y_lower = y_fit - t_val * se_pred

                # --- Build the Subplots ---
                # Two rows: Top for the regression curve and calibration medians; Bottom for unknowns’ IQR boxes.
                fig_reg = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,  # shared x-axis so that the scale is uniform
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.4],
                    subplot_titles=("Overall Regression Curve", "Measured Concentration Distributions (IQR-only)")
                )

                # Top Panel: Plot the regression line.
                fig_reg.add_trace(go.Scatter(
                    x=top_x_range,
                    y=y_fit,
                    mode='lines',
                    name='Overall Regression Line',
                    line=dict(color='black', width=2)
                ), row=1, col=1)

                # Top Panel: Plot the 95% CI as a filled ribbon.
                fig_reg.add_trace(go.Scatter(
                    x=np.concatenate([top_x_range, top_x_range[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name='95% Confidence Interval'
                ), row=1, col=1)

                # Top Panel: Overlay the known median calibration points with 95% CI error bars.
                fig_reg.add_trace(go.Scatter(
                    x=median_known_x,
                    y=median_known_y,
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Known Medians',
                    error_y=dict(
                        type='data',
                        array=error_array,
                        arrayminus=error_array,
                        visible=True
                    ),
                    hovertemplate="Known Concentration: %{x}<br>Median Transition: %{y}<br>95% CI ± %{error_y.array:.2f}"
                ), row=1, col=1)

                # Add an annotation showing the regression formula with SE.
                # We'll position it in the top panel (using xref "x1" and yref "y1").
                fig_reg.add_annotation(
                    x=shared_x_range[1] * 0.75,
                    y=y_fit[-1] * 0.95,
                    xref="x1",
                    yref="y1",
                    text=f"y = {slope_med:.2f}x (SE = {se_slope_med:.2f})",
                    showarrow=False,
                    font=dict(color="blue", size=12)
                )

                # Bottom Panel: For each unknown sample, plot a custom IQR box.
                for (col, q1, med, q3) in sample_iqr_data:
                    y = y_positions[col]
                    fig_reg.add_shape(
                        type="rect",
                        xref="x",  # using the shared x-axis
                        yref="y2",  # y-axis for row 2
                        x0=q1, x1=q3,
                        y0=y - 0.3, y1=y + 0.3,
                        fillcolor="rgba(100,200,100,0.5)",
                        line=dict(width=0),
                        layer="below"
                    )
                    fig_reg.add_trace(go.Scatter(
                        x=[med],
                        y=[y],
                        mode="markers",
                        marker=dict(color="black", size=10),
                        showlegend=False,
                        hovertemplate=f"{col}<br>Median: {med:.2f}"
                    ), row=2, col=1)

                # Update bottom panel y-axis to list sample names.
                fig_reg.update_yaxes(
                    tickvals=list(y_positions.values()),
                    ticktext=[abbreviate_name(name) for name in list(y_positions.keys())],
                    row=2, col=1
                )

                # Set the shared x-axis range for both panels.
                fig_reg.update_xaxes(range=shared_x_range, row=1, col=1)
                fig_reg.update_xaxes(range=shared_x_range, row=2, col=1)

                # Remove the x-axis title from the top panel and add it only to the bottom.
                fig_reg.update_xaxes(title_text="", row=1, col=1)
                fig_reg.update_xaxes(title_text="Concentration", row=2, col=1)

                fig_reg.update_layout(
                    title="Overall Regression Curve with Measured Concentration Distributions (IQR-only)",
                    height=600
                )

                st.plotly_chart(fig_reg)
            else:
                st.info("Not enough known data points to compute an overall regression.")
            import io

            # --- Prepare QC Statistics for Download as CSV ---

            # Recompute Calibration Statistics (known_stats_df)
            csv_buffer = io.StringIO()
            csv_buffer.write("Calibration Stats\n")
            if len(all_known_x) > 0:
                # Group the known calibration data by rounding the concentration to 2 decimals.
                rounded_known_x = np.round(all_known_x, 2)
                unique_known = np.unique(rounded_known_x)
                known_stats_list = []
                for val in unique_known:
                    indices = np.where(rounded_known_x == val)[0]
                    median_val = np.median(all_known_y[indices])
                    q1_val = np.percentile(all_known_y[indices], 25)
                    q3_val = np.percentile(all_known_y[indices], 75)
                    known_stats_list.append({
                        "Calibration": val,
                        "Median Transition": median_val,
                        "Q1": q1_val,
                        "Q3": q3_val
                    })
                import pandas as pd

                known_stats_df = pd.DataFrame(known_stats_list)
                known_stats_df.to_csv(csv_buffer, index=False)
            else:
                csv_buffer.write("No calibration stats available.\n")

            csv_buffer.write("\nUnknown Group Stats\n")
            # Compute unknown group statistics from wide_df and experiment_group_mapping.
            unknown_group_stats = []
            for col in wide_df.columns:
                if col != "Protein_Peptide" and not col.startswith("SE_"):
                    # Use the user-specified group if provided; otherwise, use the column name.
                    group_label = experiment_group_mapping.get(col, col)
                    vals = pd.to_numeric(wide_df[col], errors='coerce').dropna().values
                    if len(vals) > 0:
                        q1 = np.percentile(vals, 25)
                        med = np.percentile(vals, 50)
                        q3 = np.percentile(vals, 75)
                        unknown_group_stats.append({
                            "Group": group_label,
                            "Channel": col,
                            "n": len(vals),
                            "Q1": q1,
                            "Median": med,
                            "Q3": q3
                        })
            import pandas as pd

            unknown_group_stats_df = pd.DataFrame(unknown_group_stats)
            unknown_group_stats_df.to_csv(csv_buffer, index=False)

            csv_buffer.write("\nRegression Parameters\n")
            # Build overall regression parameters from regression_info.
            qc_regression_list = []
            for rep_key, info in regression_info.items():
                qc_regression_list.append({
                    "Replicate_Sample": rep_key,
                    "Slope": info['slope'],
                    "SE": info['se'],
                    "n_known": len(info['known_x']),
                    "n_unknown": len(info['unknown_points'])
                })
            qc_regression_df = pd.DataFrame(qc_regression_list)
            qc_regression_df.to_csv(csv_buffer, index=False)

            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Download QC Statistics (CSV)",
                data=csv_data,
                file_name="QC_Statistics.csv",
                mime="text/csv"
            )

