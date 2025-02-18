import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
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
menu = st.sidebar.radio("Select Operation", ["Generate Template", "Calculate Concentrations", "Protein Concentration analysis"], key="page_menu")

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
                # Let the user choose the template type
                template_option = st.radio("Select Template Type",
                                           options=["Simple Template (Small Sample Size)", "Large Cohort Template"],
                                           index=0)

                if template_option == "Simple Template (Small Sample Size)":
                    # Proceed as before:
                    df['Fragment Ion'] = df['Fragment Ion'].astype(str)
                    df_tmt = df[df['Fragment Ion'].str.startswith('TMT', na=False)]
                    unique_labels = sorted(df_tmt['Fragment Ion'].unique())
                    template_df = pd.DataFrame({'Label': unique_labels, 'conc': [""] * len(unique_labels)})
                    st.write("Edit the concentration values below:")
                else:
                    # Large Cohort Template:
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
                    st.write("Edit the 'Sample Name', 'Replicate', and 'conc' values below:")

                    # Ask if auto-population of concentrations is desired.
                    auto_populate = st.radio("Auto-populate concentrations for unique TMT channels?",
                                             options=["No", "Yes"], index=0, key="auto_populate")
                    if auto_populate == "Yes":
                        # Get a table of unique TMT channels
                        unique_tmt = template_df["TMT Channel"].unique()
                        auto_df = pd.DataFrame({
                            "TMT Channel": unique_tmt,
                            "conc": [""] * len(unique_tmt)
                        })
                        st.write("Enter concentration for each unique TMT channel (leave blank to skip):")
                        try:
                            edited_auto = st.data_editor(auto_df, key="auto_editor")
                        except Exception as e:
                            st.warning(
                                "Your Streamlit version does not support st.data_editor. Using text inputs instead.")
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

                # Use data_editor if available; otherwise, fall back to text inputs.
                try:
                    edited_template = st.data_editor(template_df, key="template_editor")
                except Exception as e:
                    st.warning("Your Streamlit version does not support st.data_editor. Using text inputs instead.")
                    if template_option == "Simple Template (Small Sample Size)":
                        edited_concs = []
                        for idx, row in template_df.iterrows():
                            new_val = st.text_input(f"Concentration for {row['Label']}:", value=row['conc'],
                                                    key=f"conc_{idx}")
                            edited_concs.append(new_val)
                        template_df['conc'] = edited_concs
                        edited_template = template_df.copy()
                    else:
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
    st.subheader("Step 1: Upload Your Data")
    st.markdown("Both an **Input CSV** and a **Template CSV** are required.")
    uploaded_input = st.file_uploader("Upload Input CSV", type=["csv"], key="process_input")
    uploaded_template = st.file_uploader("Upload Template CSV", type=["csv"], key="process_template")
    if uploaded_input is not None:
        st.session_state["process_input_data"] = uploaded_input.getvalue()
    if uploaded_template is not None:
        st.session_state["process_template_data"] = uploaded_template.getvalue()

    st.subheader("Step 2: Prepare Data")
    if st.session_state["edited_template"] is not None:
        st.info("Using saved template from Generate Template page.")
        template_df = st.session_state["edited_template"]
    else:
        if st.session_state["process_template_data"] is None:
            st.error("Please upload a Template CSV.")
            st.stop()
        try:
            template_df = pd.read_csv(io.BytesIO(st.session_state["process_template_data"]))
        except Exception as e:
            st.error(f"Error reading the Template CSV: {e}")
            st.stop()
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

    st.subheader("Step 3: Group Unknown TMT Channels")
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

    st.subheader("Step 4: Rename Samples/Replicates")
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

    st.subheader("Step 5: Select Calibration Points")
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

    st.subheader("Step 6: Regression Processing (Per Replicate, Protein, Peptide, [Precursor if needed])")
    wide_data = {}      # Final wide-format table keyed by a unique group key.
    regression_info = {}  # Regression details keyed by the unique group key.
    group_counts = {}  # for unique key generation

    # First, group by Replicate, Protein, Peptide.
    for key, group in filtered_df.groupby(['Replicate', 'Protein', 'Peptide']):
        rep, prot, pep = key
        # If there are multiple precursors, further group by 'Precursor'
        if group['Precursor'].nunique() > 1:
            for prec, sub_group in group.groupby('Precursor'):
                base_key = f"{sample_names[rep]}_{prot}_{pep}_{prec}"
                if base_key in group_counts:
                    group_counts[base_key] += 1
                    unique_key = f"{base_key}_{group_counts[base_key]}"
                else:
                    group_counts[base_key] = 1
                    unique_key = base_key
                rep_key = unique_key
                process_group = sub_group.copy()
                process_group['conc_numeric'] = pd.to_numeric(process_group['conc'], errors='coerce')
                # *** Added Protein column ***
                if unique_key not in wide_data:
                    wide_data[unique_key] = {}
                wide_data[unique_key]["Protein"] = prot
                known = process_group[process_group['conc_numeric'].notnull()].copy()
                known = known[known['Fragment Ion'].isin(replicate_inclusions.get(sample_names[rep], []))]
                unknown = process_group[process_group['conc_numeric'].isnull()].copy()
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
                        sample_col = f"{sample_names[rep]}_{channel}"
                        se_col = f"SE_{sample_names[rep]}_{channel}"
                        if unique_key not in wide_data:
                            wide_data[unique_key] = {}
                        wide_data[unique_key][sample_col] = estimated_conc
                        wide_data[unique_key][se_col] = se_percent
                for grp, vals in exp_group_values.items():
                    if len(vals) == 0:
                        continue
                    median_val = np.median(vals)
                    if len(vals) > 1:
                        combined_se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                    else:
                        combined_se = se / slope if slope != 0 else np.nan
                    exp_col = f"{sample_names[rep]}_Exp_{grp}"
                    se_exp_col = f"SE_{sample_names[rep]}_Exp_{grp}"
                    if unique_key in wide_data:
                        wide_data[unique_key][exp_col] = median_val
                        wide_data[unique_key][se_exp_col] = combined_se
                    else:
                        wide_data[unique_key] = {exp_col: median_val, se_exp_col: combined_se}
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
                    channel = row['Fragment Ion']
                    group_label = experiment_group_mapping.get(channel, channel)
                    sample_label = f"{sample_names[rep]}_{group_label}"
                    regression_info[rep_key]['unknown_points'].append({
                        'sample': sample_label,
                        'channel': channel,
                        'group': group_label,
                        'estimated_conc': estimated_conc,
                        'conc_error': conc_error,
                        'transition_result': trans_result
                    })
        else:
            base_key = f"{sample_names[rep]}_{prot}_{pep}"
            if base_key in group_counts:
                group_counts[base_key] += 1
                unique_key = f"{base_key}_{group_counts[base_key]}"
            else:
                group_counts[base_key] = 1
                unique_key = base_key
            rep_key = unique_key
            process_group = group.copy()
            process_group['conc_numeric'] = pd.to_numeric(process_group['conc'], errors='coerce')
            # *** Added Protein column ***
            if unique_key not in wide_data:
                wide_data[unique_key] = {}
            wide_data[unique_key]["Protein"] = prot
            known = process_group[process_group['conc_numeric'].notnull()].copy()
            known = known[known['Fragment Ion'].isin(replicate_inclusions.get(sample_names[rep], []))]
            unknown = process_group[process_group['conc_numeric'].isnull()].copy()
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
                    sample_col = f"{sample_names[rep]}_{channel}"
                    se_col = f"SE_{sample_names[rep]}_{channel}"
                    if unique_key not in wide_data:
                        wide_data[unique_key] = {}
                    wide_data[unique_key][sample_col] = estimated_conc
                    wide_data[unique_key][se_col] = se_percent
            for grp, vals in exp_group_values.items():
                if len(vals) == 0:
                    continue
                median_val = np.median(vals)
                if len(vals) > 1:
                    combined_se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                else:
                    combined_se = se / slope if slope != 0 else np.nan
                exp_col = f"{sample_names[rep]}_Exp_{grp}"
                se_exp_col = f"SE_{sample_names[rep]}_Exp_{grp}"
                if unique_key in wide_data:
                    wide_data[unique_key][exp_col] = median_val
                    wide_data[unique_key][se_exp_col] = combined_se
                else:
                    wide_data[unique_key] = {exp_col: median_val, se_exp_col: combined_se}
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
                channel = row['Fragment Ion']
                group_label = experiment_group_mapping.get(channel, channel)
                sample_label = f"{sample_names[rep]}_{group_label}"
                regression_info[rep_key]['unknown_points'].append({
                    'sample': sample_label,
                    'channel': channel,
                    'group': group_label,
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
        st.subheader("Step 7: Processed Data (Wide Format)")
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
        st.session_state["wide_df"] = wide_df_edited
    st.subheader("Step 8: Regression Plot Viewer")
    # --- Add selectboxes to filter QC analysis by replicate ---
    if regression_info:
        replicates_qc = sorted({ key.split('_')[0] for key in regression_info.keys() })
        selected_qc_replicate = st.selectbox("Select replicate for QC Analysis", replicates_qc, key="qc_replicate")
        qc_rep_keys = [ key for key in regression_info.keys() if key.startswith(f"{selected_qc_replicate}_") ]
        selected_rep_key = st.selectbox("Select a replicate group", qc_rep_keys, key="qc_rep_group")
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
            label="Download Regression CSV (User-Friendly Format) for this replicate group",
            data=regression_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_rep_key}_regression.csv",
            mime="text/csv"
        )

    st.subheader("Step 9: QC Analysis")
    # --- Add selectboxes to filter QC Analysis by replicate ---
    if regression_info:
        qc_replicates = sorted({ key.split('_')[0] for key in regression_info.keys() })
        selected_qc_rep = st.selectbox("Select replicate for QC Analysis", qc_replicates, key="qc_analysis_replicate")
        # Filter wide_df to only those rows where the Protein_Peptide key starts with the selected replicate.
        qc_wide_df = wide_df[wide_df['Protein_Peptide'].str.startswith(selected_qc_rep)]
        st.write(f"Displaying QC Analysis for replicate: {selected_qc_rep}")
    else:
        qc_wide_df = wide_df.copy()
    with st.expander("View QC Metrics and Plots", expanded=True):
        ##########################################################
        # 1. Custom Percentage Error Distributions (IQR-only)
        ##########################################################
        sample_perc_data = []
        y_positions_perc = {}
        pos = 0
        for col in qc_wide_df.columns:
            if col != "Protein_Peptide" and not col.startswith("SE_"):
                se_col = "SE_" + col
                if se_col in qc_wide_df.columns:
                    measured = pd.to_numeric(qc_wide_df[col], errors='coerce')
                    se_vals = pd.to_numeric(qc_wide_df[se_col], errors='coerce')
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
            fig_perc.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=q1, x1=q3,
                y0=y - 0.3, y1=y + 0.3,
                fillcolor="rgba(100,200,100,0.5)",
                line=dict(width=0),
                layer="below"
            )
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
        ##################################################################
        # Overall Regression Plot with Measured Concentration Distributions
        # (Custom IQR-only Boxplots with Known Calibration Medians, 95% CI,
        #  and an Annotation Showing the Regression Equation with SE)
        ##################################################################
        all_known_x = np.concatenate([info['known_x'] for info in regression_info.values()])
        all_known_y = np.concatenate([info['known_y'] for info in regression_info.values()])
        if len(all_known_x) > 0:
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
            slope_med = np.sum(median_known_x * median_known_y) / np.sum(median_known_x ** 2)
            residuals_med = median_known_y - slope_med * median_known_x
            sigma2_med = np.sum(residuals_med ** 2) / (n_med - 1)
            se_slope_med = np.sqrt(sigma2_med / np.sum(median_known_x ** 2))
            t_val = t.ppf(0.975, n_med - 1)
            error_array = t_val * (median_known_x * np.sqrt(sigma2_med / np.sum(median_known_x ** 2)))
            max_known = np.max(median_known_x)
            sample_iqr_data = []
            y_positions = {}
            pos = 0
            for col in qc_wide_df.columns:
                if col != "Protein_Peptide" and not col.startswith("SE_"):
                    vals = pd.to_numeric(qc_wide_df[col], errors='coerce').dropna()
                    if not vals.empty:
                        q1 = np.percentile(vals, 25)
                        med = np.percentile(vals, 50)
                        q3 = np.percentile(vals, 75)
                        sample_iqr_data.append((col, q1, med, q3))
                        y_positions[col] = pos
                        pos += 1
            if sample_iqr_data:
                max_unknown = max(q3 for (_, _, _, q3) in sample_iqr_data)
            else:
                max_unknown = 0
            max_x_val = max(max_known, max_unknown)
            shared_x_range = [0, max_x_val * 1.05]
            top_x_range = np.linspace(0, max_known, 100)
            y_fit = slope_med * top_x_range
            se_pred = top_x_range * np.sqrt(sigma2_med / np.sum(median_known_x ** 2))
            y_upper = y_fit + t_val * se_pred
            y_lower = y_fit - t_val * se_pred
            fig_reg = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.4],
                subplot_titles=("Overall Regression Curve", "Measured Concentration Distributions (IQR-only)")
            )
            fig_reg.add_trace(go.Scatter(
                x=top_x_range,
                y=y_fit,
                mode='lines',
                name='Overall Regression Line',
                line=dict(color='black', width=2)
            ), row=1, col=1)
            fig_reg.add_trace(go.Scatter(
                x=np.concatenate([top_x_range, top_x_range[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='95% Confidence Interval'
            ), row=1, col=1)
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
            fig_reg.add_annotation(
                x=shared_x_range[1] * 0.75,
                y=y_fit[-1] * 0.95,
                xref="x1",
                yref="y1",
                text=f"y = {slope_med:.2f}x (SE = {se_slope_med:.2f})",
                showarrow=False,
                font=dict(color="blue", size=12)
            )
            for (col, q1, med, q3) in sample_iqr_data:
                y = y_positions[col]
                fig_reg.add_shape(
                    type="rect",
                    xref="x",
                    yref="y2",
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
            fig_reg.update_yaxes(
                tickvals=list(y_positions.values()),
                ticktext=[abbreviate_name(name) for name in list(y_positions.keys())],
                row=2, col=1
            )
            fig_reg.update_xaxes(range=shared_x_range, row=1, col=1)
            fig_reg.update_xaxes(range=shared_x_range, row=2, col=1)
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
        csv_buffer = io.StringIO()
        csv_buffer.write("Calibration Stats\n")
        if len(all_known_x) > 0:
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
        unknown_group_stats = []
        for col in wide_df.columns:
            if col != "Protein_Peptide" and not col.startswith("SE_"):
                group_label = experiment_group_mapping.get(col, col)
                sample = col.split("_")[0]
                vals = pd.to_numeric(wide_df[col], errors='coerce').dropna().values
                if len(vals) > 0:
                    q1 = np.percentile(vals, 25)
                    med = np.percentile(vals, 50)
                    q3 = np.percentile(vals, 75)
                    unknown_group_stats.append({
                        "Sample": sample,
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

# ------------------
# Protein Concentration analysis
# ------------------
elif menu == "Protein Concentration analysis":
    st.header("Protein Concentration Analysis")

    # ---------------------------------------------------------
    # Step 1: Select Wide Table Source
    # ---------------------------------------------------------
    st.markdown("### Step 1: Select Wide Table Source")
    source_option = st.radio("Select the source for the wide table",
                             ["Use processed wide table from previous step", "Upload your own wide table"])
    wide_df = None
    if source_option == "Use processed wide table from previous step":
        if "wide_df" in st.session_state:
            wide_df = st.session_state["wide_df"]
            st.success("Using processed wide table from previous step.")
            st.dataframe(wide_df)
        else:
            st.error("No processed wide table available in session state. Please upload your wide table.")
            uploaded_wide = st.file_uploader("Upload Wide Table CSV", type=["csv"], key="pc_wide_table_upload")
            if uploaded_wide is not None:
                try:
                    wide_df = pd.read_csv(uploaded_wide)
                    st.success("Wide table uploaded successfully.")
                    st.dataframe(wide_df)
                except Exception as e:
                    st.error(f"Error reading uploaded wide table: {e}")
    else:
        uploaded_wide = st.file_uploader("Upload Wide Table CSV", type=["csv"], key="pc_wide_table_upload")
        if uploaded_wide is not None:
            try:
                wide_df = pd.read_csv(uploaded_wide)
                st.success("Wide table uploaded successfully.")
                st.dataframe(wide_df)
            except Exception as e:
                st.error(f"Error reading uploaded wide table: {e}")
    if wide_df is None:
        st.stop()

    # ---------------------------------------------------------
    # Step 1.5: Select Numeric Concentration Columns
    # ---------------------------------------------------------
    st.markdown("### Step 1.5: Select Numeric Concentration Columns")
    # Automatically exclude typical identifier columns and those starting with "SE"
    excluded_columns = {"Protein", "Protein_Peptide", "Identifier"}
    auto_exclude = {col for col in wide_df.columns if col.upper().startswith("SE")}
    excluded_columns = excluded_columns.union(auto_exclude)
    all_columns = list(wide_df.columns)
    # Default selection: all columns not in excluded_columns that can be converted to numeric
    default_numeric_cols = []
    for col in all_columns:
        if col in excluded_columns:
            continue
        try:
            pd.to_numeric(wide_df[col].dropna().iloc[:5])
            default_numeric_cols.append(col)
        except Exception:
            continue
    numeric_cols = st.multiselect("Select the columns containing numeric concentration data",
                                  options=all_columns,
                                  default=default_numeric_cols,
                                  key="pc_numeric_cols")
    if not numeric_cols:
        st.error("Please select at least one numeric concentration column.")
        st.stop()

    # ---------------------------------------------------------
    # Step 2: Group Samples/Replicates and Specify Protein Column
    # ---------------------------------------------------------
    st.markdown("### Step 2: Group Samples/Replicates and Specify Protein Column")
    default_key = "Protein_Peptide" if "Protein_Peptide" in wide_df.columns else all_columns[0]
    key_col = st.selectbox("Select the identifier column", all_columns,
                           index=all_columns.index(default_key))

    # Use only the numeric columns (selected above) as the sample (concentration) columns.
    sample_cols = numeric_cols
    st.write("The following numeric concentration columns will be used for aggregation:")
    st.write(sample_cols)

    # Create a default grouping by splitting each sample column name on '_' and taking the first part.
    default_groups = [col.split('_')[0] for col in sample_cols]
    grouping_df = pd.DataFrame({
        "Column": sample_cols,
        "Group": default_groups
    })
    st.write("Edit sample grouping as needed:")
    edited_grouping_df = st.data_editor(grouping_df, key="pc_grouping_editor")
    group_mapping = dict(zip(edited_grouping_df["Column"], edited_grouping_df["Group"]))
    st.write("Sample to Group Mapping:")
    st.write(group_mapping)

    # Build a dictionary mapping each group to its corresponding columns.
    grouped_cols = {}
    for col, grp in group_mapping.items():
        grouped_cols.setdefault(grp, []).append(col)

    # Next, let the user select which column to use as the protein column.
    if "Protein" in wide_df.columns:
        protein_col = st.selectbox("Select the protein column", all_columns, index=all_columns.index("Protein"))
    else:
        st.info(
            "No dedicated protein column found. The identifier column will be used to derive protein names by splitting on underscores.")
        protein_col = key_col

    # ---------------------------------------------------------
    # Step 3: Calculate Aggregated Protein Concentrations
    # ---------------------------------------------------------
    st.markdown("### Step 3: Calculate Aggregated Protein Concentrations")
    agg_method = st.selectbox("Select aggregation method", ["Median", "Mean"], index=0)
    aggregation_level = st.selectbox("Select aggregation level",
                                     ["Unique proteins (collapse peptides/precursors)",
                                      "Unique identifier (do not collapse)"],
                                     index=0)

    # Compute per-row aggregated values for each group using the selected sample columns.
    agg_intermediate = wide_df[[key_col]].copy()
    for grp, cols in grouped_cols.items():
        if agg_method == "Median":
            agg_intermediate[grp] = wide_df[cols].median(axis=1)
        else:
            agg_intermediate[grp] = wide_df[cols].mean(axis=1)

    # Aggregate by protein if desired.
    if aggregation_level.startswith("Unique proteins"):
        if protein_col != key_col:
            agg_intermediate["Protein"] = wide_df[protein_col]
        else:
            agg_intermediate["Protein"] = agg_intermediate[key_col].astype(str).str.split('_').str[0]
        agg_cols = list(grouped_cols.keys())
        if agg_method == "Median":
            agg_df = agg_intermediate.groupby("Protein")[agg_cols].median().reset_index()
        else:
            agg_df = agg_intermediate.groupby("Protein")[agg_cols].mean().reset_index()
    else:
        agg_df = agg_intermediate.copy()
        agg_df.rename(columns={key_col: "Identifier"}, inplace=True)

    st.write("Aggregated Protein Concentrations:")
    st.dataframe(agg_df)

    st.markdown("#### Summary Statistics")
    if aggregation_level.startswith("Unique proteins"):
        total_entries = agg_df["Protein"].nunique()
        st.write(f"**Total unique proteins identified:** {total_entries}")
    else:
        total_entries = agg_df.shape[0]
        st.write(f"**Total unique identifiers (peptides/precursors) identified:** {total_entries}")
    stats_list = []
    for grp in list(grouped_cols.keys()):
        grp_data = pd.to_numeric(agg_df[grp], errors='coerce').dropna()
        if not grp_data.empty:
            stats_list.append({
                "Group": grp,
                "Min": np.min(grp_data),
                "Median": np.median(grp_data),
                "Mean": np.mean(grp_data),
                "Max": np.max(grp_data)
            })
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        st.dataframe(stats_df)

    st.download_button("Download Aggregated Protein Concentrations CSV",
                       data=agg_df.to_csv(index=False).encode('utf-8'),
                       file_name="aggregated_protein_concentrations.csv",
                       mime="text/csv")

    st.session_state["agg_df"] = agg_df

    # ---------------------------------------------------------
    # Step 4: Histogram Plots for Each Sample Group
    # ---------------------------------------------------------
    st.markdown("### Step 4: Histogram Plots for Each Sample Group")
    for grp in list(grouped_cols.keys()):
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=agg_df[grp], nbinsx=50, name=grp))
        fig_hist.update_layout(
            title=f"Histogram of {grp} Concentrations",
            xaxis_title="Concentration",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_hist)

    # ---------------------------------------------------------
    # Step 5: Volcano Plot Analysis (Requires ≥3 replicates per group)
    # ---------------------------------------------------------
    st.markdown("### Step 5: Volcano Plot Analysis (Requires ≥3 replicates per group)")
    valid_groups = [grp for grp, cols in grouped_cols.items() if len(cols) >= 3]
    if len(valid_groups) < 2:
        st.info("Not enough groups with 3 or more replicates for volcano plot analysis.")
    else:
        group1 = st.selectbox("Select Group 1 for Volcano Plot", valid_groups, key="pc_volcano_group1")
        group2 = st.selectbox("Select Group 2 for Volcano Plot", valid_groups,
                              index=1 if len(valid_groups) > 1 else 0, key="pc_volcano_group2")
        from scipy.stats import ttest_ind

        volcano_data = []
        for idx, row in agg_intermediate.iterrows():
            if aggregation_level.startswith("Unique proteins"):
                if protein_col != key_col:
                    protein_name = wide_df[protein_col].iloc[idx]
                else:
                    protein_name = row[key_col].split('_')[0]
            else:
                protein_name = row[key_col]
            if protein_col != key_col:
                subset = wide_df[wide_df[protein_col].astype(str) == str(protein_name)]
            else:
                subset = wide_df[wide_df[key_col].astype(str).str.split('_').str[0] == protein_name]
            if subset.empty:
                continue
            values1 = pd.to_numeric(subset[grouped_cols[group1]], errors='coerce').values.flatten()
            values2 = pd.to_numeric(subset[grouped_cols[group2]], errors='coerce').values.flatten()
            values1 = values1[~np.isnan(values1)]
            values2 = values2[~np.isnan(values2)]
            if len(values1) >= 3 and len(values2) >= 3:
                mean1 = np.mean(values1)
                mean2 = np.mean(values2)
                fc = np.log2(mean1 / mean2) if mean1 > 0 and mean2 > 0 else np.nan
                try:
                    t_stat, p_val = ttest_ind(values1, values2, equal_var=False)
                except Exception:
                    p_val = np.nan
                volcano_data.append({
                    "Protein": protein_name,
                    "log2FC": fc,
                    "p_value": p_val
                })
        if volcano_data:
            volcano_df = pd.DataFrame(volcano_data)
            volcano_df["minus_log10_p"] = -np.log10(volcano_df["p_value"])
            fig_volcano = go.Figure()
            fig_volcano.add_trace(go.Scatter(
                x=volcano_df["log2FC"],
                y=volcano_df["minus_log10_p"],
                mode="markers",
                marker=dict(color="blue"),
                text=volcano_df["Protein"]
            ))
            fig_volcano.update_layout(
                title=f"Volcano Plot: {group1} vs {group2}",
                xaxis_title="Log2 Fold Change",
                yaxis_title="-Log10(p-value)"
            )
            st.plotly_chart(fig_volcano)
        else:
            st.info("No valid data for volcano plot analysis.")

    # ---------------------------------------------------------
    # Step 6: PCA Plot of Protein Concentrations
    # ---------------------------------------------------------
    st.markdown("### Step 6: PCA Plot of Protein Concentrations")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if aggregation_level.startswith("Unique proteins"):
        pca_data = agg_df.drop(columns=["Protein"]).dropna()
    else:
        pca_data = agg_df.drop(columns=["Identifier"]).dropna()
    if pca_data.shape[0] > 1:
        scaler = StandardScaler()
        pca_scaled = scaler.fit_transform(pca_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_scaled)
        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        if aggregation_level.startswith("Unique proteins"):
            pca_df["Protein"] = agg_df["Protein"].loc[pca_data.index].values
        else:
            pca_df["Identifier"] = agg_df["Identifier"].loc[pca_data.index].values
        label = "Protein" if aggregation_level.startswith("Unique proteins") else "Identifier"
        fig_pca = go.Figure(data=go.Scatter(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            mode="markers+text",
            text=pca_df[label],
            textposition="top center"
        ))
        fig_pca.update_layout(
            title="PCA Plot of Aggregated Protein Concentrations",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% Variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% Variance)"
        )
        st.plotly_chart(fig_pca)
    else:
        st.info("Not enough data for PCA analysis.")
