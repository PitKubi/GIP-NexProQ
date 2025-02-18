import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
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

st.title("GIP - NexProQ: Data extraction and analysis  pipeline")
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
					st.success("Data forwarded to processing. Please switch to the Process Data page.")
		else:
			st.info("Please upload an input CSV to generate a template.")

# ------------------
# Process Data
# ------------------
elif menu == "Process Data":
	st.header("Process Data")
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

	# ===== Experiment Grouping for Unknown Channels =====
	unknown_channels = template_df.loc[template_df['conc'].isnull(), 'Label'].tolist()
	if unknown_channels:
		# Create a default DataFrame from unknown channels.
		unknown_df = pd.DataFrame({'Channel': unknown_channels})
		unknown_df['Experiment Group'] = unknown_df['Channel']

		# If a mapping exists, pre-populate the table.
		if "experiment_group_mapping" in st.session_state and st.session_state["experiment_group_mapping"]:
			mapping = st.session_state["experiment_group_mapping"]
			unknown_df['Experiment Group'] = unknown_df['Channel'].apply(lambda ch: mapping.get(ch, ch))

		with st.expander("Group Unknown TMT Channels into Experiment Groups", expanded=False):
			edited_unknown_df = st.data_editor(unknown_df, key="unknown_groups", use_container_width=True)
			if st.button("Apply Grouping Changes", key="apply_grouping"):
				new_mapping = dict(zip(edited_unknown_df['Channel'], edited_unknown_df['Experiment Group']))
				# Exclude channels with blank or "delete" values.
				new_mapping = {ch: grp for ch, grp in new_mapping.items() if
				               grp.strip() and grp.strip().lower() != "delete"}
				st.session_state["experiment_group_mapping"] = new_mapping
				st.session_state["grouping_committed"] = True
			# If the grouping has been committed, show a persistent success message.
			if st.session_state.get("grouping_committed", False):
				st.success("Experiment grouping updated.")
		experiment_group_mapping = st.session_state.get("experiment_group_mapping",
		                                                dict(zip(edited_unknown_df['Channel'],
		                                                         edited_unknown_df['Experiment Group'])))
	else:
		experiment_group_mapping = {}
		st.session_state["experiment_group_mapping"] = experiment_group_mapping
	# ===== End of Experiment Grouping =====

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

	# ===== Replicate-Specific Calibration Point Selection =====
	replicate_inclusions = {}
	for rep in unique_reps:
		rep_df = filtered_df[filtered_df['Replicate'] == rep].copy()
		rep_df['conc_numeric'] = pd.to_numeric(rep_df['conc'], errors='coerce')
		available_labels = sorted(rep_df.loc[rep_df['conc_numeric'].notnull(), 'Fragment Ion'].unique())
		# Retrieve previous selection if exists (keyed by the renamed sample name)
		default_selection = st.session_state.get("replicate_inclusions", {}).get(sample_names[rep], available_labels)
		selection = st.multiselect(
			f"Select TMT labels to include for sample {sample_names[rep]} (calibration points)",
			options=available_labels,
			default=default_selection,
			key=f"include_{rep}"
		)
		replicate_inclusions[sample_names[rep]] = selection
	st.session_state["replicate_inclusions"] = replicate_inclusions
	# ===== End of Calibration Point Selection =====

	# --- Regression Processing (Per Replicate, Protein, Peptide) ---
	wide_data = {}  # Final wide-format table keyed by Protein_Peptide.
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
			se_percent = (se / estimated_conc) * 100 if (
						estimated_conc not in [0, np.nan] and not np.isnan(estimated_conc)) else np.nan
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
				# Instead of using 'se' (transition error), use error in estimated concentration.
				combined_se = se / slope if slope != 0 else np.nan
			exp_col = f"{sample_name}_Exp_{grp}"
			se_exp_col = f"SE_{sample_name}_Exp_{grp}"
			if protein_peptide not in wide_data:
				wide_data[protein_peptide] = {}
			wide_data[protein_peptide][exp_col] = median_val
			wide_data[protein_peptide][se_exp_col] = combined_se

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
			# Compute the error in estimated concentration as (se / slope)
			conc_error = se / slope if slope != 0 else np.nan
			sample_col = f"{sample_name}_{row['Fragment Ion']}"
			regression_info[rep_key]['unknown_points'].append({
				'sample': sample_col,
				'estimated_conc': estimated_conc,
				'conc_error': conc_error,
				'transition_result': trans_result
			})

	wide_df = pd.DataFrame.from_dict(wide_data, orient='index')
	wide_df.index.name = 'Protein_Peptide'
	wide_df.reset_index(inplace=True)

	if wide_df.empty:
		st.info(
			"No valid regression groups found. Please verify that your template CSV provides valid known concentrations for at least two rows per group, and that you have selected calibration points for each replicate.")
	else:
		st.subheader("Processed Data (Wide Format)")
		#st.dataframe(wide_df)

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

		wide_filename = st.text_input("Enter file name for the processed wide CSV:", value="wide_output.csv",
		                              key="wide_filename")
		wide_csv = wide_df_edited.to_csv(index=False).encode('utf-8')
		st.download_button(
			label="Download Processed Wide CSV",
			data=wide_csv,
			file_name=wide_filename,
			mime="text/csv"
		)

	# --- Regression Plot Section ---
	if regression_info:
		st.subheader("Regression Plot Viewer")
		selected_rep_key = st.selectbox("Select a replicate sample to view its regression curve",
		                                list(regression_info.keys()))
		details = regression_info[selected_rep_key]
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
			legend=dict(
				orientation="h",
				yanchor="top",
				y=-0.2,
				xanchor="center",
				x=0.5
			)
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
		if rows:
			regression_df = pd.DataFrame(rows)
		else:
			regression_df = pd.DataFrame()
		st.download_button(
			label="Download Regression CSV (User-Friendly Format) for this replicate sample",
			data=regression_df.to_csv(index=False).encode('utf-8'),
			file_name=f"{selected_rep_key}_regression.csv",
			mime="text/csv"
		)
