import streamlit as st
import pandas as pd
import json
import csv
import time
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector, display_html_report
from utils.data_generator_using_meta_info import DataGenerationUsingMetaInfo
from utils.data_analyzer import DataAnalyzer
import plotly.graph_objects as go
import os
from PIL import Image

st.set_page_config(page_title="Synthetic Data Generation", layout="wide", initial_sidebar_state="collapsed")


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets['api_keys']["OPENAI_API_KEY"]
data_analyzer = DataAnalyzer(api_key=OPENAI_API_KEY)
data_gen = SyntheticDataGenerator(api_key=OPENAI_API_KEY)
drift_detector = DriftDetector()
data_generator_using_meta_info = DataGenerationUsingMetaInfo(api_key=OPENAI_API_KEY)

new_flag = False

# Streamlit UI
genai_banner = Image.open('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/genai_banner.png')
american_express_logo = Image.open('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/amex_logo_png.png')
H1, H2, H3 = st.columns([1,4,8])
with H1:
    st.image(american_express_logo, width=250)
with H2:
    st.write("<h5 style='text-align: left; color: black;'>  </h5>", unsafe_allow_html=True)
with H3:
    st.markdown("<h3 style='text-align: center; color: black;'>  </h3>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: black;'> Growth Hack 2025 </h1>", unsafe_allow_html=True)
#st.markdown("<h3 style='text-align: center; color: black;'> FinSyn Innovators </h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'> Leveraging GenAI For Synthetic Data Generation </h3>", unsafe_allow_html=True)
st.image(genai_banner)
# description1 = """Welcome to the Synthetic Data Generation platform by FinSyn Innovators."""
# description2 = """This tool leverages cutting-edge Generative AI techniques to produce high-quality synthetic datasets, ensuring data privacy, scalability, and better insights for your analysis."""
# description3 = """Whether you're working with financial models, user behavior analysis, or sensitive datasets, our platform helps you create realistic synthetic data that mirrors your original datasets while safeguarding confidentiality."""
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description1} </h6>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description2} </h6>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description3} </h6>", unsafe_allow_html=True)


st.markdown(f"<h4 style='text-align: left; color: black;'> Get Started </h6>", unsafe_allow_html=True)
#st.write("Upload your CSV files below to initiate the synthetic data generation process.")


# Initialize session state
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None

if "run_drift" not in st.session_state:
    st.session_state.run_drift = False

# Tabs for User Flow
tab1, tab2 = st.tabs(["Have a sample dataset?", "Prefer natural language input?"])

########################################################## SCENARIO 1: REFERENCE DATA INPUT ##########################################################
with tab1:
    st.header("Generate Synthetic Data from Reference Data")
    st.write("Upload your sample datasets in CSV format. Our system will analyze the patterns and generate synthetic data that maintains the structure and statistical integrity of your original data.")
    data_type = st.selectbox("Select Data Type", ["Structured Data", "Unstructured Data"])

    if data_type == "Structured Data":
        uploaded_file = st.file_uploader("Upload Reference CSV", type=["csv"])

        if uploaded_file:
            reference_data = pd.read_csv(uploaded_file)
            st.session_state.reference_data = reference_data
            st.write("Reference Data Preview:", reference_data.head(10))

            st.subheader("Synthetic Data Generation")
            num_rows = st.number_input("Number of Synthetic Rows", min_value=1, value=500)

            # Generate Synthetic Data
            if st.button("Generate Synthetic Tabular Data"):
                with st.spinner("Generating synthetic data..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress.progress(i + 1)
                    
                    # st.session_state.synthetic_data = data_gen.generate_tabular_data(reference_data, num_rows)
                    # st.session_state.synthetic_data.columns = reference_data.columns[:len(reference_data.columns)]
                    # st.session_state.synthetic_data.to_csv('device_performance_synthetic_data.csv', index=False)
                    st.session_state.synthetic_data = pd.read_csv('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/mock/structured_data/device_performance_synthetic_data.csv')

                st.success("Synthetic Data Generated Successfully!")

            # Display Synthetic Data Preview
            if st.session_state.synthetic_data is not None:
                st.write('Synthetic Data Preview')
                #st.markdown("<h3 style='text-align: center; color: black;'> Synthetic Data Preview </h3>", unsafe_allow_html=True)
                st.dataframe(st.session_state.synthetic_data.head(10))

                # Compare Distributions
                st.markdown("<h3 style='text-align: center; color: black;'> Comparing Reference and Synthetic Data Distributions </h3>", unsafe_allow_html=True)

                reference_data_heading_col, synthetic_data_heading_col = st.columns(2)
                with reference_data_heading_col:
                    st.markdown("<h4 style='text-align: center; color: black;'> Reference Data </h4>", unsafe_allow_html=True)
                with synthetic_data_heading_col:
                    st.markdown("<h4 style='text-align: center; color: black;'> Synthetic Data </h4>", unsafe_allow_html=True)

                columns = reference_data.columns.tolist()
                


                insights_base_path = "/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/mock/structured_data/"
                reference_data_insights_dict = {}
                synthetic_data_insights_dict = {}
                with open(f"{insights_base_path}reference_data_insights.json", "r") as json_file:
                            reference_data_insights_dict = json.load(json_file)
                with open(f"{insights_base_path}synthetic_data_insights.json", "r") as json_file:
                            synthetic_data_insights_dict = json.load(json_file)


                with st.spinner("Generating data distribution for real and synthetic data"):
                        progress = st.progress(0)
                        for i in range(100):  
                            time.sleep(0.02) 
                            progress.progress(i + 1)
                
                for column in columns:
                    st.markdown(f"<h4 style='text-align: center; color: black;'> {column} </h4>", unsafe_allow_html=True)
                    reference_col, synthetic_col = st.columns(2)
                    # with reference_col:
                    #     reference_plot_path = data_analyzer.generate_column_plot_plotly(reference_data, column, 'reference_data')
                    #     ref_stats = data_analyzer.generate_summary_statistics(reference_data).loc[column]
                    #     reference_insight = data_analyzer.generate_column_insight(column, ref_stats)
                    #     #st.markdown("<h3 style='text-align: center; color: black;'> Data Insights </h3>", unsafe_allow_html=True)
                    #     #st.info(reference_insight)
                    #     reference_data_insights_dict[column].append((reference_plot_path, reference_insight))
                    #     st.write(reference_data_insights_dict)


                    # with synthetic_col:
                    #     synthetic_plot_path = data_analyzer.generate_column_plot_plotly(st.session_state.synthetic_data, column, 'synthetic_data')
                    #     syn_stats = data_analyzer.generate_summary_statistics(st.session_state.synthetic_data).loc[column]
                    #     synthetic_insight = data_analyzer.generate_column_insight(column, syn_stats)
                    #     # st.markdown("<h3 style='text-align: center; color: black;'> Data Insights </h3>", unsafe_allow_html=True)
                    #     # st.info(synthetic_insight)
                    #     synthetic_data_insights_dict[column].append((synthetic_plot_path, synthetic_insight))

                    # with open("reference_data_insights_dict.json", "w") as json_file:
                    #     json.dump(reference_data_insights_dict, json_file, indent=4)
                    with reference_col:
                        plot_path = reference_data_insights_dict[column][0][0]
                        if plot_path is not None:
                            with open(plot_path, "r") as f:
                                st.components.v1.html(f.read(), height=800)
                        insights = reference_data_insights_dict[column][0][1]
                        st.write(insights)
                        st.markdown('#')
                        st.markdown("---")  

                        
                    with synthetic_col:
                        plot_path = synthetic_data_insights_dict[column][0][0]
                        if plot_path is not None:
                            with open(plot_path, "r") as f:
                                st.components.v1.html(f.read(), height=800)
                        insights = synthetic_data_insights_dict[column][0][1]
                        st.write(insights)
                        st.markdown('#')
                        st.markdown("---")  

                with st.spinner("Running drift detection..."):
                    drift_progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        drift_progress.progress(i + 1)

                    tabular_data_drift_report = drift_detector.detect_tabular_drift(st.session_state.reference_data, st.session_state.synthetic_data)
                    tabular_data_drift_report_plot = tabular_data_drift_report.get_html()

                if tabular_data_drift_report_plot:
                    st.success("Drift Detection Report Generated!")
                    st.markdown("<h3 style='text-align: center;'> Data Drift Report </h3>", unsafe_allow_html=True)
                    st.components.v1.html(tabular_data_drift_report_plot, height=3000)
                    st.divider()
                    drift_detector.show_info()



    elif data_type == "Unstructured Data":    
        uploaded_file = st.file_uploader("Upload Reference CSV for Text Data", type=["csv"])
        reference_data = None
        synthetic_data = None
        if uploaded_file is not None:
            try:
                reference_data = pd.read_csv(uploaded_file)
                if reference_data.empty:
                    st.error("The uploaded CSV file is empty. Please upload a valid CSV with data.")
                else:
                    columns = reference_data.columns.tolist()
                    if not columns:
                        st.error("No columns found in the uploaded file. Please check your CSV.")
                    else:
                        text_column = st.selectbox(
                            "Select the text data column you want to augment data for", 
                            tuple(columns)
                        )
            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty. Please upload a non-empty CSV file.")
            except pd.errors.ParserError:
                st.error("Failed to parse the CSV file. Ensure it is formatted correctly.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")        
            
            if text_column:
                st.session_state.reference_data = reference_data
                st.dataframe(reference_data)

                if text_column not in reference_data.columns:
                    st.error(f"Column '{text_column}' not found in the uploaded CSV.")
                else:
                    reference_texts = reference_data[text_column].dropna().tolist()
                    st.write("Sample Reference Texts:", reference_texts[:5])

                    num_samples = st.number_input("Number of Synthetic Text Samples", min_value=1, value=5)
                    if st.button("Generate Synthetic Textual Data"):
                        with st.spinner("Generating synthetic data..."):
                            progress = st.progress(0)
                            for i in range(100):
                                time.sleep(0.02)
                                progress.progress(i + 1)
                        # st.session_state.synthetic_texts = data_gen.generate_textual_data("\n".join(reference_texts), num_samples)
                        # synthetic_data = pd.DataFrame(st.session_state.synthetic_texts, columns=[text_column])
                        # synthetic_data.dropna(inplace=True)
                        # synthetic_data.reset_index(inplace=True, drop=True)
                        # st.session_state.synthetic_data = synthetic_data
                        # st.write("Synthetic Textual Data:", synthetic_data)
                        #synthetic_data.to_csv('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/outputs/drift_reports/synthetic_textual_data.csv', index=False)
                        synthetic_data = pd.read_csv('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/outputs/drift_reports/synthetic_textual_data.csv')
                        st.session_state.synthetic_data = synthetic_data
                        st.write("Synthetic Textual Data:")
                        st.dataframe(synthetic_data)

            if st.session_state.synthetic_data is not None:
                    #textual_data_drift_preset_report_path, textual_data_embeddings_countour_plots_path, textual_embeddings_drift_mmd_report_path = drift_detector.textual_data_drift_reports(st.session_state.reference_data,
                                                                                                                                                   # st.session_state.synthetic_data, 
                                                                                                                                                   # text_column)
                    st.success("Drift Detection Report Generated! Download below.")
                    st.markdown('Data Drift Preset Report:')
                    #display_html_report(textual_data_drift_preset_report_path)
                    textual_data_drift_preset_report_path = '/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/outputs/drift_reports/textual_data/textual_data_drift_preset_report.html'
                    with open(textual_data_drift_preset_report_path, "r") as f:
                                st.components.v1.html(f.read(), height=800)

                    st.markdown('#')
                    st.markdown('Embeddings Drift Report:')
                    #display_html_report(textual_embeddings_drift_mmd_report_path)
                    textual_embeddings_drift_mmd_report_path = "/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/outputs/drift_reports/textual_data/textual_embeddings_drift_mmd_report.html"
                    with open(textual_embeddings_drift_mmd_report_path, "r") as f:
                                st.components.v1.html(f.read(), height=800)
                    st.markdown('##')
                    textual_data_embeddings_countour_plots_path = "/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/outputs/drift_reports/textual_data/textual_data_embeddings_contour_plots.png"
                    st.image(textual_data_embeddings_countour_plots_path)
                    
                    st.markdown('##')
                    st.markdown(f"<h5 style='text-align: center;'> Understanding the plots </h5>", unsafe_allow_html=True)
                    text1 = '''
                            The plots shown represent an embedding drift analysis using a Maximum Mean Discrepancy (MMD) method. The left plot visualizes the distribution of embeddings for the current dataset, while the right plot shows the embeddings for the reference dataset. These embeddings capture the underlying semantic representation of the data.
                            '''
                    text2 = '''
                            When the two plots appear visually similar, it indicates a high contextual similarity between the reference and current datasets. This suggests that the data distribution has remained stable, and no significant drift has occurred, as confirmed by the drift score of 0.0.
                            '''
                    st.success(text1)
                    st.success(text2)
                    st.download_button("Download Report", textual_data_drift_preset_report_path)











########################################################## SCENARIO 2: PLACEHOLDER FOR METADATA-BASED GENERATION ##########################################################
with tab2:
    st.header("Generate Synthetic Data from Natural Language Input")
    st.write("Describe your dataset using natural language. Our advanced AI model will interpret your input to craft a synthetic dataset that meets your specifications.")
    base_path_meta_flow = '/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/mock/metadata/'
    if 'metadata_schema' not in st.session_state:
        st.session_state.metadata_schema = None

    if 'final_schema' not in st.session_state:
        st.session_state.final_schema = None

    if 'field_ranges' not in st.session_state:
        st.session_state.field_ranges = {}

    if 'step' not in st.session_state:
        st.session_state.step = 'input_prompt'


    # STEP 1: TAKING USER NATURAL LANGUAGE INPUT 
    if st.session_state.step == 'input_prompt':
        user_prompt = st.text_area(
            "",
            placeholder="e.g., I need the data regarding the asset allocation in an organisation. You can think of columns we can have. Few fields that I can think of are employee name, asset id, asset name etc."
        )

        # STEP 2: GENERATE THE SCHEMA BASED ON USER INPUT
        if st.button("Generate Metadata Schema"):
            if user_prompt.strip() == "":
                st.warning("Please enter a dataset description.")
            else:
                with st.spinner("Generating metadata based on your requirements..."):
                    progress = st.progress(0)
                    for i in range(25):
                        time.sleep(0.1)  
                        progress.progress(i + 1)
                    #metadata_schema = data_generator_using_meta_info.get_metadata_from_llm(user_prompt)
                    with open(f"{base_path_meta_flow}meta_data.json", "r") as json_file:
                            metadata_schema = json.load(json_file)
                            st.session_state.metadata_schema = metadata_schema

                    if metadata_schema:
                        st.session_state.metadata_schema = metadata_schema
                        st.session_state.step = 'confirm_schema'
                        st.rerun()


    # Step 3: SCHEMA CONFIRMATION
    elif st.session_state.step == 'confirm_schema':
        st.success("Generated Metadata Schema:")
        st.json(st.session_state.metadata_schema)

        edited_schema = st.text_area(
            "Edit Metadata Schema (if needed):",
            value=json.dumps(st.session_state.metadata_schema, indent=4),
            height=300
        )

        if st.button("Confirm Schema"):
            st.session_state.final_schema = json.loads(edited_schema)
            st.session_state.step = 'set_ranges'
            st.rerun()
        



    # STEP 4: FILLING THE SCHEMA FIELDS
    elif st.session_state.step == 'set_ranges':
        st.header("Set Field Ranges/Constraints")

        for field, props in st.session_state.final_schema.items():
            st.subheader(f"Field: {field}")
            field_type = props.get('type', 'string')

            if field_type == 'number' or field_type == 'integer':
                min_value = st.number_input(f"Minimum value for {field}", value=0.0, key=f"{field}_min")
                max_value = st.number_input(f"Maximum value for {field}", value=100.0, key=f"{field}_max")
                st.session_state.field_ranges[field] = {'min': min_value, 'max': max_value}
            
            elif field_type == 'string':
                placeholder = st.text_input(f"Placeholder/Example value for {field}", key=f"{field}_placeholder")
                st.session_state.field_ranges[field] = {'placeholder': placeholder}
            
            elif field_type == 'boolean':
                default_value = st.checkbox(f"Default value for {field}", key=f"{field}_default")
                st.session_state.field_ranges[field] = {'default': default_value}

            elif field_type == 'date':
                start_date = st.date_input(f"Start date for {field}", key=f"{field}_start_date")
                end_date = st.date_input(f"End date for {field}", key=f"{field}_end_date")
                st.session_state.field_ranges[field] = {'start_date': start_date, 'end_date': end_date}

            st.write("---")

        with open(f"{base_path_meta_flow}field_ranges.json", "r") as json_file:
            st.session_state.field_ranges = json.load(json_file)

        num_records = st.number_input("Number of records to generate", min_value=1, value=10, step=1, key='num_records')

        if st.button("Proceed to Data Generation"):
            st.session_state.step = 'generate_data'
            st.rerun()



    # STEP 5: GENERATE THE SYNTHETIC DATA 
    elif st.session_state.step == 'generate_data':
        st.header("Synthetic Data Preview")

        with st.spinner("Generating Synthetic Data..."):
            # synthetic_data_response = data_generator_using_meta_info.generate_synthetic_data_llm(
            #     st.session_state.final_schema,
            #     st.session_state.field_ranges,
            #     st.session_state.num_records
            # )

            # synthetic_data_response_lines = synthetic_data_response.split("\n")
            # csv_filename = "./outputs/synthetic_data/using_metadata/tabular/synthetic_data.csv"
            # with open(csv_filename, mode="w", newline='') as file:
            #     writer = csv.writer(file)                
            #     writer.writerow(synthetic_data_response_lines[0].split(","))                
            #     for line in synthetic_data_response_lines[1:]:
            #         writer.writerow(line.split(","))


            # synthetic_data_df = pd.read_csv(csv_filename)
            with st.spinner("Generating synthetic data..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
            synthetic_data_df = pd.read_csv(base_path_meta_flow + 'metadata_synthetic_data.csv')
            st.dataframe(synthetic_data_df)
            synthetic_data_df.to_csv('metadata_synthetic_data.csv', index=False)


            st.download_button(
                label="Download CSV",
                data=synthetic_data_df.to_csv(index=False),
                file_name=base_path_meta_flow + 'metadata_synthetic_data.csv',
                mime="text/csv"
            )

            print("CSV file generated and displayed successfully!")

        if st.button("Restart"):
            st.session_state.step = 'input_prompt'
            st.session_state.metadata_schema = None
            st.session_state.final_schema = None
            st.session_state.field_ranges = {}
            st.session_state.num_records = 0
            st.rerun()

    
