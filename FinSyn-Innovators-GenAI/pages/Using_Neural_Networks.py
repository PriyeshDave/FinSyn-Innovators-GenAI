import streamlit as st
import pandas as pd
import time
from utils.ai_ml.SyntheticDataGeneratorRCTGAN import SyntheticDataGeneratorRCTGAN
from PIL import Image


st.set_page_config(page_title="Synthetic Data Generation", layout="wide", initial_sidebar_state="collapsed")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Streamlit UI
genai_banner = Image.open('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/genai_banner.png')
american_express_logo = Image.open('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/amex_logo_png.png')
H1, H2, H3 = st.columns([1,4,8])

with H1:
    st.image(american_express_logo, width=250)
with H2:
    st.write("<h5 style='text-align: left; color: black;'>  </h5>", unsafe_allow_html=True)
with H3:
    st.markdown("<h3 style='text-align: center; color: black;'> </h3>", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: black;'> Growth Hack 2025 </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'> Leveraging Neural Networks For Synthetic Data Generation </h3>", unsafe_allow_html=True)
st.image(genai_banner)

# Description Section
# description1 = "Welcome to the Synthetic Data Generation platform by FinSyn Innovators."
# description2 = "Our platform uses cutting-edge AI models, including internally trained GANs, to create high-fidelity synthetic datasets."
# description3 = "We prioritize data privacy by ensuring that all data is processed securely on internally hosted models, giving you complete control and confidentiality."
# description4 = "Whether you're working with financial models, customer insights, or sensitive datasets, our GAN-based approach provides realistic synthetic data that replicates the statistical properties of your original dataset."

# st.markdown(f"<h6 style='text-align: center; color: black;'> {description1} </h6>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description2} </h6>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description3} </h6>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='text-align: center; color: black;'> {description4} </h6>", unsafe_allow_html=True)

st.markdown(f"<h4 style='text-align: left; color: black;'> Get Started </h4>", unsafe_allow_html=True)

# Upload multiple CSV files (folder)
uploaded_files = st.file_uploader("Upload your CSV files below to initiate the synthetic data generation process.", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Load files into a dictionary {filename (without .csv): dataframe}
    num_rows = st.number_input("Number of Synthetic Rows", min_value=1, value=500)

    data_dict = {file.name.replace(".csv", ""): pd.read_csv(file) for file in uploaded_files}
    
    # Remove columns containing 'date'
    data_dict = {name: df.loc[:, ~df.columns.str.contains("date", case=False)] for name, df in data_dict.items()}
    
    st.write(f"Loaded {len(data_dict)} CSV files successfully.")

    if st.button("Generate Synthetic Data"):
        st.write("Generating synthetic data...")
        
        # Initialize and generate synthetic data
        generator = SyntheticDataGeneratorRCTGAN()
        generator.real_data = data_dict  # Passing real data
        synthetic_data = generator.generate_synthetic_data()

        with st.spinner("Generating synthetic data..."):
            progress = st.progress(0)
            for i in range(100):  
                time.sleep(0.02) 
                progress.progress(i + 1)

        st.write("Synthetic data generated successfully!")

        # Display real and synthetic data side by side
        for key in data_dict.keys():  # Iterate over uploaded file names
            if key in synthetic_data:  # Ensure synthetic data exists for the key
                #st.markdown(f"### Table Name: {key}")
                st.markdown(f"<h3 style='text-align: center; color: black;'> Table Name: {key} </h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<h5 style='text-align: center; color: black;'> Real Data </h5>", unsafe_allow_html=True)
                    st.dataframe(data_dict[key])
                with col2:
                    st.markdown(f"<h5 style='text-align: center; color: black;'> Test Data </h5>", unsafe_allow_html=True)
                    st.dataframe(synthetic_data[key])
                
                table_evaluator_results_path = f'/Users/apple/Documents/Priyesh/Repositories/2025/Synthetic-Data-Generation/outputs/table_evaluator_results/regulatory_reporting/{key}/'
                for i in range(1, 6):
                    st.image(f'{table_evaluator_results_path}{key}_{i}.png')
                st.markdown("---")