import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image

# Set page config
st.set_page_config(page_title="Synthetic Data Generation", layout="wide", initial_sidebar_state="collapsed")

# american_express_logo = Image.open('/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/amex_logo_png.png')
# H1, H2, H3 = st.columns([1,4,8])
# with H1:
#     st.image(american_express_logo, width=250)
# with H2:
#     st.write("<h5 style='text-align: left; color: black;'>  </h5>", unsafe_allow_html=True)
# with H3:
#     st.markdown("<h3 style='text-align: center; color: black;'>  </h3>", unsafe_allow_html=True)

# Display an image
#st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/genai_banner.png", use_container_width=True)
st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/FinSyn Innovators.png", use_container_width=True)

# Define card layout
col1, col2 = st.columns(2)

# with col1:
#     st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/genai.png", use_container_width=True)
#     st.markdown("""
#         <h5 style='text-align: center; color: black;'> 
#             Generates synthetic data by extracting statistical summaries and sending them to the LLM for public data generation. 
#         </h5>
#         <div style="display: flex; justify-content: center; margin-top: 20px;">
#             <form action="/Leveraging_GenAI" target="_self">
#                 <input type="submit" value="Leveraging GenAI" style="padding: 8px 20px; font-size: 16px; border: none; border-radius: 8px; background-color: #F0F0F5; cursor: pointer;">
#             </form>
#         </div>
#     """, unsafe_allow_html=True)

    

# with col2:
#     st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/neural_network.png", use_container_width=True)
#     st.markdown("""
#         <h5 style='text-align: center; color: black;'> 
#             Utilizes a fine-tuned neural network (GAN networks) to generate synthetic data without exposing sensitive information 
#         </h5>
#         <div style="display: flex; justify-content: center;">
#             <form action="/Using_Neural_Networks" target="_self">
#                 <input type="submit" value="Leveraging GANs" style="padding: 8px 20px; font-size: 16px; border: none; border-radius: 8px; background-color: #F0F0F5; cursor: pointer;">
#             </form>
#         </div>
#     """, unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/genai.png", use_container_width=True)
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
            <h5 style="text-align: center; color: black;">
                Generates synthetic data by extracting statistical summaries and sending them to the LLM for public data generation.
            </h5>
            <form action="/Leveraging_GenAI" target="_self">
                <input type="submit" value="Leveraging GenAI" style="padding: 8px 20px; font-size: 16px; border: none; border-radius: 8px; background-color: #F0F0F5; cursor: pointer;">
            </form>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/ui/neural_network.png", use_container_width=True)
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
            <h5 style="text-align: center; color: black;">
                Utilizes a fine-tuned neural network (GAN networks) to generate synthetic data without exposing sensitive information.
            </h5>
            <form action="/Using_Neural_Networks" target="_self">
                <input type="submit" value="Leveraging GANs" style="padding: 8px 20px; font-size: 16px; border: none; border-radius: 8px; background-color: #F0F0F5; cursor: pointer;">
            </form>
        </div>
    """, unsafe_allow_html=True)

