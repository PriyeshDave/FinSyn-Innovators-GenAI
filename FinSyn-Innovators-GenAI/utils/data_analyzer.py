import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from scipy.stats import norm
import numpy as np
import streamlit as st
import time
from plotly.subplots import make_subplots
import plotly.io as pio




class DataAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm_client = OpenAI(api_key=self.api_key)
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.
        """
        return data.describe(include='all').transpose()
    


    def generate_column_plot_plotly(self, data: pd.DataFrame, column: str, data_type: str) -> str | None:
        """
        Generate a 2x2 subplot with interactive plots for a given numeric column using Plotly.
        Skips non-numeric columns.
        """
        # Skip non-numeric columns
        if not np.issubdtype(data[column].dtype, np.number):
            return None

        # Create a 2x2 subplot layout
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Gaussian Distribution", "Distribution Plot",
            "Box Plot", "Outlier Detection"
        ))

        # Gaussian Distribution Plot (Colored Histogram + Gaussian Fit)
        mean = data[column].mean()
        std = data[column].std()
        x = np.linspace(data[column].min(), data[column].max(), 100)
        y = norm.pdf(x, mean, std)

        fig.add_trace(
            go.Histogram(
                x=data[column],
                histnorm='probability density',
                opacity=0.7,
                name='Data Distribution',
                marker=dict(color='royalblue')  # Ensure color is set explicitly
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Gaussian Fit',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Distribution Plot (Ensure colorful traces)
        dist_plot = px.histogram(data, x=column, marginal="box", nbins=30, color_discrete_sequence=px.colors.qualitative.Set1)
        for trace in dist_plot.data:
            fig.add_trace(trace, row=1, col=2)

        # Box Plot (Add colorful markers)
        box_plot = px.box(data, y=column, color_discrete_sequence=px.colors.qualitative.Set2)
        for trace in box_plot.data:
            fig.add_trace(trace, row=2, col=1)

        # Outlier Detection Plot (Color the box points)
        fig.add_trace(
            go.Box(
                y=data[column],
                boxpoints='all',
                name='Outliers',
                marker=dict(color='purple')  # Ensure outliers are colored
            ),
            row=2, col=2
        )

        # Update layout with colorful theme and improved aesthetics
        fig.update_layout(
            title=f"Plots for {column}",
            height=800,
            width=1000,
            showlegend=False,
            template="plotly",  # Ensure a colorful theme
            font=dict(family="Arial", size=12, color="black")
        )

        # Save HTML with full Plotly JS and style
        path = f"/Users/apple/Documents/Priyesh/Repositories/2025/GrowthHack/FinSyn-Innovators-GenAI/mock/structured_data/insights_plot/{data_type}/"
        image_path = path + column + '.html'

        html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')  # Ensures all styling is included
        with open(image_path, "w") as f:
            f.write(html_content)

        return image_path






    def generate_column_insight(self, column_name: str, stats: pd.Series) -> str:
        """
        Generate insights for a column using LLM.
        """
        with st.spinner(f"Gererating statistical insights for {column_name}..."):
            progress = st.progress(0)
            for i in range(100):  
                time.sleep(0.02) 
                progress.progress(i + 1)
        prompt = f"""
        Analyze the following statistical summary for the column '{column_name}':
        {stats.to_string()}
        
        Provide a concise summary and key insights based on this information.
        Limit the summary with in 50 words.
        """
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    

    def show_plots_and_insights(self, dataset):
        columns = dataset.columns.tolist()
        columns = columns[0:5]
        summary_stats = self.generate_summary_statistics(dataset)
        st.dataframe(summary_stats)
        for column in columns[2:5]:
            st.write(f"### Column: {column}")
            combined_plot = self.generate_column_plot_plotly(dataset, column)

            # Ensure we only show valid plots
            if combined_plot is not None and isinstance(combined_plot, go.Figure):
                st.plotly_chart(combined_plot, use_container_width=True)
            else:
                st.write("No plot available for non-numeric column.")

            column_stats = summary_stats.loc[column]
            column_insight = self.generate_column_insight(column, column_stats)

            st.write("**Insights:**")
            st.info(column_insight)
