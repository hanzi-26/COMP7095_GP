# Group 8: Traffic Accident Rate Prediction Project

## Overview

This project aims to predict the traffic accident rate for the year 2025 and to perform a detailed correlation analysis of various factors influencing traffic safety. Our approach leverages big data technologies, distributed computing, and machine learning (using PySparkML) to handle large-scale datasets and derive actionable insights. The project encompasses comprehensive data cleaning, transformation, visualization, and predictive analytics.

## Repository Structure

- **preprocess.py**  
  Contains scripts for data ingestion and cleaning, including:
  - Collecting datasets
  - Establishing guidelines for handling missing values
  - Verifying the integrity of merged datasets

- **predict.py**  
  Implements the machine learning model using PySparkML to predict the 2025 traffic accident rate.

- **visual.py**  
  Responsible for generating data visualizations that display both the correlation analysis results and the predictive outcomes.

- **correlation.py**  
  Focuses on conducting a detailed correlation analysis between multiple traffic-related variables.

- **image.png**  
  Contains sample visualizations or diagrams that illustrate the project's architecture and data processing workflow.

## Team Members and Contributions

- **Han LI**  
  - Collected the datasets.
  - Led data ingestion and cleaning tasks.
  - Established guidelines for addressing missing values.
  - Verified the integrity of the final merged datasets.
  - Utilized PySparkML for correlation analysis and predicting the 2025 traffic accident rate.
  - Produced data visualizations.
  - Created this README file to clearly document the group's contributions and project structure.

- **Yi HUANG**  
  - Collected additional datasets.
  - Initiated the project process.
  - Tested the HDFS environment using Linux systems.
  - Designed and implemented the pipeline architecture.
  - Developed core project algorithms.

- **Yifan LI**  
  - Configured and managed the Hadoop Distributed File System (HDFS) environment.
  - Coordinated with containerization technologies to ensure a smooth, production-like environment.
  - Oversaw the harmonious integration of data ingestion, transformation, and modeling components.

- **Huanhua LIN**  
  - Authored the project report.
  - Prepared the presentation materials.

- **Shiwei DENG**  
  - Led the development of data visualization and presentation.

## How to Run the Project

1. **Environment Setup**  
   - Ensure all required dependencies are installed, including Python, PySpark, Hadoop, and any necessary libraries for data visualization.
   - Configure your Linux environment to support HDFS and containerization if deploying in a distributed setup.

2. **Data Ingestion and Preprocessing**  
   - Run the `preprocess.py` script to ingest and clean the datasets.
   - Confirm that datasets are merged correctly and missing values are handled as per the established guidelines.

3. **Prediction and Analysis**  
   - Execute `predict.py` to run the machine learning model and predict the 2025 traffic accident rate.
   - Use `correlation.py` to perform the correlation analysis among different variables.

4. **Visualization**  
   - Run `visual.py` to generate visualizations that clearly display the outcomes of the correlation analysis and prediction model.

5. **Review Documentation**  
   - Refer to `image.png` for sample diagrams and visualizations that provide an overview of the project architecture and data flow.

## Requirements

- Python 3.x
- PySpark
- Hadoop (properly configured on a Linux system)
- Additional Python libraries such as pandas and matplotlib for data handling and visualization

## Contact Information

For further details or any inquiries about the project, please reach out via our Group 8 collaboration platform or contact the lead coordinator via email.
