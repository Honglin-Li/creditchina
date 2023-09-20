# Introduction
The project is aligned with the thesis titled “China’s Social Credit System and Its Impact on Firm Behavior and Performance.” This project consists of two main parts:

- Creating a Social Credit Dataset

In the first part of the project, I focus on the creation of a comprehensive social credit dataset. This dataset forms the foundation for my research and analysis, and it allows me to examine various aspects of China’s Social Credit System.

- Investigating the Impact of Credit on Firm Performance

The second part of the project involves an in-depth investigation into how credit within the Social Credit System affects firm behavior and performance. I employ Python for data setup and analysis and Stata to run regression analyses.

My project is designed to provide a holistic view of China’s Social Credit System and its implications on businesses. To learn more about specific details and findings, please refer to the presentation slides available in the “creditchina” directory.

Thank you for exploring our project’s documentation. I hope you find it informative and insightful.

Usage
==================


Running the Code
----------------

1. Open a command prompt (cmd) in the `creditchina` directory.

2. Activate the virtual environment by running the following command: 

  ``conda activate ./env``

3. If you need to use Jupyter Notebook, you need to change the data_path in config.py to absolute path, then run: 

  ``jupyter-notebook``
	
4. To perform specific tasks, use the following commands:

- To create the Credit dataset, run:

  ``python cli.py createdb``

- For Event analysis generation, execute:

  ``python cli.py analysis``

- To set up Regression data, use either of the following commands:

  - To calculate firm performance growth with a specific year difference and consider relationships between parent and subsidiary companies, run:

  ``python cli.py regdb -y 1 -r True``

  - For default settings (without specifying year difference and relationships), use:

  ``python cli.py regdb``

For the `regdb` command, the `-y` argument is used to specify the year difference for calculating firm performance growth, and the `-r` argument determines whether relationships between parent and subsidiary companies are considered. Use `-r True` to include relationships.

5. To run the regressions, you need to open the Dofile-cpr in data/processed_data/regression_data/ directory and run the code.



Source Code Structure
---------------------

The source code is organized in the `src` directory with the following subdirectories, each containing both `.py` and `.ipynb` versions:

1. `data_acquisition`: Code for downloading credit reports (PDF).

2. `data_extraction`: Code to extract data from downloaded PDFs.

3. `data_transformation`: Code to create the Credit dataset from extracted data.

4. `data_mining`: Code to set up regression data and perform regressions.

In addition to the subdirectories, there are two Python files:

- `config.py`: Contains all configuration variables. **You need to write the right path for the data directory in the data_path variable first if you want to run the Jupyter Notebook.**

- `cli.py`: Contains code for enabling command-line commands.



Data Structure
--------------

All data is stored in the `data` directory, which has the following structure:


**original/**: Stores the original credit reports and extracted data

   - `original/mothers_original.xlsx`: Extracted data of parent companies.

   - `original/daughters_original.xlsx`: Extracted data of subsidiary companies.

   - `original/all_companies_original.xlsx`: Combination of mothers_original and daughters_original. **(Please DO NOT re-extract these three files as they contain manual data corrections.)**

   - `original/daughters/`: Stores batch results of daughters_original data extraction (due to multiprocessing).

**stat/**: Contains statistical analysis tables and figures.

   - `stat/corr/`: Correlation matrix figures of firm performance.

**models/**: Stores trained models for theme classification.

**data_explore/**: Contains initial data exploration files before data processing.

**company_list/**: Stores company name lists of parent and subsidiary companies.

**company_attributes/**: Contains the basis of company and credit event classifications.

**clean_sub_events/**: Stores pre-processed and split metadata and event data.

**processed_data/**: The most important directory, containing the final dataset:

   - `processed_data/events/`: Processed event data.

   - `processed_data/meta_events/`: Processed event data with categories (e.g., penalty type and theme)
   
   - `processed_data/performance_panel/`: The dataset combined with credit, performance, and relationship.

   - `processed_data/regression_data/`: The data for regressions

