"""
data_transformation

The package creates the final dataset.

The dataset creation starts from the orginal extracted data of the data_extraction package.
Then, The metadata and event data are processed and classified separately to generate the credit dataset.
The credit dataset is combined with the firm performance and relationship data to get the final dataset.


Examples:
    You can use this package to perform various tasks involved in fina data setup, such as:
    - Data exploration & analysis
    - Data processing
    - theme classification
    - Dataset creation
    
Contents:
    - data_explore_on_mothers: A notebook to explore the original data before processing.
    - utils: Provide utility functions for the package.
    - split_events: Split and pre-processing metadata and event data.
    - event1_pemmit_processing: Processing event permit.
    - event1_penalty_processing: Processing event penalty.
    - event2_processing: Processing event 2 redlists.
    - event3_processing: Processing event 3 blacklists.
    - event4_processing: Processing event 4 watchlists.
    - event5_processing: Processing event 5 commitments.
    - theme_classification: Provide theme classification.
    - add_themes: Add predicted themes to unseen observations.
    - event_stat: Provide statistical analysis for the credit dataset
    - pipeline: Provide all the functions to create the final dataset & Internal consistency check.
    - create_dataset: Provides functions to create the final dataset and sample data.
"""
