# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !java -version

import pandas as pd
import xlsxwriter
import os
from os import listdir
from pyprojroot import here
import re
from tqdm import tqdm
from tabula.io import read_pdf
from pdfminer.high_level import extract_text, extract_pages
import shutil
import numpy as np

# Execute for supressing warnings 
import warnings
warnings.filterwarnings('ignore')
