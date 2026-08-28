import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
def load_data(path):
    df = pd.read_csv(path)
    return df
file_name_no_ext = os.path.splitext(os.path.basename(__file__))[0]
os.makedirs(f'pictures/{file_name_no_ext}', exist_ok=True)