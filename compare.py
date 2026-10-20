"""
Compares counts of the ratings per category for the baseline and for each
model. Also compares how often refinement changed a correct translation by over
editing it. 

"""

import pandas as pd

blind_rating = pd.read_csv('data/blind_rating.csv')
blind_key = pd.read_csv('data/blind_key.csv')

merged = pd.merge(blind_key, blind_rating, on = ["id", "option"])

print(merged.columns)