from numpy.core.numeric import indices
import pandas as pd
import numpy as np
import streamlit as st
from utils import tfidf_matrix, indices, titles

st.title('Anime recommendation')

# option = st.selectbox(
#     'How would you like to be contacted?',
#    ('Email', 'Home phone', 'Mobile phone'))


option = st.selectbox(
    'How would you like to be contacted?',
   titles)
st.write('You eselected:', option)
