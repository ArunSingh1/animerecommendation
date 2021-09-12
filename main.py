from numpy.core.numeric import indices
import pandas as pd
import numpy as np
import streamlit as st
from utils import tfidf_matrix, indices, titles, content_recommender, df
import base64


#st.title('Anime recommendation')

#st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-family: 'Lobster', cursive; serif; font-size:75px !important;
}

# </style>
# """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Anime recommendation</p>', unsafe_allow_html=True)


##background iamge 
# import base64

# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

# set_png_as_page_bg('background.png')


# title1 = titles[:100]

option = None
option = st.selectbox(
    'Choose your favorite',
   titles,index=0,key = None)
st.write('Recommends Top 10 animes with similar story lines')


if option is not None:

    prediction = content_recommender(option, tfidf_matrix=tfidf_matrix, df=df, indices=indices)

    for index, row in prediction.iterrows():
        #print(row['title']),
        st.subheader(row['title'])
        #print(row['synopsis'])
        # st.write('Synopsis')
        st.write('Synopsis:::', """:book:""",  row['synopsis'])
        st.write('Genres:::', """:musical_score:""",  row['genres'], " | ", 'Popularity:::', """:thumbsup:""", row['popularity'])
        st.write('Rating:::', """:loudspeaker:""", row['rating'], " | ",""":movie_camera:""", row['type']," | ", row['status'])

else:
    pass



# col1, col2, col3 = st.beta_columns([1,6,1])

# with col1:
#     st.write("")

# with col2:
#     st.image("https://i.imgflip.com/amucx.jpg")

# with col3:
#     st.write("")

# url = "https://unsplash.com/photos/nKO_1QyFh9o"
# #"https://images.unsplash.com/photo-1542281286-9e0a16bb7366"
# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)



# option = st.selectbox(
#     'How would you like to be contacted?',
#    ('Email', 'Home phone', 'Mobile phone'))

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded


# header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#     img_to_bytes("header.png")
# )
# st.markdown(
#     header_html, unsafe_allow_html=True,
# )