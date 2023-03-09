
import pandas as pd
import numpy as np
import streamlit as st

import plotly.express as px


df = pd.DataFrame()
# Load the raw data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
try:
    raw_data = df

    # Clean and transform the data
    pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    cols_to_drop = ['Intensive English 0- Twinning Program',
        'Intensive English 01- Twinning Program',
        'Intensive English 02- Twinning Program',
        'Intensive English 03- Twinning Program',
        'Intensive English 1- Twinning Program',
        'Intensive English 2- Twinning Program',
        'Intensive English 3- Twinning Program', 'Listening & Speaking IE1',
        'Listening & Speaking IE2',
        'Listening & Speaking IE2 (for twinning program)','Physical Training 1', 'Physical Training 2', 'Reading & Writing IE1',
        'Reading & Writing IE2', 'Reading & Writing IE2 (for twinning program)']
    existing_cols = [col for col in cols_to_drop if col in pivot_df.columns]
    if existing_cols:
        pivot_df.drop(existing_cols, axis=1, inplace=True)


    # Merge with the XepLoaiNH column
    df = pd.merge(pivot_df, raw_data[['MaSV', 'XepLoaiNH']], on='MaSV')
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    df.loc[df['XepLoaiNH'].isin(['Khá', 'Trung Bình Khá', 'Giỏi', 'Kém', 'Trung Bình', 'Yếu', 'Xuất sắc']), 'XepLoaiNH'] = df['XepLoaiNH'].map({'Khá': 'K', 'Trung Bình Khá': 'TK', 'Giỏi': 'G', 'Kém': 'Km', 'Trung Bình': 'TB', 'Yếu': 'Y', 'Xuất sắc': 'X'})
    df.drop(['MaSV', 'XepLoaiNH'], axis=1, inplace=True)
    df.replace('WH', np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    df

    # Streamlit app
    st.title('IT Student Scores')

    # Select course dropdown
    course = st.selectbox('Select a course:', df.columns)

    # Filter the data for the selected course
    course_data = df[course].dropna()

    # Calculate summary statistics for the course
    mean = course_data.mean()
    median = course_data.median()
    std_dev = course_data.std()

    # Show summary statistics
    st.write('Course:', course)
    st.write('Mean:', mean)
    st.write('Median:', median)
    st.write('Standard deviation:', std_dev)

    graph_type = st.selectbox('Select a graph type:', ['Histogram', 'Scatter plot', 'Box plot'])

    if graph_type == 'Histogram':
        # Create histogram using Plotly
        fig = px.histogram(course_data, nbins=20, range_x=[0, 100], labels={'value': 'Score'})
        fig.update_layout(title='Distribution of Scores for {}'.format(course))
        st.plotly_chart(fig)
    else:
        # Create box plot using Plotly
        fig = px.box(df, y=course, labels={'value': 'Score'})
        fig.update_layout(title='Box plot of {}'.format(course))
        st.plotly_chart(fig)
except:
    print("Not found csv!")