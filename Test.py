import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go


df = pd.DataFrame()
# Load the raw data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

raw_data = df
try:
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
    df2=df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    df.loc[df['XepLoaiNH'].isin(['Khá', 'Trung Bình Khá', 'Giỏi', 'Kém', 'Trung Bình', 'Yếu', 'Xuất sắc']), 'XepLoaiNH'] = df['XepLoaiNH'].map({'Khá': 'K', 'Trung Bình Khá': 'TK', 'Giỏi': 'G', 'Kém': 'Km', 'Trung Bình': 'TB', 'Yếu': 'Y', 'Xuất sắc': 'X'})
    df=df.drop(['MaSV', 'XepLoaiNH'], axis=1, inplace=True)
    df.replace('WH', np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    df2

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

    graph_type = st.selectbox('Select a graph type:', ['Histogram', 'Z plot', 'Box plot'])

    if graph_type == 'Histogram':
        # Create histogram using Plotly
        fig = px.histogram(course_data, nbins=40, range_x=[0, 100], labels={'value': 'Score'})
        fig.update_layout(title='Distribution of Scores for {}'.format(course))
        st.plotly_chart(fig)
    elif graph_type == 'Z plot':
        z_scores = stats.zscore(course_data)
        # Create histogram of z-scores
        fig = go.Figure(data=[go.Histogram(x=z_scores)])
        fig.update_layout(title='Z-Score Distribution for {}'.format(course))
        st.plotly_chart(fig)
    else:
        # Create box plot using Plotly
        fig = px.box(df, y=course, labels={'value': 'Score'})
        fig.update_layout(title='Box plot of {}'.format(course))
        st.plotly_chart(fig)

    weak_students = df2[df2['XepLoaiNH'].isin(['Yếu', 'Kém'])]

    # Create a dictionary to store the tables for each year
    year_tables = {}

    # Loop through the rows of the weak_students DataFrame
    for _, row in weak_students.iterrows():
    # Extract the year from the MaSV column
        year = row['MaSV'][6:8]

    # If the year table doesn't exist in the dictionary, create a new one
        if year not in year_tables:
            year_tables[year] = pd.DataFrame(columns=weak_students.columns)

    # Append the row to the year table
        year_tables[year] = pd.concat([year_tables[year], row.to_frame().transpose()], ignore_index=True)

    # Display the tables for each year in Streamlit
    for year, year_table in year_tables.items():
        st.write(f"Year {20}{year}")
        st.write(year_table["MaSV","XepLoaiMH"])
        st.write('---')
except:
    st.title('Add CSV')
