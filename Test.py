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

raw_data = df.copy()

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
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    df.loc[df['XepLoaiNH'].isin(['Khá', 'Trung Bình Khá', 'Giỏi', 'Kém', 'Trung Bình', 'Yếu', 'Xuất sắc']), 'XepLoaiNH'] = df['XepLoaiNH'].map({'Khá': 'K', 'Trung Bình Khá': 'TK', 'Giỏi': 'G', 'Kém': 'Km', 'Trung Bình': 'TB', 'Yếu': 'Y', 'Xuất sắc': 'X'})
    df.drop(['MaSV', 'XepLoaiNH'], axis=1, inplace=True)
    df.replace('WH', np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)

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

except:
    st.title('Add CSV to analysis')

yeu_kem_students = raw_data.loc[raw_data['XepLoaiNH'].isin(['Yếu', 'Kém'])]

# Split students by year
def get_year(ma_sv):
    year = int(ma_sv[6:8]) + 2000
    return year

yeu_kem_students['year'] = yeu_kem_students['MaSV'].apply(get_year)
years = yeu_kem_students['year'].unique()

year_options = yeu_kem_students['year'].unique()
selected_year = st.selectbox('Select year', year_options)

# Filter dataframe based on selected year
year_students = yeu_kem_students.loc[yeu_kem_students['year'] == selected_year]
year_students.drop_duplicates(subset='MaSV', keep='last', inplace=True)
# Display dataframe
st.write(year_students[['MaSV', 'XepLoaiNH']])

def get_scores(ma_sv):
    scores = raw_data.loc[raw_data['MaSV'] == ma_sv, ['TenMH', 'DiemHP']]
    return scores

# Define a function to format the scores as a table
def format_scores(scores):
    scores_table = scores.set_index('TenMH')
    scores_table.index.name = None
    scores_table.columns = ['DiemHP']
    return scores_table

# Create a dataframe widget for the students
students_df = yeu_kem_students[['MaSV', 'XepLoaiNH']]
student_selected = st.empty()
scores_table = st.empty()

# Add an on_click event to the students dataframe
def on_student_click(row):
    student_ma_sv = row['MaSV']
    student_selected.markdown(f"**Selected student:** {student_ma_sv}")
    student_scores = get_scores(student_ma_sv)
    scores_table.dataframe(format_scores(student_scores))

students_df = students_df.assign(hack='').set_index('hack')
students_df = st.dataframe(students_df, height=500, width=800)
students_df.selectbox('Select a student', options=students_df.index, on_change=on_student_click, key='selectbox')