import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go
import pickle
# from preprocess import predict_late_student

df = pd.DataFrame()



def process_student_data(raw_data):
  dtk = raw_data[["MaSV", "DTBTKH4"]].copy()
  dtk.drop_duplicates(subset="MaSV", keep="last", inplace=True)

  count_duplicates = raw_data.groupby(["MaSV", "TenMH"]).size().reset_index(name="Times")

  courses_list = [
      " Calculus 1",
      " Calculus 2",
      " Calculus 3",
      " Chemistry Laboratory",
      " Chemistry for Engineers",
      " Critical Thinking",
      " History of Vietnamese Communist Party",
      " Internship",
      " Philosophy of Marxism and Leninism",
      " Physics 1",
      " Physics 2",
      " Physics 3",
      " Physics 3 Laboratory",
      " Physics 4",
      " Political economics of Marxism and Leninism",
      " Principles of Database Management",
      " Principles of Marxism",
      " Principles of Programming Languages",
      " Probability, Statistic & Random Process",
      " Regression Analysis",
      " Revolutionary Lines of Vietnamese Communist Party",
      " Scientific socialism",
      " Speaking AE2",
      " Special Study of the Field",
      " Thesis",
      " Writing AE1",
      " Writing AE2",
      " Intensive English 0- Twinning Program",
      " Intensive English 01- Twinning Program",
      " Intensive English 02- Twinning Program",
      " Intensive English 03- Twinning Program",
      " Intensive English 1- Twinning Program",
      " Intensive English 2- Twinning Program",
      " Intensive English 3- Twinning Program",
      " Listening & Speaking IE1",
      " Listening & Speaking IE2",
      " Listening & Speaking IE2 (for twinning program)",
      " Physical Training 1",
      " Physical Training 2",
      " Reading & Writing IE1",
      " Reading & Writing IE2",
      " Reading & Writing IE2 (for twinning program)",
  ]

  # Create two new columns for counting courses that are in the courses_list or not
  count_duplicates["fail_courses_list"] = (
      (count_duplicates["TenMH"].isin(courses_list)) & (count_duplicates["Times"] >= 2)
  ).astype(int)

  count_duplicates["fail_not_courses_list"] = (
      (~count_duplicates["TenMH"].isin(courses_list)) & (count_duplicates["Times"] >= 2)
  ).astype(int)

  count_duplicates["pass_courses"] = (
      (~count_duplicates["TenMH"].isin(courses_list)) & (count_duplicates["Times"] == 1)
  ).astype(int)

  # Group the data by "MaSV" and sum the counts for the two new columns
  fail = (
      count_duplicates.groupby("MaSV")[["fail_courses_list", "fail_not_courses_list"]]
      .sum()
      .reset_index()
  )

  # Rename the columns to reflect the split of courses_list and not courses_list
  fail.columns = ["MaSV", "fail_courses_list_count", "fail_not_courses_list_count"]

  df = pd.merge(dtk, fail, on="MaSV")
  df = df.rename(columns={"DTBTKH4": "GPA"})

  data = raw_data[['MaSV','NHHK','SoTCDat']]
  data = data.drop_duplicates()
  data = data.groupby(['MaSV'])['SoTCDat'].median().reset_index(name='Median_Cre').round(1)

  df = pd.merge(df, data, on='MaSV')

  return df

def predict_late_student(test_df):
    # Load the pre-trained model
    with open('modelTimePredict.pkl', 'rb') as file:
        model = pickle.load(file)
    # Process the student data
    test_dfed = process_student_data(test_df)

    # Save the student ID column
    std_id = test_dfed.iloc[:, 0]

    # Drop the student ID column
    test_dfed = test_dfed.drop(test_dfed.columns[0], axis=1)

    # Make predictions using the pre-trained model
    prediction = model.predict(test_dfed)

    # Add a new column to the student data indicating if the student is late
    test_dfed['Result'] = ['late' if p == 1 else 'not late' for p in prediction]

    # Add the student ID column back to the beginning of the DataFrame
    test_dfed.insert(0, 'MaSV', std_id)

    return test_dfed

# Load the raw data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

raw_data = df.copy()

try:
    # # Clean and transform the data
    # pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    # pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    # pivot_df.columns.name = None
    # pivot_df = pivot_df.dropna(thresh=50, axis=1)
    # pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    # cols_to_drop = ['Intensive English 0- Twinning Program',
    #     'Intensive English 01- Twinning Program',
    #     'Intensive English 02- Twinning Program',
    #     'Intensive English 03- Twinning Program',
    #     'Intensive English 1- Twinning Program',
    #     'Intensive English 2- Twinning Program',
    #     'Intensive English 3- Twinning Program', 'Listening & Speaking IE1',
    #     'Listening & Speaking IE2',
    #     'Listening & Speaking IE2 (for twinning program)','Physical Training 1', 'Physical Training 2', 'Reading & Writing IE1',
    #     'Reading & Writing IE2', 'Reading & Writing IE2 (for twinning program)']
    # existing_cols = [col for col in cols_to_drop if col in pivot_df.columns]
    # if existing_cols:
    #     pivot_df.drop(existing_cols, axis=1, inplace=True)

    # # Merge with the XepLoaiNH column
    # df = pd.merge(pivot_df, raw_data[['MaSV', 'XepLoaiNH']], on='MaSV')
    # df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    # df.loc[df['XepLoaiNH'].isin(['Khá', 'Trung Bình Khá', 'Giỏi', 'Kém', 'Trung Bình', 'Yếu', 'Xuất sắc']), 'XepLoaiNH'] = df['XepLoaiNH'].map({'Khá': 'K', 'Trung Bình Khá': 'TK', 'Giỏi': 'G', 'Kém': 'Km', 'Trung Bình': 'TB', 'Yếu': 'Y', 'Xuất sắc': 'X'})
    # df.drop(['MaSV', 'XepLoaiNH'], axis=1, inplace=True)
    # df.replace('WH', np.nan, inplace=True)
    # df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)

    # # Streamlit app
    # st.title('IT Student Scores')

    # # Select course dropdown
    # course = st.selectbox('Select a course:', df.columns)

    # # Filter the data for the selected course
    # course_data = df[course].dropna()

    # # Calculate summary statistics for the course
    # mean = course_data.mean()
    # median = course_data.median()
    # std_dev = course_data.std()

    # # Show summary statistics
    # st.write('Course:', course)
    # st.write('Mean:', mean)
    # st.write('Median:', median)
    # st.write('Standard deviation:', std_dev)

    # graph_type = st.selectbox('Select a graph type:', ['Histogram', 'Z plot', 'Box plot'])

    # if graph_type == 'Histogram':
    #     # Create histogram using Plotly
    #     fig = px.histogram(course_data, nbins=40, range_x=[0, 100], labels={'value': 'Score'})
    #     fig.update_layout(title='Distribution of Scores for {}'.format(course))
    #     st.plotly_chart(fig)
    # elif graph_type == 'Z plot':
    #     z_scores = stats.zscore(course_data)
    #     # Create histogram of z-scores
    #     fig = go.Figure(data=[go.Histogram(x=z_scores)])
    #     fig.update_layout(title='Z-Score Distribution for {}'.format(course))
    #     st.plotly_chart(fig)
    # else:
    #     # Create box plot using Plotly
    #     fig = px.box(df, y=course, labels={'value': 'Score'})
    #     fig.update_layout(title='Box plot of {}'.format(course))
    #     st.plotly_chart(fig)

    predict=predict_late_student(raw_data)
    st.dataframe(predict)
    

except:
    st.title('Add CSV to analysis')