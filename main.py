import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from function import process_data,predict_late_student, predict_rank,predict_one_student
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO


df = pd.DataFrame()


def color_cell(val):
    if val == "not late":
        color = "green"
    elif val == "may late":
        color = "yellow"
    elif val == "late":
        color = "red"
    else:
        color = "black"
    return "color: %s" % color


def get_year(student_id):
    return int(student_id[6:8])


def generate_comment(median):
    if median < 30:
        comment = f"The median score for {course} is quite low at {median}. Students may need to work harder to improve their performance."
    elif median < 50:
        comment = f"The median score for {course} is below average at {median}. Students should work on improving their understanding of the material."
    elif median < 80:
        comment = f"The median score for {course} is solid at {median}. Students are making good progress but could still work on improving their skills."
    else:
        comment = f"The median score for {course} is outstanding at {median}. Students are doing an excellent job in this course."
    return comment

favicon = 'R.png'

st.set_page_config(
page_title='Student System',
page_icon=favicon,
layout='wide',
)
currentYear = datetime.now().year
im1 = Image.open("R.png")

# get the image from the URL


# create a three-column layout
col1, col2 = st.columns([1, 3])

# add a centered image to the first and third columns
with col1:
    st.image(im1, width=150)


# add a centered title to the second column
with col2:
    st.title("Student Performance Prediction System")


# Load the raw data
# uploaded_file = st.file_uploader("Choose a score file", type=["xlsx", "csv"])

# if uploaded_file is not None:
#     file_contents = uploaded_file.read()
#     file_ext = uploaded_file.name.split(".")[-1].lower()  # Get the file extension
    
#     if file_ext == "csv":
#         df = pd.read_csv(BytesIO(file_contents))
#     elif file_ext in ["xls", "xlsx"]:
#         df = pd.read_excel(BytesIO(file_contents))
#     else:
#         st.error("Invalid file format. Please upload a CSV or Excel file.")

# raw_data = df.copy()
raw_data = pd.read_csv("All_major.csv")
st.sidebar.title("Analysis Tool")

option = ["Dashboard", "Predict"]
# Add an expander to the sidebar
tabs = st.sidebar.selectbox("Select an option", option)


# draw histogram
# Streamlit app
if tabs == "Dashboard":
#     try:

        df = process_data(raw_data)
        unique_values_major = df["Major"].unique()
        unique_values_year = df["Year"].unique()
        all_values_year = np.concatenate([["All"],unique_values_year ])
        unique_values = df["MaSV_school"].unique()
        all_values = np.concatenate([["All"],unique_values ])
        major=st.selectbox("Select a major:", unique_values_major)
        school = st.selectbox("Select a school:", all_values)
        year = st.selectbox("Select a year:", all_values_year)
        
        
        if school == "All":
        # If so, display the entire DataFrame
          filtered_df = df.copy()
        else:
        # Otherwise, filter the DataFrame based on the selected value
          filtered_df = df[df["MaSV_school"] == school]
          filtered_df  = filtered_df.dropna(axis=1, how="all")
        
        # Select course dropdown
        df=filtered_df
        
        # Fix code to show the filtered data with df['Year']

        if year == "All":
            # If so, display the entire DataFrame
            filtered_df = df.copy()
        else:
            # Otherwise, filter the DataFrame based on the selected value
            filtered_df = df[df["Year"] == year]
            filtered_df = filtered_df.dropna(axis=1, how="all")
        df=filtered_df
        
        options = df.columns[:-2]
        course = st.selectbox("Select a course:", options)

        # Filter the data for the selected course
        course_data = df[course].dropna()
        
        # Calculate summary statistics for the course

        
        st.write(generate_comment(course_data.median()))
        # Show summary statistics
        
        st.write("Course:", course, " of ", school," student" )


        col1, col2,col3= st.columns(3)

        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=course_data, nbinsx=40, name="Histogram"
                )
            )
            fig.update_layout(
                title="Histogram of Scores for {}".format(course),
                xaxis_title="Score",
                yaxis_title="Count",
                height=400,
                width=400
            )
            st.plotly_chart(fig)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    y=course_data, name="Box plot"
                )
            )
            fig.update_layout(
                title="Box plot of Scores for {}".format(course),
                yaxis_title="Score",
                height=400,
                width=400
            )
            st.plotly_chart(fig)
        with col3:
            raw_data['MaSV_school'] = raw_data['MaSV'].str.slice(2, 4)
            if school == "All":
        # If so, display the entire DataFrame
                data = raw_data.copy()
            else:
        # Otherwise, filter the DataFrame based on the selected value
                data = raw_data[raw_data["MaSV_school"] == school]
            df1=data[['TenMH','NHHK','DiemHP']].copy()
            df1['DiemHP'] = pd.to_numeric(df1['DiemHP'], errors='coerce')
            df1['NHHK'] = df1['NHHK'].apply(lambda x: str(x)[:4] + ' S ' + str(x)[4:])
            selected_TenMH = " " + course
            filtered_df1 = df1[df1['TenMH'] == selected_TenMH]
            mean_DiemHP = filtered_df1.groupby('NHHK')['DiemHP'].mean().round(1).reset_index(name='Mean')
            # Create Plotly line graph
            fig = px.line(mean_DiemHP, x='NHHK', y='Mean', title=f"Mean DiemHP for{selected_TenMH} thought period")
            fig.update_layout(
              height=400,
              width=400)           
            st.plotly_chart(fig)


#     except:
#         st.write("Add CSV to analysis")


# predict student

elif tabs == "Predict":
    try:
        raw_data = pd.read_csv("dataScore.csv")
        predict = predict_late_student(raw_data)
        rank = predict_rank(raw_data)

        predict = pd.merge(predict, rank, on="MaSV")
        rank_mapping = {
            "Khá": "Good",
            "Trung Bình Khá": "Average good",
            "Giỏi": "Very good",
            "Kém": "Very weak",
            "Trung Bình": "Ordinary",
            "Yếu": "Weak",
            "Xuất Sắc": "Excellent",
        }
        predict["Pred Rank"].replace(rank_mapping, inplace=True)

        # Filter students who have a Result value of "late"
        df_late = predict

        MaSV = st.text_input("Enter Student ID:")
        if MaSV:
            df_filtered = predict[predict["MaSV"] == MaSV]
            styled_table = (
                df_filtered[["MaSV", "GPA", "Mean_Cre", "Pred Rank", "Result", "Period"]]
                .style.applymap(color_cell)
                .format({"GPA": "{:.2f}", "Mean_Cre": "{:.1f}", "Period": "{:.1f}"})
            )
            
            with st.container():
                st.write(styled_table)
                predict_one_student(raw_data,MaSV)
        else:
            df_late = predict
            # df_late = predict[(predict['Pred Rank'] == 'Yếu') | (predict['Pred Rank'] == 'Kém')]
            df_late["Year"] = 2000 + df_late["MaSV"].apply(get_year)
            df_late = df_late[
                (df_late["Year"] != currentYear - 1) & (df_late["Year"] != currentYear - 2)
            ]
            year = st.selectbox("Select Year", options=df_late["Year"].unique())
            df_filtered = df_late[df_late["Year"] == year]
            styled_table = (
                df_filtered[["MaSV", "GPA", "Mean_Cre", "Pred Rank", "Result", "Period"]]
                .style.applymap(color_cell)
                .format({"GPA": "{:.2f}", "Mean_Cre": "{:.2f}", "Period": "{:.2f}"})
            )
            csv = df_filtered.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="Preidct data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            fig1 = px.pie(
                df_filtered,
                names="Pred Rank",
                title="Pred Rank",
                color_discrete_sequence=px.colors.sequential.Mint,
                height=400,
                width=400,
            )
            fig2 = px.pie(
                df_filtered,
                names="Result",
                title="Result",
                color_discrete_sequence=px.colors.sequential.Peach,
                height=400,
                width=400,
            )
            fig1.update_layout(
                title={
                    "text": "Pred Rank",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )
            fig2.update_layout(
                title={
                    "text": "Result",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )
            st.dataframe(styled_table)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig1)
            with col2:
                st.plotly_chart(fig2)

            

        # display the grid of pie charts using Streamlit

    except:
        st.write('Add CSV to analysis')
