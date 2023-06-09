import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from function import (
    process_data,
    predict_late_student,
    predict_rank,
    predict_one_student,
)
from datetime import datetime
from PIL import Image
import base64
import re
import sqlite3

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
    year_str = ""
    for char in student_id:
        if char.isdigit():
            year_str += char
            if len(year_str) == 2:  # Stop when we have extracted two numbers
                break
    return int(year_str)


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


favicon = "R.png"
hcm = "HCM.png"
st.set_page_config(
    page_title="Student System",
    page_icon=favicon,
    layout="wide",
)
currentYear = datetime.now().year
im1 = Image.open("R.png")

# get the image from the URL


# create a three-column layout
col1, col2, col3 = st.columns([1, 3, 1])

# add a centered image to the first and third columns
with col1:
    st.image(im1, width=150)


# add a centered title to the second column
with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Student Performance Prediction System</h1>",
        unsafe_allow_html=True,
    )
#     st.header("Student Performance Prediction System")

with col3:
    st.image(hcm, width=250)


@st.cache_data()
def read_sql_query():
    """Reads the SQL query from the database and returns a DataFrame."""
    conn = sqlite3.connect("database.db")
    query = """SELECT MaSV, TenMH, DiemHP, NHHK, DTBTKH4, MaMH, SoTCDat, DTBTK
    FROM scoreTable;
    """
    df = pd.read_sql_query(query, conn)
    return df


raw_data = read_sql_query()

# raw_data = pd.read_sql_query(query, conn)
df = process_data(raw_data)

# raw_data = pd.read_csv("All_major.csv")
st.sidebar.title("Analysis Tool")

option = ["Dashboard", "Prediction Performance"]
# Add an expander to the sidebar
tabs = st.sidebar.selectbox("Select an option", option)


def filter_dataframe(df, column, value):
    if value == "All":
        return df
    else:
        return df[df[column] == value]


# draw histogram
# Streamlit app
if tabs == "Dashboard":
    #     try:
    additional_selection = " "
    # Filter by Major
    unique_values_major = df["Major"].unique()
    unique_values_major = [
    "BA",
    "BE",
    "BT",
    "CE",
    "EE",
    "EN",
    "EV",
    "IE",
    "MA",
    "SE",
    "IT"]
    unique_values_major = sorted(unique_values_major, key=lambda s: s)
    major = st.selectbox("Select a school:", unique_values_major)
    df = filter_dataframe(df, "Major", major)
    dfa= filter_dataframe(df, "Major", major)

    # Filter by MaSV_school
    unique_values_school = df["MaSV_school"].unique()
    all_values_school = np.concatenate([["All"], unique_values_school])
    no_numbers = [x for x in all_values_school if not re.search(r"\d", str(x))]

    if len(no_numbers) == 2:
        school = no_numbers[1]
    else:
        col1, col2 = st.columns(2)  # Split the screen into two columns

        with col1:
            school = st.selectbox("Select a major:", no_numbers)

        if school != "All":
            values = [x for x in no_numbers if x != "All" and x != school]
            values = np.concatenate([[" "], values])

            with col2:
                additional_selection = st.selectbox("Select another major for comparisons:", values)
                if additional_selection != " ":
                    dfa = filter_dataframe(dfa, "MaSV_school", additional_selection)
        
    df = filter_dataframe(df, "MaSV_school", school)
    # Filter by Year
    unique_values_year = df["Year"].unique()
    all_values_year = np.concatenate([["All"], unique_values_year])

    # Split the selectboxes into two columns
    col1, col2 = st.columns(2)

    # First selectbox in the first column
    with col1:
        year = st.selectbox("Select a year:", all_values_year)

    # Second selectbox in the second column
    with col2:
        year_a = " "
        if year != "All" and additional_selection == " ":
            year_list = [x for x in all_values_year if x != "All" and x != year]
            year_list = np.concatenate([[" "], year_list])
            year_a = st.selectbox("Select another year for comparisons:", year_list)

            if year_a != " ":
                dfa = filter_dataframe(dfa, "Year", year_a)
                dfa.dropna(axis=1, thresh=1, inplace=True)
    
    df = filter_dataframe(df, "Year", year)

    # Drop NaN columns
    df.dropna(axis=1, thresh=1, inplace=True)
    
    new_df = df.iloc[:, :-3].dropna(axis=1, thresh=10).apply(pd.to_numeric)
    new_dfa = dfa.iloc[:, :-3].dropna(axis=1, thresh=10).apply(pd.to_numeric)
    list1=new_df.columns.tolist()
    list2=new_dfa.columns.tolist()
    if (year != "All" and year_a != " ") or (school != "All" and additional_selection != " "):
        dfac=new_dfa.columns[:-3].tolist()
        common_elements = np.intersect1d(list1, list2)
        # Merge arrays with common elements
        merged_array = np.concatenate((list1, list2), axis=None)

        # Filter merged array to retain only common elements
        list3 = np.intersect1d(merged_array, common_elements)
        new_df = new_df[list3]
        new_dfa = new_dfa[list3]
    if additional_selection != " ":
        show_boxplot = st.checkbox("Show Boxplot for All Course")

        if show_boxplot:
            fig = px.box(new_df)
            fig1 = px.box(new_dfa)
            fig.update_layout(title="Boxplot of " + major + school + " student at " + year )
            fig1.update_layout(title="Boxplot of " + major + additional_selection + " student at " + year)
            # fig.update_layout(width=1250)
            # fig1.update_layout(width=1250)
            st.plotly_chart(fig,use_container_width=True)
            st.plotly_chart(fig1,use_container_width=True)

    elif additional_selection == " " and year_a != "All":
        show_boxplot = st.checkbox("Show Boxplot for All Course")

        if show_boxplot:
            fig = px.box(new_df)
            fig1 = px.box(new_dfa)
            fig.update_layout(title="Boxplot of " + major + school + " student at " + year)
            fig1.update_layout(title="Boxplot of " + major + school + " student at " + year_a)
            # fig.update_layout(width=1250)
            # fig1.update_layout(width=1250)
            st.plotly_chart(fig,use_container_width=True)
            st.plotly_chart(fig1,use_container_width=True)

    elif additional_selection == " ":
        show_boxplot = st.checkbox("Show Boxplot for All Course")

        if show_boxplot:
            fig = px.box(new_df)
            fig.update_layout(title="Boxplot of " + major + school + " student at " + year)
            # fig.update_layout(width=1250)
            st.plotly_chart(fig,use_container_width=True)


    # Select course dropdown
    options = df.columns[:-3]


    course_data_dict = {course: df[course].dropna() for course in options}
    valid_courses = [
        course for course, data in course_data_dict.items() if len(data) > 1
    ]
    if (year != "All" and year_a != " ") or (school != "All" and additional_selection != " "):
        dfac=new_dfa.columns[:-3].tolist()
        common_elements = np.intersect1d(valid_courses, dfac)
        # Merge arrays with common elements
        merged_array = np.concatenate((valid_courses, dfac), axis=None)

        # Filter merged array to retain only common elements
        valid_courses = np.intersect1d(merged_array, common_elements)


    if len(valid_courses) > 5:
        course = st.selectbox("Select a course:", valid_courses)
    elif len(valid_courses) == 1:
        course = valid_courses[0]
    else:
        st.write("No valid course data found!")
        st.stop()

    # Filter the data for the selected course
    course_data = course_data_dict[course]

    # Generate comment and summary statistics
    if len(course_data) > 1:
        if school == "All":
            st.write("Course:", course, " of ", major, " student")
        else:
            st.write("Course:", course, " of ", major + school, " student")
        st.write(generate_comment(course_data.median()))
    else:
        st.write("No data available for the selected course.")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=course_data, nbinsx=40, name="Histogram"))
        fig.update_layout(
            title="Histogram of Scores for {}".format(course),
            xaxis_title="Score",
            yaxis_title="Count",
            height=400,
            width=400,
        )
        st.plotly_chart(fig)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=course_data, name="Box plot"))
        fig.update_layout(
            title="Box plot of Scores for {}".format(course),
            yaxis_title="Score",
            height=400,
            width=400,
        )
        st.plotly_chart(fig,use_container_width=True)
        
    with col3:
        raw_data1=raw_data.copy()
        raw_data1["major"] = raw_data1["MaSV"].str.slice(0, 2)
        raw_data1.replace(["WH", "VT", "I"], np.nan, inplace=True)
        raw_data1 = raw_data1[~raw_data1["DiemHP"].isin(["P", "F", "PC"])]
        if major != "All":
            raw_data1 = raw_data1[raw_data1["major"] == major]

        # Filter by MaSV_school
        raw_data1["MaSV_school"] = raw_data1["MaSV"].str.slice(2, 4)
        if school != "All":
            raw_data1 = raw_data1[raw_data1["MaSV_school"] == school]

        # Prepare DataFrame for visualization
        df1 = raw_data1[["TenMH", "NHHK", "DiemHP"]].copy()
        df1["DiemHP"] = df1["DiemHP"].astype(float)
        df1["NHHK"] = df1["NHHK"].apply(lambda x: str(x)[:4] + " S " + str(x)[4:])

        # Filter by selected_TenMH
        selected_TenMH = " " + course
        filtered_df1 = df1[df1["TenMH"] == selected_TenMH]

        # Calculate mean DiemHP
        mean_DiemHP = (
            filtered_df1.groupby("NHHK")["DiemHP"]
            .mean()
            .round(1)
            .reset_index(name="Mean")
        )

        # Create Plotly line graph
        if year != "All":
            st.write("")
        else:
            fig = px.line(
                mean_DiemHP,
                x="NHHK",
                y="Mean",
                title=f"Mean DiemHP for{selected_TenMH} through Semeters",
            )
            fig.update_layout(height=400, width=400)
            st.plotly_chart(fig,use_container_width=True)




    course_data_dict = {course: dfa[course]}
    course_data = course_data_dict[course]
    
    if (year != "All" and year_a != " ") or (school != "All" and additional_selection != " "):
        
        st.write("Course:", course, " of ", major + additional_selection, " student in ", year_a)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=course_data , nbinsx=40, name="Histogram"))
            fig.update_layout(
                title="Histogram of Scores for {}".format(course),
                xaxis_title="Score",
                yaxis_title="Count",
                height=400,
                width=400,
            )
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(y=course_data , name="Box plot"))
            fig.update_layout(
                title="Box plot of Scores for {}".format(course),
                yaxis_title="Score",
                height=400,
                width=400,
            )
            st.plotly_chart(fig,use_container_width=True)
            
        with col3:
            raw_data["major"] = raw_data["MaSV"].str.slice(0, 2)
            raw_data.replace(["WH", "VT", "I"], np.nan, inplace=True)
            raw_data = raw_data[~raw_data["DiemHP"].isin(["P", "F", "PC"])]
            if major != "All":
                raw_data = raw_data[raw_data["major"] == major]

            # Filter by MaSV_school
            raw_data["MaSV_school"] = raw_data["MaSV"].str.slice(2, 4)
            raw_data = raw_data[raw_data["MaSV_school"] == additional_selection]

            # Prepare DataFrame for visualization
            df1 = raw_data[["TenMH", "NHHK", "DiemHP"]].copy()
            df1["DiemHP"] = df1["DiemHP"].astype(float)
            df1["NHHK"] = df1["NHHK"].apply(lambda x: str(x)[:4] + " S " + str(x)[4:])

            # Filter by selected_TenMH
            selected_TenMH = " " + course
            filtered_df1 = df1[df1["TenMH"] == selected_TenMH]

            # Calculate mean DiemHP
            mean_DiemHP = (
                filtered_df1.groupby("NHHK")["DiemHP"]
                .mean()
                .round(1)
                .reset_index(name="Mean")
            )

            # Create Plotly line graph
            if year != "All":
                st.write("")
            else:
                fig = px.line(
                    mean_DiemHP,
                    x="NHHK",
                    y="Mean",
                    title=f"Mean DiemHP for{selected_TenMH} through Semeters",
                )
                fig.update_layout(height=400, width=400)
                st.plotly_chart(fig,use_container_width=True)



#     except:
#         st.write("Add CSV to analysis")


# predict student

elif tabs == "Prediction Performance":
    # try:

    df = read_sql_query()
    df["Major"] = df["MaSV"].str.slice(0, 2)
    unique_values_major = [
        "BA",
        "BE",
        "BT",
        "CE",
        "EE",
        "EN",
        "EV",
        "IE",
        "MA",
        "SE",
        "IT"
    ]
    unique_values_major = sorted(unique_values_major, key=lambda s: s)
    major = st.selectbox("Select a school:", unique_values_major)
    df = filter_dataframe(df, "Major", major)
    predict = predict_late_student(df)
    rank = predict_rank(df)
    predict = pd.merge(predict, rank, on="MaSV")
    predict.rename(columns={'Mean_Cre': 'Mean Credit'}, inplace=True)
    
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

    # Filter students who have a Progress value of "late"
    df_late = predict
    
   
    MaSV = st.text_input("Enter Student ID:", key="MaSV")

    # Add a clear button


    def clear_form():
        st.session_state["MaSV"] = ""

    if st.button("Clear", on_click=clear_form):
        MaSV = ""  # Clear the input value

    if MaSV:
        df_filtered = predict[predict["MaSV"] == MaSV]
        styled_table = (
            df_filtered[["MaSV", "GPA", "Mean Credit", "Pred Rank", "Progress", "Semeters"]]
            .style.applymap(color_cell)
            .format({"GPA": "{:.2f}", "Mean Credit": "{:.1f}", "Semeters": "{:.1f}"})
        )

        with st.container():
            st.write(styled_table)
            predict_one_student(df, MaSV)
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
            df_filtered[["MaSV", "GPA", "Mean Credit", "Pred Rank", "Progress", "Semeters"]]
            .style.applymap(color_cell)
            .format({"GPA": "{:.2f}", "Mean Credit": "{:.2f}", "Semeters": "{:.2f}"})
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
            names="Progress",
            title="Progress",
            color_discrete_sequence=px.colors.sequential.Peach,
            height=400,
            width=400,
        )
        fig1.update_layout(
            title={
                "text": "Pred Rank",
                "y": 0.95,
                "x": 0.35,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        fig2.update_layout(
            title={
                "text": "Progress",
                "y": 0.95,
                "x": 0.35,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        
        col3,col1, col2= st.columns([2, 1,1])
        with col3:
            st.dataframe(styled_table)
        with col1:
            st.plotly_chart(fig1,use_container_width=True)
        with col2:
            st.plotly_chart(fig2,use_container_width=True)

    # display the grid of pie charts using Streamlit

# except:
#     st.write('Add CSV to analysis')
