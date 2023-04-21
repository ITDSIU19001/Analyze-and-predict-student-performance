import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st



def process_data(raw_data):
    # Pivot the DataFrame
    pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())

    # Drop unnecessary columns
    cols_to_drop = []
    with open('cols_to_drop.txt', 'r') as f:
      for line in f:
        cols_to_drop.append(str(line.strip()))
    existing_cols = [col for col in cols_to_drop if col in pivot_df.columns]
    if existing_cols:
        pivot_df.drop(existing_cols, axis=1, inplace=True)

    # Merge with the XepLoaiNH column
    df = pd.merge(pivot_df, raw_data[['MaSV', 'XepLoaiNH']], on='MaSV')
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    dfid=df['MaSV']
    df.drop(['MaSV', 'XepLoaiNH'], axis=1, inplace=True)
    df.replace('WH', np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    df = pd.merge(dfid,df,left_index=True, right_index=True)
    df['MaSV_school'] = df['MaSV'].str.slice(2, 4)
    df=df.drop(columns='MaSV')
    
    return df
    
def process_data_per(raw_data):
    # Pivot the DataFrame
    pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())

    # Drop unnecessary columns
    cols_to_drop = []
    with open('cols_to_drop.txt', 'r') as f:
      for line in f:
        cols_to_drop.append(str(line.strip()))
    existing_cols = [col for col in cols_to_drop if col in pivot_df.columns]
    if existing_cols:
        pivot_df.drop(existing_cols, axis=1, inplace=True)
    pivot_df.replace('WH', np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)
    # Merge with the XepLoaiNH column
    df = pd.merge(pivot_df, raw_data[['MaSV', 'XepLoaiNH']], on='MaSV')
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    df.drop(['XepLoaiNH'], axis=1, inplace=True)
    
    return df


def process_student_data(raw_data):
    dtk = raw_data[["MaSV", "DTBTKH4"]].copy()
    dtk.drop_duplicates(subset="MaSV", keep="last", inplace=True)

    count_duplicates = raw_data.groupby(["MaSV", "TenMH"]).size().reset_index(name="Times")

    courses_list = []
    with open('courses_list.txt', 'r') as f:
      for line in f:
        courses_list.append(str(line.strip()))

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
    courses_list=[' Intensive English 0- Twinning Program',
  ' Intensive English 01- Twinning Program',
  ' Intensive English 02- Twinning Program',
  ' Intensive English 03- Twinning Program',
  ' Intensive English 1- Twinning Program',
  ' Intensive English 2- Twinning Program',
  ' Intensive English 3- Twinning Program',
  ' Listening & Speaking IE1',
  ' Listening & Speaking IE2',
  ' Listening & Speaking IE2 (for twinning program)',
  ' Physical Training 1',
  ' Physical Training 2',
  ' Reading & Writing IE1',
  ' Reading & Writing IE2',
  ' Reading & Writing IE2 (for twinning program)']
    df1=raw_data[['MaSV','TenMH','NHHK']]
    filtered_df = df1[df1['TenMH'].isin(courses_list)]
    nhhk_counts = filtered_df.groupby('MaSV')['NHHK'].nunique().reset_index(name='EPeriod')
    df = pd.merge(df, nhhk_counts, on='MaSV')
    return df

def predict_late_student(test_df):
    # Load the pre-trained model
    with open('modelLate.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('modelPeriod.pkl', 'rb') as file:
        model1 = pickle.load(file)
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

    prediction = model1.predict(test_dfed)

    # Add a new column to the student data indicating if the student is late
    test_dfed['Period'] = prediction

    # Add the student ID column back to the beginning of the DataFrame
    test_dfed.insert(0, 'MaSV', std_id)
    
    for index, row in test_dfed.iterrows():
      if row['Period'] == 8 and row['Result'] == 'late':
        test_dfed.loc[index, 'Period'] = row['Period'] / 2
        test_dfed.loc[index, 'Result'] = 'may late'
      elif row['Period'] == 9 and row['Result'] == 'late':
        test_dfed.loc[index, 'Period'] = row['Period'] / 2
        test_dfed.loc[index, 'Result'] = 'may late'
      else:
        test_dfed.loc[index, 'Period'] = row['Period'] / 2

    return test_dfed

def predict_rank(raw_data):
    # Pivot the DataFrame
    raw_data = raw_data[raw_data["MaSV"].str.startswith("IT")]
    pivot_df = pd.pivot_table(
        raw_data, values="DiemHP", index="MaSV", columns="TenMH", aggfunc="first"
    )
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())

    # Drop unnecessary columns
    cols_to_drop = []
    with open('cols_to_drop.txt', 'r') as f:
      for line in f:
        cols_to_drop.append(str(line.strip()))
    existing_cols = [col for col in cols_to_drop if col in pivot_df.columns]
    if existing_cols:
        pivot_df.drop(existing_cols, axis=1, inplace=True)

    pivot_df.replace("WH", np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

    # Merge with the XepLoaiNH column
    df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
    df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
    col=df.drop(['MaSV', 'DTBTK'], axis=1)
    
    columns_data = []
    columns_to_fill = []
    with open('column_all.txt', 'r') as f:
      for line in f:
        columns_to_fill.append(str(line.strip()))
        columns_data.append(str(line.strip()))
        
    merge=[]
    with open('colum_merge.txt', 'r') as f:
      for line in f:
        merge.append(str(line.strip()))
    
    dup=pd.DataFrame(columns=columns_data)
    df= pd.merge(dup, df, on=merge, how='outer')
    for col in df.columns:
        if col in columns_to_fill:
            df[col].fillna(value=df["DTBTK"], inplace=True)
    std_id = df['MaSV'].copy()
    df=df.drop(['MaSV', 'DTBTK'], axis=1)


    with open('modelRank2.pkl', 'rb') as file:
      model = pickle.load(file)
    prediction = model.predict(df)
    df['Pred Rank'] = prediction
    df.insert(0, 'MaSV', std_id)
    df=df[['MaSV','Pred Rank']]
    return df


def process_data_per1(raw_data, student_id):
    # Subset the DataFrame to relevant columns and rows
      student = process_data_per(raw_data)
      filtered_df = student[student["MaSV"] == student_id]
      if len(filtered_df) > 0:
        selected_row = filtered_df.iloc[0, 1:].dropna()
        colname = filtered_df.dropna().columns.tolist()
        values = selected_row.values.tolist()

      # create a line chart using plotly
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=values, nbinsx=20, name=student_id))

        # set the chart title and axis labels
        fig1.update_layout(
            title="Histogram for student {}".format(student_id),
            xaxis_title="Value",
            yaxis_title="Frequency",
            width=500
        )

        # create a bar chart using plotly express
        data = raw_data[['MaSV', 'NHHK', 'TenMH', 'DiemHP']]
        data['TenMH'] = data['TenMH'].str.lstrip()
        data['NHHK'] = data['NHHK'].apply(lambda x: str(x)[:4] + ' S ' + str(x)[4:])
        rows_to_drop = []
        with open('rows_to_drop.txt', 'r') as f:
            for line in f:
                rows_to_drop.append(str(line.strip()))
        data = data[~data['TenMH'].isin(rows_to_drop)]
        student_data = data[data['MaSV'] == student_id][['NHHK', 'TenMH', 'DiemHP']]
        student_data['DiemHP'] = pd.to_numeric(student_data['DiemHP'], errors='coerce')

        fig2 = px.bar(student_data, x='TenMH', y='DiemHP', color='NHHK', title='Student Score vs. Course')
        fig2.update_layout(
            title="Student Score vs. Course",
            xaxis_title=None,
            yaxis_title="Score",
        )

        # display the charts using st.column
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)

        with col2:
            st.plotly_chart(fig2)
      else:
        st.write("No data found for student {}".format(student_id))