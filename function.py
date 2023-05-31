import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import joblib
import re

def get_year(student_id):
    year_str = ""
    for char in student_id:
        if char.isdigit():
            year_str += char
            if len(year_str) == 2:  # Stop when we have extracted two numbers
                break
    return int(year_str)

@st.cache_data()
def process_data(raw_data):
    # Pivot the DataFrame
    raw_data = raw_data[~raw_data['TenMH'].str.contains('IE|Intensive English|IE2|IE1|IE3|IE0')]
#     raw_data = raw_data[~raw_data['DiemHP'].isin(['P','F','PC'])]
    pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    # Drop unnecessary columns

    # Merge with the XepLoaiNH column
    df = pd.merge(pivot_df, raw_data[['MaSV']], on='MaSV')
    df.drop_duplicates(subset='MaSV', keep='last', inplace=True)
    dfid=df['MaSV']
    df.drop(['MaSV'], axis=1, inplace=True)
    df.replace(['WH', 'VT',"I"], np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    df = pd.merge(dfid,df,left_index=True, right_index=True)
    df['MaSV_school'] = df['MaSV'].str.slice(2, 4)
    df['Major'] = df['MaSV'].str.slice(0, 2)
    df["Year"] = 2000 + df["MaSV"].apply(get_year)
    df["Year"]=df["Year"].astype(str)
    df=df.drop(columns='MaSV')
    
    return df
    
def process_data_per(raw_data):
    # Pivot the DataFrame
    raw_data = raw_data[~raw_data['TenMH'].str.contains('IE|Intensive English|IE2|IE1|IE3|IE0')]
    pivot_df = pd.pivot_table(raw_data, values='DiemHP', index='MaSV', columns='TenMH', aggfunc='first')
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    # Drop unnecessary columns
    
    pivot_df.replace(['WH', 'VT',"I"], np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)


    
    return pivot_df


def process_predict_data(raw_data):
    dtk = raw_data[["MaSV", "DTBTKH4"]].copy()
    dtk.drop_duplicates(subset="MaSV", keep="last", inplace=True)

    count_duplicates = raw_data.groupby(["MaSV", "MaMH"]).size().reset_index(name="Times")
    courses = raw_data[raw_data['MaMH'].str.startswith(('IT', 'BA', 'BM', 'BT', 'MA', 'CE', 'EE', 'EL', 'ENEE', 'IS', 'MAFE', 'PH'))]

    courses_list=courses['MaMH'].unique().tolist()

  # Create two new columns for counting courses that are in the courses_list or not
    count_duplicates["fail_courses_list"] = (
        (count_duplicates["MaMH"].isin(courses_list)) & (count_duplicates["Times"] >= 2)
    ).astype(int)

    count_duplicates["fail_not_courses_list"] = (
        (~count_duplicates["MaMH"].isin(courses_list)) & (count_duplicates["Times"] >= 2)
    ).astype(int)

    count_duplicates["pass_courses"] = (
        (~count_duplicates["MaMH"].isin(courses_list)) & (count_duplicates["Times"] == 1)
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
    data = data.groupby(['MaSV'])['SoTCDat'].median().reset_index(name='Mean_Cre').round(2)

    df = pd.merge(df, data, on='MaSV')
    df1=raw_data[['MaSV','MaMH','NHHK']]
    courses_list = raw_data[(raw_data['MaMH'].str.startswith('EN')) & ~(raw_data['MaMH'].str.contains('EN007|EN008|EN011|EN012'))].MaMH.tolist()
    filtered_df = df1[df1['MaMH'].isin(courses_list)]
    nhhk_counts = filtered_df.groupby('MaSV')['NHHK'].nunique().reset_index(name='EPeriod')
    df = pd.merge(df, nhhk_counts, on='MaSV', how='left').fillna(0)
    df=df[['MaSV','GPA'	,'Mean_Cre',	'fail_courses_list_count'	,'fail_not_courses_list_count'	,'EPeriod']]
    return df

def predict_late_student(test_df):
    # Load the pre-trained model
    model=joblib.load("model/IT/IT_Late.joblib")
    model1=joblib.load("model/IT/IT_Sem.joblib")
    # Process the student data
    test_dfed = process_predict_data(test_df)

    # Save the student ID column
    std_id = test_dfed.iloc[:, 0]

    # Drop the student ID column
    test_dfed = test_dfed.drop(test_dfed.columns[0], axis=1)

    # Make predictions using the pre-trained model
    prediction = model.predict(test_dfed)

    # Add a new column to the student data indicating if the student is late
    

    prediction1 = model1.predict(test_dfed)

    # Add a new column to the student data indicating if the student is late
    test_dfed['Period'] = prediction1
    test_dfed['Result'] = ['late' if p == 1 else 'not late' for p in prediction]

    # Add the student ID column back to the beginning of the DataFrame
    test_dfed.insert(0, 'MaSV', std_id)
    
    for index, row in test_dfed.iterrows():
      if row['Period'] <= 9 and row['Result'] == 'late':
        test_dfed.loc[index, 'Period'] = row['Period'] / 2
        test_dfed.loc[index, 'Result'] = 'may late'
      else:
        test_dfed.loc[index, 'Period'] = row['Period'] / 2

    return test_dfed
# def predict_rank(raw_data):
#     # Pivot the DataFrame
#     raw_data["Major"] = raw_data["MaSV"].str.slice(0, 2)
#     if raw_data["Major"].str.contains("IT").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.contains("IT")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_IT.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/IT/IT_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("BA").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("BA")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_BA.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/BA_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("BT").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("BT")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_BT.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/BT_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("CE").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("CE")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_CE.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/CE_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("EE").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("EE")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_EE.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/EE_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("EN").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("EL")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_EN.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/EN_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("BE").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("BM")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_BE.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/BE_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("IE").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("IS")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_IE.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/IE_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("MA").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("MAFE")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_MA.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/MA_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("SE").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("PH")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.dropna(thresh=50, axis=1)
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_PH.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/PH_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df
#     elif raw_data["Major"].str.contains("EV").any():
#         raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
#         raw_data = raw_data[raw_data["MaMH"].str.startswith("ENEE")]

#         pivot_df = pd.pivot_table(
#             raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
#         )
#         pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
#         pivot_df.columns.name = None
#         pivot_df = pivot_df.rename(columns=lambda x: x.strip())

#         pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
#         pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

#         # Merge with the XepLoaiNH column
#         df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
#         df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
#         col = df.drop(["MaSV", "DTBTK"], axis=1)

#         columns_data = []
#         with open("Columns/column_EV.txt", "r") as f:
#             for line in f:
#                 columns_data.append(str(line.strip()))

#         dup = pd.DataFrame(columns=columns_data)
#         df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
#         for col in df.columns:
#             if df[col].isnull().values.any():
#                 df[col].fillna(value=df["DTBTK"], inplace=True)
#         std_id = df["MaSV"].copy()
#         df = df.drop(["MaSV", "DTBTK"], axis=1)
#         df.sort_index(axis=1, inplace=True)
#         model = joblib.load("model/EV_rank.joblib")
#         prediction = model.predict(df)
#         df["Pred Rank"] = prediction
#         df.insert(0, "MaSV", std_id)
#         df = df[["MaSV", "Pred Rank"]]
#         return df

import pandas as pd
import numpy as np
from sklearn.externals import joblib

def predict_rank(raw_data):
    major, ma_mh = get_major(raw_data)
    if major:
        raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
        raw_data = raw_data[raw_data["MaMH"].str.startswith(ma_mh)]

        pivot_df = create_pivot_table(raw_data)
        pivot_df = drop_nan_columns(pivot_df)

        df = merge_with_xeploainh(pivot_df, raw_data)
        df = fill_missing_values(df)

        std_id = df["MaSV"].copy()
        df = prepare_data(df)

        model = joblib.load(f"model/{major}_rank.joblib")
        prediction = model.predict(df)
        df["Pred Rank"] = prediction

        df.insert(0, "MaSV", std_id)
        df = df[["MaSV", "Pred Rank"]]
        return df
    else:
        return None

def get_major(raw_data):
    major_mapping = {
        "EN": "EL",
        "EV": "ENEE",
        "MA": "MAFE",
        "SE": "PH",
        "IT": "IT",
        "BA":"BA",
        "EE": "EE",
        "BE":"BM",
        "CE":"CE",
        "IE":"IS"
    }
    for major, ma_mh in major_mapping.items():
        if raw_data["Major"].str.contains(major).any():
            return major, ma_mh
    return None, None

def create_pivot_table(raw_data):
    pivot_df = pd.pivot_table(
        raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
    )
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    return pivot_df

def drop_nan_columns(pivot_df):
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)
    return pivot_df

def merge_with_xeploainh(pivot_df, raw_data):
    df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
    df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
    return df

def fill_missing_values(df):
    col = df.drop(["MaSV", "DTBTK"], axis=1)
    columns_data = get_column_data(df)
    dup = pd.DataFrame(columns=columns_data)
    df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
    for col in df.columns:
        if df[col].isnull().values.any():
            df[col].fillna(value=df["DTBTK"], inplace=True)
    return df

def get_column_data(df):
    major = df["Major"].unique()[0]
    column_file = f"Columns/column_{major}.txt"
    columns_data = []
    with open(column_file, "r") as f:
        for line in f:
            columns_data.append(str(line.strip()))
    return columns_data

def prepare_data(df):
    std_id = df["MaSV"].copy()
    df = df.drop(["MaSV", "DTBTK"], axis=1)
    df.sort_index(axis=1, inplace=True)
    return df





def predict_one_student(raw_data, student_id):
    # Subset the DataFrame to relevant columns and rows
      student = process_data_per(raw_data)
      filtered_df = student[student["MaSV"] == student_id]
      if len(filtered_df) > 0:
        selected_row = filtered_df.iloc[0, 1:].dropna()
        colname = filtered_df.dropna().columns.tolist()
        values = selected_row.values.tolist()

      # create a line chart using plotly
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=values, nbinsx=40, name=student_id,marker=dict(color='rgba(50, 100, 200, 0.7)')))

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
        fig2.add_shape(
            type="line",
            x0=0,
            y0=50,
            x1=len(student_data['TenMH'])-1,
            y1=50,
            line=dict(color='red', width=3)
        )

        # display the charts using st.column
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)

        with col2:
            st.plotly_chart(fig2)
      else:
        st.write("No data found for student {}".format(student_id))
