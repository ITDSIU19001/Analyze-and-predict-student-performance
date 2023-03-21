import pickle
import pandas as pd

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