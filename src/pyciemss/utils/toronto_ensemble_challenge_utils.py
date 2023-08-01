# Load inital dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

US_regions = ['US', 'AL', 'AK', 'Skip', 'AZ', 'AR', 'CA', 'Skip 2', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'Skip 3',
              'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
              'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'Skip 4', 'RI', 'SC', 'SD', 'TN',
              'TX', 'UT', 'VT', 'VA', 'Skip 5', 'WA', 'WV', 'WI', 'WY']
fips_dict = {state: fips for state, fips in zip(US_regions, range(0, 100))}
fips_dict["US"] = "US"


def get_fips(US_region):
    '''This function returns the 1 or 2 digit FIPS code corresponding to the 2-letter state abbreviation.

    :param US_state: 2-letter state abbreviation as a string
    :returns: 1 or 2 digit FIPS code as a string
    '''
    return str(fips_dict[US_region])

def get_all_data():

    # Get incident case data (by county) and sort by date
    url = 'https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Cases.csv'
    raw_cases = pd.read_csv(url)
    raw_cases['date'] = pd.to_datetime(raw_cases.date, infer_datetime_format=True)
    raw_cases.sort_values(by='date', ascending=True, inplace=True)

    # Get hosp census data (by state) and sort by date
    url = 'https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Hospitalizations.csv'
    raw_hosp = pd.read_csv(url)
    raw_hosp['date'] = pd.to_datetime(raw_hosp.date, infer_datetime_format=True)
    raw_hosp.sort_values(by='date', ascending=True, inplace=True)

    # Get cumulative death data (by county) and sort by date
    url = 'https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Cumulative%20Deaths.csv'
    raw_deaths = pd.read_csv(url)
    raw_deaths['date'] = pd.to_datetime(raw_deaths.date, infer_datetime_format=True)
    raw_deaths.sort_values(by='date', ascending=True, inplace=True)
    
    return raw_cases, raw_hosp, raw_deaths

def get_case_hosp_death_data(US_region, infectious_period, make_csv=True):
    '''This function returns the number of cases, hospitalizations, and deaths for a given US region and infectious period.

    :param US_region: 2-letter state abbreviation as a string
    :param infectious_period: number of days infectiousness is assumed to be
    :returns: number of cases, hospitalizations, and deaths for a given US region and infectious period
    '''
    # Get the FIPS code for the given region as a string
    fips_code = get_fips(US_region)
    raw_cases, raw_hosp, raw_deaths = get_all_data()

    # Select data for the given region
    regional_hosp = raw_hosp[raw_hosp["location"] == fips_code]
    if fips_code == "US":
        regional_cases = raw_cases[raw_cases["location"] == "US"]
        regional_deaths = raw_deaths[raw_deaths["location"] == "US"]
    elif len(fips_code) == 1:
        regional_cases = raw_cases[(raw_cases["location"].astype(str).str.len() == 4.0) & (
                    raw_cases["location"].astype(str).str[:1] == fips_code)]
        regional_deaths = raw_deaths[(raw_deaths["location"].astype(str).str.len() == 4.0) & (
                    raw_deaths["location"].astype(str).str[:1] == fips_code)]
    elif len(fips_code) == 2:
        regional_cases = raw_cases[(raw_cases["location"].astype(str).str.len() == 5.0) & (
                    raw_cases["location"].astype(str).str[:2] == fips_code)]
        regional_deaths = raw_deaths[(raw_cases["location"].astype(str).str.len() == 5.0) & (
                    raw_deaths["location"].astype(str).str[:2] == fips_code)]

    # Set up DataFrame to hold COVID data and convert incident cases to case census and
    regional_cases["case census"] = 0
    regional_cases = regional_cases.reset_index(drop=True)
    regional_hosp = regional_hosp.reset_index(drop=True)
    regional_hosp = regional_hosp.groupby("date")["value"].sum()
    regional_hosp = pd.DataFrame({"hosp_census": regional_hosp})

    regional_deaths = regional_deaths.reset_index(drop=True)
    regional_deaths = regional_deaths.groupby("date")["value"].sum()
    regional_deaths = pd.DataFrame({"cumulative_deaths": regional_deaths})

    covid_data_df = {}
    covid_data_df["date"] = regional_cases["date"].unique()
    covid_data_df["value"] = regional_cases.groupby("date")["value"].sum()
    covid_data_df = pd.DataFrame(covid_data_df)
    covid_data_df["case_census"] = covid_data_df["value"].rolling(infectious_period, min_periods=1).sum()
    covid_data_df = covid_data_df.drop(columns=["value"])
    covid_data_df = covid_data_df.set_index("date")

    # Add hosp and death data to covid_data_df
    covid_data_df = pd.merge(covid_data_df, regional_hosp, how="outer", left_index=True, right_index=True)
    covid_data_df = pd.merge(covid_data_df, regional_deaths, how="outer", left_index=True, right_index=True)

    if make_csv:
        filename = US_region + "_case_hospital_death.csv"
        covid_data_df.to_csv(filename, index=True, header=True)

    return covid_data_df


def get_train_test_data(data: pd.DataFrame, train_start_date: str, test_start_date: str,
                        test_end_date: str) -> pd.DataFrame:
    train_df = data[(data['date'] >= train_start_date) & (data['date'] < test_start_date)]
    train_data = [0] * train_df.shape[0]
    start_time = train_df.index[0]

    train_cases = np.array(train_df["case_census"].astype("float"))  # / data_total_population
    train_timepoints = np.array(train_df.index.astype("float"))

    test_df = data[(data['date'] >= test_start_date) & (data['date'] < test_end_date)]
    test_cases = np.array(test_df["case_census"].astype("float"))  # / data_total_population
    test_timepoints = np.array(test_df.index.astype("float"))

    for time, row in train_df.iterrows():
        row_dict = {}
        row_dict["Cases"] = row["case_census"]  # / data_total_population
        row_dict["Deaths"] = row["cumulative_deaths"]  # / data_total_population
        if row["hosp_census"] > 0:
            row_dict["Hospitalizations"] = row["hosp_census"]  # / data_total_population

        index = time - start_time
        train_data[index] = (float(time), row_dict)

    all_timepoints = np.concatenate((train_timepoints, test_timepoints))

    return train_data, train_cases, train_timepoints, test_cases, test_timepoints, all_timepoints

def train_data_to_csv(train_data, data_file_name):
    # Get training data in the correct form for ensemble model calibration
    ensemble_data = pd.DataFrame(train_data, columns = ["Timestep", "Data"])
    case_list = []
    hosp_list = []
    death_list = []
    for i in range(0, len(ensemble_data)):
        case_list.append(ensemble_data.iloc[i]["Data"]["Cases"])
        if "Hospitalizations" in ensemble_data.iloc[i]["Data"].keys():
            hosp_list.append(ensemble_data.iloc[i]["Data"]["Hospitalizations"])
        else: 
            hosp_list.append(float("NaN"))
        death_list.append(ensemble_data.iloc[i]["Data"]["Deaths"])
    ensemble_data["Cases"] = case_list
    ensemble_data["Hospitalizations"] = hosp_list
    ensemble_data["Deaths"] = death_list
    ensemble_data = ensemble_data.drop(columns=["Data"])
    ensemble_data.to_csv(data_file_name, index=False, header=True)
    
    return None