import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib.parse

# load environment variables
load_dotenv()

def upload_data_to_db(file_name,table_name):

    try:
        user = os.getenv('MYSQL_USER')
        password = os.getenv('MYSQL_PASSWORD')
        host = os.getenv('MYSQL_HOST')
        port = os.getenv('MYSQL_PORT')
        database = os.getenv('MYSQL_DB')

        safe_password = urllib.parse.quote_plus(password)

        db_url = f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{database}"
        print(db_url)
        engine = create_engine(db_url)

        columns_to_use = ['NEWID','QINTRVYR','QINTRVMO','REGION','DIVISION','POPSIZE','PSU','URBAN','STATE','FINLWT21','AGE_REF','SEX_REF','REF_RACE','EDUC_REF',
                          'MARITAL1','FAM_SIZE','FAM_TYPE','FINCBTAX','FINCBTXM','TOTEXPCQ','TOTEXPPQ','NO_EARNR']

        print(f"Reading file {file_name}......")
        df = pd.read_csv(file_name, usecols=columns_to_use)

        df = df.rename(columns={'QINTRVYR':'interview_year','QINTRVMO':'interview_month','FINLWT21':'calibration_weight',
                                'MARITAL1':'martial_status','FINCBTAX':'family_income_before_tax','FINCBTXM':'imputed_income_before_tax',
                                'TOTEXPCQ':'total_expenditure_current_quarter','TOTEXPPQ':'total_expenditure_prior_quarter','NO_EARNR':'number_of_earners'})



        df.to_sql(table_name,con=engine,if_exists='append',index=False)
        print(f"Successfully uploaded {file_name} to table {table_name}")

    except Exception as e:
        print(f"An error occurred while processing the file:{file_name}: {e}")


if __name__ == "__main__":
    files_to_collect ={
        "/home/alain/Documents/Data science notes/intrvw24/fmli242.csv" : "household_2024_metadata",
        "/home/alain/Documents/Data science notes/intrvw24/fmli243.csv" : "household_2024_metadata",
        "/home/alain/Documents/Data science notes/intrvw24/fmli244.csv" : "household_2024_metadata",
        "/home/alain/Documents/Data science notes/intrvw24/fmli251.csv" : "household_2024_metadata"
    }

    for file, table in files_to_collect.items():
        upload_data_to_db(file,table)




