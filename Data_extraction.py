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

        columns_to_use = ['NEWID','UCC','EXPNAME','COST','REF_MO','REF_YR']

        print(f"Reading file {file_name}......")
        df = pd.read_csv(file_name, usecols=columns_to_use)

        df = df.rename(columns={'REF_MO':'reference_month','REF_YR':'reference_year'})



        df.to_sql(table_name,con=engine,if_exists='append',index=False)
        print(f"Successfully uploaded {file_name} to table {table_name}")

    except Exception as e:
        print(f"An error occurred while processing the file:{file_name}: {e}")


files_to_collect ={
        "/home/alain/Documents/Data science notes/intrvw22/mtbi222.csv" : "monthly_expenditure",
        "/home/alain/Documents/Data science notes/intrvw22/mtbi223.csv" : "monthly_expenditure",
        "/home/alain/Documents/Data science notes/intrvw22/mtbi224.csv" : "monthly_expenditure",
        "/home/alain/Documents/Data science notes/intrvw22/mtbi231.csv" : "monthly_expenditure"



    }

for file, table in files_to_collect.items():
        upload_data_to_db(file,table)