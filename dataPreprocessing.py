import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

def get_db_connection():
    
    user = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    host = os.getenv('MYSQL_HOST')
    port = os.getenv('MYSQL_PORT')
    database = os.getenv('MYSQL_DB')

    safe_password = urllib.parse.quote_plus(password)

    db_url = f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    return engine

def query_data(query):
    try:
        engine = get_db_connection()
        df_result = pd.read_sql(query,con=engine)
        return df_result
    except Exception as e:
        print(f"Error in querying data: {e}")
        return None

my_query = "select * from monthly_expenditure where UCC in ('690114','690111','690120','270102','270106','690113'," \
"'690114','690310','300311','300312','300321','300322','300331','300332','320522','320232','690117','690119','690116'," \
"'480100','480213','490501','310316','310140','270310','620930','310231','310232','310400','340610','340902','310314'," \
"'310350','610130','310243','620917','620918','310333','690320','690330','590230','690118','300311','300312','300321'," \
"'300322','320331','320332','320522','320232','690111','690117','690119','690120','690115','690116','690210','270106','690310'," \
"'620930','270310','310140','310231','310232','620917','620918','310243','310400','310316','310314','610130','310333','310350') and REF_YR = 2024"

df_2024_data = query_data(my_query)

df_2024_data.drop_duplicates(inplace=True)

df_2024_data.dropna(inplace = True)

ucc_2024_map ={
    # Telecommunications
    270102: "Cellular phone service",
    270106: "Residential telephone including VOIP",
    270310: "Cable and satellite television services",
    690114: "Computer information services (internet)",
    690116: "Internet services away from home",
    
    # Computing Hardware & Accessories
    690111: "Computers and computer hardware for nonbusiness use",
    690117: "Portable memory",
    690120: "Computer accessories",
    690115: "Personal digital assistants",
    320232: "Telephones and accessories",
    690210: "Telephone answering devices",
    
    # Software & Digital Services
    690119: "Computer software",
    620930: "Online gaming services",
    310400: "Applications, games, and ringtones for handheld devices",
    
    # Computing Services
    690113: "Repair of computer systems for nonbusiness use",
    690310: "Installation of computers",
    
    # Streaming & Digital Media
    310350: "Streaming and downloading audio",
    310243: "Rental, streaming, and downloading videos",
    620917: "Rental of video hardware/accessories",
    620918: "Rental of video software",
    
    # Gaming
    310231: "Video game software",
    310232: "Video game hardware and accessories",
    
    # Audio/Visual Equipment
    310140: "Televisions",
    310316: "Stereos, radios, speakers, and sound components",
    310314: "Personal digital audio players",
    310333: "Accessories and other sound equipment",
    340610: "Repair of televisions, radio, and sound equipment",
    340902: "Rental of televisions",
    690320: "Installation of televisions",
    690330: "Installation of satellite television equipment",
    
    # Musical Instruments
    610130: "Musical instruments and accessories",
    
    # Digital Reading
    590230: "Books, digital books, or book subscriptions",
    690118: "Digital book readers",
    
    # Appliances (Non-Digital - appear to be duplicates/errors in original list)
    300311: "Cooking stoves and ovens (renter)",
    300312: "Cooking stoves and ovens (owned home)",
    300321: "Microwave ovens (renter)",
    300322: "Microwave ovens (owned home)",
    300331: "Portable dishwashers (renter)",
    300332: "Portable dishwashers (owned home)",
    320522: "Portable heating and cooling equipment",
    
    # Vehicle Accessories (Non-Digital - appear to be errors in original list)
    480100: "Vehicle parts, accessories, fluid excluding tires",
    480213: "Parts, equipment, and accessories",
    490501: "Vehicle accessories including labor",

}

df_2024_data['UCC_Description'] = df_2024_data['UCC'].map(ucc_2024_map).fillna(df_2024_data['UCC'])

df_2024_data.drop(columns=['ALCNO','PUBFLAG','UCCSEQ'],inplace=True)

df_2024_data['GIFT'] = df_2024_data['GIFT'].replace({1: True, 2: False}).astype(bool)

print(df_2024_data.head())