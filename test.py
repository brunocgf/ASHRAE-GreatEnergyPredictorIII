import requests
import pyodbc
import json
from datetime import datetime,timedelta
import pandas as pd


hoy = (datetime.now().date()+ timedelta(days=-7)).strftime("%Y/%m/%d")
print(hoy)