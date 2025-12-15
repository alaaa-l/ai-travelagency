import requests 
import os
from dotenv import load_dotenv


load_dotenv()
from dotenv import load_dotenv

load_dotenv()

def get_travel_cost(origin:str,destination:str,date_of_travel:str) -> str:
    """
    Get the travel cost from origin to destination for given date and number of travelers.
    Returns only the total price as a string.
    """
    url = "https://sky-scrapper.p.rapidapi.com/api/v1/flights/getPriceCalendar"

    querystring = {"originSkyId":origin,
                   "destinationSkyId":destination,
                   "fromDate":date_of_travel
                   }
    headers = {
          "x-rapidapi-key":os.getenv("RAPIDAPI_KEY"),
          "x-rapidapi-host":os.getenv("RAPIDAPI_HOST")
        
    }

    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()

    data = response.json()

    days = data.get("data", {}).get("flights", {}).get("days", [])

    if not days:
        return -1  

    cheapest_price = min(day["price"] for day in days)

    return int(cheapest_price)



