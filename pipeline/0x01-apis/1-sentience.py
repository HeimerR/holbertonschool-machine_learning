#!/usr/bin/env python3
"""  Where I am?  """
import requests

def sentientPlanets():
    """  returns the list of ships that can hold a
         given number of passengers

         If no ship available, return an empty list.
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []
    while url != None:
        r = requests.get(url)
        results = r.json()["results"]
        for specie in results:
            if (specie["designation"] == "sentient" or
                specie["classification"] == "sentient"):

                planet_url = specie["homeworld"]
                if planet_url != None:
                    p = requests.get(planet_url).json()
                    planets.append(p["name"])
        url = r.json()["next"]
    return planets
