#!/usr/bin/env python3
""" Where can I learn Python? """


def schools_by_topic(mongo_collection, topic):
    """ function that returns the list of school having a specific topic

        mongo_collection will be the pymongo collection object
        topic (string) will be topic searched
    """
    match = []
    results = mongo_collection.find({"topics": {"$all": [topic]}})
    for result in results:
        match.append(result)
    return match
