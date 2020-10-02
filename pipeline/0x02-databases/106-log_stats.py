#!/usr/bin/env python3
""" og stats - new version """
from pymongo import MongoClient


if __name__ == "__main__":
    """ script that provides some stats about Nginx logs stored in MongoDB:

        Database: logs
        Collection: nginx
        Display:

        first line: x logs where x is the number of documents in this
            collection
        second line: Methods:
        5 lines with the number of documents with the
            method = ["GET", "POST", "PUT", "PATCH", "DELETE"] in this order
        one line with the number of documents with:
            method=GET
            path=/status
        adding the top 10 of the most present IPs in the collection nginx
            of the database logs
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    num_doc = logs_collection.count_documents({})
    print("{} logs".format(num_doc))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in methods:
        num_method = logs_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_method))
    filter_path = {"method": "GET", "path": "/status"}
    num_path = logs_collection.count_documents(filter_path)
    print("{} status check".format(num_path))
    print("IPs:")
    pipeline = [{"$group": {"_id": "$ip", "count": {"$sum": 1}}}]
    ips = logs_collection.aggregate(pipeline)
    list_ips = []

    for ip in ips:
        list_ips.append(ip)
    sorted_ips = sorted(list_ips, key=lambda i: i["count"], reverse=True)
    i = 0
    limit = 10
    if len(sorted_ips) < 10:
        limit = len(sorted_ips)
    while i < limit:
        ip = sorted_ips[i]["_id"]
        count = sorted_ips[i]["count"]
        print("\t{}: {}".format(ip, count))
        i += 1
