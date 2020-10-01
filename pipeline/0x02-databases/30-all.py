#!/usr/bin/env python3
""" List all documents in Python """


def list_all(mongo_collection):
    """ ists all documents in a collection:

        Return an empty list if no document in the collection
        mongo_collection will be the pymongo collection object
    """
    documents = []
    collection = mongo_collection.find()
    for doc in collection:
        documents.append(doc)
    return documents
