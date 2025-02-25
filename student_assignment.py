import datetime
import chromadb
import traceback
import pandas

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = 'COA_OpenData.csv'

def generate_hw01(clear_if_exist = False):
    # Create embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )

    # Create chromadb
    chroma_client = chromadb.PersistentClient(path = dbpath)

    # Clear old collection if need
    if clear_if_exist == True:
        chroma_client.delete_collection(name = "TRAVEL")

    # Create new collection to store or retrieve data
    collection = chroma_client.get_or_create_collection(
        name = "TRAVEL",
        metadata = {"hnsw:space": "cosine"},
        embedding_function = openai_ef
    )

    if collection.count == 0:
        # Read data from csv file
        data = pandas.read_csv(csv_file)
        for row in data.iterrows():
            id = row["ID"]
            metadata = {
                "file_name": csv_file,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row["CreateDate"], '%Y-%m-%d').timestamp())
            }
            document = row["HostWords"]

            # Add metadata and document to the collection
            collection.add(ids = id, metadatas = metadata, documents = document)

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()

    # Construct filters
    filters = []
    if city:
        filters.append({'city': {'$in': city}})
    if store_type:
        filters.append({'type': {'$in': store_type}})
    if start_date:
        filters.append({'date': {'$gte': int(start_date.timestamp())}})
    if end_date:
        filters.append({'date': {'$lte': int(end_date.timestamp())}})

    # Combine filters
    # {'$and': [{'city': {'$in': city}}, {'type': {'$in': store_type}}, ...]}
    where = {}
    if filters:
        where = {'$and': filters}

    # Perform the query based on the given filters
    results = collection.query(
        query_texts = [question],
        n_results = 10,
        where = where
    )

    # Filter results with similarity >= 0.8
    filtered_results = []
    for i in range(len(results['ids'])):
        for distance, metadata in zip(results['distances'][i], results['metadatas'][i]):
            cosine_similarity = 1 - distance
            if cosine_similarity >= 0.8:
                filtered_results.append(metadata["name"])

    return filtered_results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    update_store_name(
        store_name = store_name,
        new_store_name = new_store_name)

    collection = generate_hw01()

    # Construct filters
    filters = []
    if city:
        filters.append({'city': {'$in': city}})
    if store_type:
        filters.append({'type': {'$in': store_type}})

    # Combine filters
    # {'$and': [{'city': {'$in': city}}, {'type': {'$in': store_type}}, ...]}
    where = {}
    if filters:
        where = {'$and': filters}

    # Perform the query based on the given filters
    results = collection.query(
        query_texts = [question],
        n_results = 10,
        where = where
    )

    # Filter results with similarity >= 0.8
    filtered_results = []
    for i in range(len(results['ids'])):
        for distance, metadata in zip(results['distances'][i], results['metadatas'][i]):
            cosine_similarity = 1 - distance
            if cosine_similarity >= 0.8:
                if "new_store_name" in metadata:
                    filtered_results.append((metadata["new_store_name"], cosine_similarity))
                else:
                    filtered_results.append((metadata["name"], cosine_similarity))

    filtered_results.sort(key = lambda x: x[1], reverse = True)
    return [name for name, _ in filtered_results]

def update_store_name(store_name, new_store_name):
    collection = generate_hw01()

    results = collection.query(
        query_texts=[store_name],
        n_results=1
    )
    for i in range(len(results['ids'])):
        for metadata in results['metadatas'][i]:
            if metadata['name'] == store_name:
                metadata['new_store_name'] = new_store_name
                collection.update(
                    ids = results['ids'][i],
                    metadatas = [metadata]
                )
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
