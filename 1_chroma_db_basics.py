#export HNSWLIB_NO_NATIVE=1

# open source, pretty scalable for production

import chromadb

chroma_client = chromadb.Client()

collections = chroma_client.create_collection(name = "my_collection")

collections.add(
    documents = ["my name is akshath","my name is marc"],
    metadatas = [{"source":"my_source"},{"source":"my_source"}],
    ids = ["id1","id2"]
)

results = collections.query(
    query_texts = ["what is my name?"],
    n_results = 2
)

print(results)

#Clear Cache: Sometimes, cached files can become corrupted. You can try clearing the cache for chromadb by deleting the contents of the C:\Users\user-upcnb00067\.cache\chroma\ directory. Afterward, rerun your code, and the library should attempt to redownload or regenerate the necessary files.