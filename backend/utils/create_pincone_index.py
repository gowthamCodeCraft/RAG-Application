import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag-hybrid-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension = 384,
        metric= "dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"[INFO] Index {index_name} Created Successfully!!!")
else:
    print(f"[INFO] Index {index_name} already exists")