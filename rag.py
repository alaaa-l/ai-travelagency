from loading import load_documents_from_folder
from chunking import chunk_documents
from embeddings import embed_texts
from v_db import get_db_collection
from similarity import retrieve_relevant_chunks
from prompt import prepare_prompt
from call_llm import generate_answer
import os

from dotenv import load_dotenv
load_dotenv()

#step 1: load existing files
source_list = load_documents_from_folder("sample_docs/")

#step 2: chunk the contents
my_chunks_with_metadata = chunk_documents(source_list)

# Prepare data for storage
ids_list = [f"chunk_{i}" for i in range(len(my_chunks_with_metadata))]
text_list = []
metadata_list = []

for chunk in my_chunks_with_metadata:
    text_list.append(chunk["text"])
    metadata_list.append({
        'source': chunk['source'],
        'doc_id': chunk['doc_id'],
        'chunk_id': chunk['chunk_id']
    })

#step 3: generate embeddings
vectors_list = embed_texts(text_list)

#step 4: store into vector_db
my_rag_collection = get_db_collection()
my_rag_collection.upsert(
        ids = ids_list,
        embeddings = vectors_list,
        documents = text_list,
        metadatas = metadata_list
    )
print("\n" + "=" * 25)
print(f"STEP 4: Embedding: successfully added {my_rag_collection.count()} into vector database")
print("=" * 25)

#step 5: write query and generate the embeddings of the query
user_question = input("Enter your questions / query here: Country?")
question_list = []
question_list.append(user_question)

question_vector = embed_texts(question_list)

#step 6: perform semantic / similarity search to get relevant chunks
result = retrieve_relevant_chunks(question_vector, my_rag_collection, 3) #pick only top 3

#step 7: prepare a prompt
prompt = prepare_prompt(user_question, result['documents'][0])

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found. Is your .env file loaded?")


#step 8: call deepseek and get an answer
answer = generate_answer(prompt, api_key)
