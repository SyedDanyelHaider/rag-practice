
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings



from models import get_session, Article
from config import OPENAI_API_KEY, EMBEDDINGS_MODEL_NAME
from utils import setup_logging
import openai
import pickle
import faiss
import os
from joblib import Parallel, delayed
import requests
import json

logger = setup_logging()

def load_vector_store():
    logger.info('Loading FAISS vector store...')
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    index_dir = 'faiss_index'
    index_path = os.path.join(index_dir, 'index.faiss')
    docstore_path = os.path.join(index_dir, 'docstore.pkl')
    index_to_docstore_id_path = os.path.join(index_dir, 'index_to_docstore_id.pkl')

    index = faiss.read_index(index_path)

    with open(docstore_path, 'rb') as f:
        docstore = pickle.load(f)

    with open(index_to_docstore_id_path, 'rb') as f:
        index_to_docstore_id = pickle.load(f)

    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    logger.info('FAISS vector store loaded.')
    return vector_store, embeddings_model


def create_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    return retriever

def get_similar_articles(text, retriever, article_dict):
    logger.info(f'Getting similar articles for: {text}')
    try:
        # docs = retriever.get_relevant_documents(text)
        docs = retriever.invoke(text)
        results = []
        for doc in docs:
            article_id = int(doc.metadata['id'])
            article = article_dict.get(article_id)
            if article:
                results.append({
                    'id': article.id,
                    'title': article.title,
                    'content': article.content,
                    'author': article.author,
                    'tags': article.tags,
                    'created_at': article.created_at.isoformat()
                })
        return results
    except Exception as e:
        logger.error(f'Error retrieving similar articles: {e}')
        return []

# def generate_summary(context):
#     openai.api_key = OPENAI_API_KEY
#     try:
#         response = openai.ChatCompletion.create(
#             model='gpt-4o',  
#             messages=[
#                 {"role": "user", "content": f"Summarize the following content:\n\n{context}"}
#             ],
#             max_tokens=150,
#             temperature=0.7
#         )
#         summary = response.choices[0].message['content'].strip()
#         return summary
#     except Exception as e:
#         logger.error(f'Error generating summary: {e}')
#         return f"Error generating summary: {e}"


def generate_summary(context):
    # openai.api_key = OPENAI_API_KEY
    try:
        # response = openai.ChatCompletion.create(
        #     model='gpt-4o',  
        #     messages=[
        #         {"role": "user", "content": f"Summarize the following content:\n\n{context}"}
        #     ],
        #     max_tokens=150,
        #     temperature=0.7
        # )
        # summary = response.choices[0].message['content'].strip()
        # response = client.chat(model='llama3.2', messages=[
        #         {
        #             'role': 'user',
        #             'content': f'Please summraize this article: \n {context}',
        #         },
        #         ])
        # summary=response.message.content

        url = "https://tlikdz9ypqfwaj-11434.proxy.runpod.net/api/generate"

        payload = {
            "model": "llama3.2",
            "prompt": f"Please summarize this article: \n {context}"
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        # Extracting the text response from streaming output
        response_lines = response.text.strip().split("\n")  # Each line is a JSON object
        summary = "".join(json.loads(line)["response"] for line in response_lines if line.strip())
        return summary
    except Exception as e:
        logger.error(f'Error generating summary: {e}')
        return f"Error generating summary: {e}"


def generate_summaries(contexts):
    logger.info('Generating summaries...')
    summaries = []
    for context in contexts:
        summary = generate_summary(context)
        summaries.append(summary)
    logger.info('Summaries generated.')
    return summaries
