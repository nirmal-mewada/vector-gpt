import os
from flask import Flask, request, abort
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import openai
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

app = Flask(__name__)

model_name = os.environ["MODEL_NAME"]
k = int(os.environ["K"])
debug = bool(os.environ["DEBUG"])
environment = os.environ["ENVIRONMENT"]
index_name = os.environ["INDEX_NAME"]
auth_key = os.environ["AUTH_KEY"]  # Get the static API key from environment variable
api_key = os.environ["API_KEY"]
pinecone.init(api_key=api_key, environment=environment)

# Initialize LangChain
llm = OpenAI(model_name=model_name)

# Load the LangChain question answering chain
chain = load_qa_chain(llm, chain_type="stuff")

index =  Pinecone.from_existing_index(index_name, embeddings)


# Define the route for the API
@app.route('/api/qna', methods=['GET'])
def question_answering():

    # Get the query from the request
    query = request.args.get('query')
    auth_k = request.args.get('auth_key')
    print('Query: ' + query)
    # Verify the API key
    if auth_key != auth_k:
        abort(401, 'Unauthorized')

    # Get the answer
    answer = get_answer(query,score=True)

    # Return the answer as a JSON response
    return {'answer': answer}


# Define the function to get the answer to a query
def get_answer(query, score=False):
    # Get the similar documents
    similar_docs = get_similar_docs(query, score)

    # Create a list of documents
    docs = []
    for d in similar_docs:
        if debug:
            print(str(d[0].metadata) + ' - ' + str(d[1]) + ' - ' + str(d[0].page_content))
        docs.append(d[0])

    # Get the answer from the question answering chain
    answer = chain.run(input_documents=docs, question=query)

    # Return the answer
    return answer


# Define the function to get the similar documents
def get_similar_docs(query, score=False):

    # Get the similar documents from the index
    if score:
        return index.similarity_search_with_score(query, k=k)
    else:
        return index.similarity_search(query, k=k)


if __name__ == '__main__':
    print('Starting App')
    app.run(debug=True, port=5000, host='0.0.0.0')

