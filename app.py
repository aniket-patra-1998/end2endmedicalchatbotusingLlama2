from flask import Flask, render_template, jsonify,request
from src.helper import download_embedding
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import pinecone
import langchain
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embedding = download_embedding()

#Initializing the Pinecone
pc = pinecone.Pinecone(api_key = PINECONE_API_KEY)

index_name = "chatbot"


#Loading the index
index = pc.Index(index_name)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


#qa=RetrievalQA.from_chain_type(
    #llm=llm, 
    #chain_type="stuff", 
    #retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    #return_source_documents=True, 
    #chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    query1 =embedding.embed_query( input)
    data = index.query(namespace="real",
            vector=query1,
            top_k=3,
            include_metadata=True,
            include_values=False,
            filter = {}
            )
    texts = [match['metadata']['text'] for match in data['matches']]
    query_text = "".join(texts)
    p = PROMPT.format(context =query_text, question = input)
    result=llm(p)
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



