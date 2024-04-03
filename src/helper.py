from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings







#extract data from pdf
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# create chunks
def text_split(extracted_data):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap =20)
    text_chunks = chunks.split_documents(extracted_data)
    return text_chunks



# download embeding model
def download_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding



