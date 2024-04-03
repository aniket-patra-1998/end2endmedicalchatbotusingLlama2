from src.helper import load_pdf, text_split, download_embedding
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
pinecone_api_key = os.environ.get('pinecone_api_key')
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding = download_embedding()
# initialize pinecone
pc = pinecone.Pinecone(api_key = pinecone_api_key)
pc.list_indexes()
name = 'chatbot'
index = pc.Index(name)

# embed each chunk and upsert it to index
for i,t in zip(range(len(text_chunks)),text_chunks):
    query = embedding.embed_query(t.page_content)
    index.upsert(
        vectors=[
            {
                "id":str(i),
                "values":query,
                "metadata":{"text":str(text_chunks[i].page_content)}
            }
        ],

        namespace="real"
    )