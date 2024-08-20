from flask import Flask, request, jsonify, render_template
from time import time
import torch
import transformers
from transformers import AutoTokenizer
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from transformers import AutoConfig
from langchain.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

## FILL BELOW PART WITH INFORMATION OF YOUR DATASET AND MODEL
dataset_name = 'sample_dataset' # fill with dataset directory name/location
doc_column = 'text' # fill with column name that contains text data
emb_model='LazarusNLP/all-indo-e5-small-v4' # fill with embedding model name
tg_model = 'kalisai/Nusantara-7b-Indo-Chat' # fill with text-generation model name
## FILL ABOVE PART WITH INFORMATION OF YOUR DATASET AND MODEL

# Set the device
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Set the quantization config
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Set Model Config
model_config = transformers.AutoConfig.from_pretrained(
    tg_model,
    trust_remote_code=True,
    max_new_tokens=1024,
)

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    tg_model,
    trust_remote_code=True,
    quantization_config=bnb_config,
    config=model_config
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tg_model)

# Define QA pipeline
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=2048
)
llm = HuggingFacePipeline(pipeline=query_pipeline)

# Function to load dataset and embed document within it
def embed_text(dataset_name: str=dataset_name,document_column:str=doc_column,embedding_model:str=emb_model) -> pd.DataFrame:
    df = pd.read_csv(f"{dataset_name}.csv")
    df[document_column] = df[document_column].fillna('')
    embedder = SentenceTransformer(embedding_model)
    df['embedded_doc'] = df[document_column].apply(lambda text: embedder.encode(text))
    return df

# Create a collection and add documents to it
persistent_clients = chromadb.PersistentClient()
collections = persistent_clients.get_or_create_collection("private-llm")

for index, row in embed_text().iterrows():
    collections.add(
        documents=row[doc_column],
        embeddings=row['embedded_doc'].tolist(),
        metadatas=[row.drop([doc_column, 'embedded_doc']).to_dict()],
        ids=[str(index)]
    )
# Integrate ChromaDB to LangchainChroma and create a retriever
embeddings = HuggingFaceEmbeddings(model_name='LazarusNLP/all-indo-e5-small-v4')
langchain_chroma = Chroma(client=persistent_clients, collection_name="private-llm", embedding_function=embeddings)
retriever = langchain_chroma.as_retriever()

# Create a QA model Pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=False
)

# Handle cases where the response is not in the usual format
def handle_unusual_cases(response):
    start_index = response.find("Helpful Answer:")
    end_index = response.find("assistant")
    if start_index != -1:
        response = response[start_index+16:]
    if end_index != -1:
        response = response[:end_index]
    return response

# Main function
def ask(qa, query):
    time_start = time()
    response = handle_unusual_cases(qa.run(query))
    time_end = time()
    total_time = round(time_end - time_start, 3)
    print(f"Debug info - Query: {query}")
    print(f"Debug info - Response: {response}")
    print(f"Debug info - Total time: {total_time} sec.")
    torch.cuda.empty_cache()
    return response

# Define the API endpoint
@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    try:
        data = request.json
        print("Received data:", data)  # Log received data
        query = data.get('query')
        if query:
            response = ask(qa, query)
            return jsonify({'response': response})
        else:
            return jsonify({'response': 'Invalid input'}), 400
    except Exception as e:
        print("Error:", e)  # Log any exception that occurs
        return jsonify({'response': 'Internal server error'}), 500

@app.route('/')
def index():
    # Serve the HTML file
    return render_template('web\src\index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
