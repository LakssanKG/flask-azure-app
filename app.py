import os
import requests
from flask import Flask, request, jsonify, render_template
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

# Azure Configuration (Use environment variables for security)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

# Upload File to Azure Blob Storage
def upload_file_to_blob(file_path, container_name="uploads"):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=os.path.basename(file_path))
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    return blob_client.url

# Extract Text Using Azure Document Intelligence
def extract_text(file_url):
    client = DocumentIntelligenceClient(DOCUMENT_INTELLIGENCE_ENDPOINT, AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY))
    poller = client.begin_analyze_document("prebuilt-read", {"urlSource": file_url})
    result = poller.result()
    
    text_content = "\n".join([line.content for page in result.pages for line in page.lines])
    return text_content

# Summarize Text Using Azure OpenAI
def summarize_text(text):
    headers = {"Authorization": f"Bearer {AZURE_OPENAI_KEY}", "Content-Type": "application/json"}
    data = {
        "messages": [{"role": "system", "content": "Summarize the following document in key points."},
                     {"role": "user", "content": text}],
        "max_tokens": 500
    }
    
    response = requests.post(f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")  # Ensure templates/index.html exists

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)
    file_url = upload_file_to_blob(file_path)
    extracted_text = extract_text(file_url)
    summary = summarize_text(extracted_text)
    
    return jsonify({"filename": file.filename, "summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
