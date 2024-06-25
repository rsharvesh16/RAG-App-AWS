
# RAG - Chat with PDF using AWS Bedrock

<div align="center">
<img width="446" alt="image" src="Chat-PDF (4).png">
</div>

This project demonstrates how to use the AWS Bedrock Titan Embeddings model to generate embeddings, store them in a FAISS vector store, and interact with them using a Streamlit app. You can ask questions about PDF documents, and the app will provide detailed answers using an LLM model (Mistral) from AWS Bedrock.

## Prerequisites

Before you can run this code, ensure you have the following installed:

- Python 3.7+
- Required Python packages:
  - `boto3`
  - `streamlit`
  - `langchain`
  - `langchain_community`
  - `numpy`
  - `faiss-cpu`
  - `pypdf`
  - `awscli`

You will also need AWS credentials configured to access AWS Bedrock.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rsharvesh16/RAG-App-AWS.git
    cd RAG-App-AWS
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure your AWS credentials are set up correctly to access AWS Bedrock services.

## Project Structure

- `app.py`: The main script to run the Streamlit app.
- `data/`: Directory where your PDF files should be placed.

## How to Run

1. Place your PDF files in the `data/` directory.

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Open your browser and go to the URL provided by Streamlit (usually http://localhost:8501).

## Notes

- Ensure your PDF files are in the `data/` directory.
- The script uses AWS Bedrock services, so make sure your AWS credentials are properly configured.

## Example

1. Place your PDF files in the `data/` directory.
2. Run the Streamlit app.
3. In the sidebar, click "Vectors Update" to process the PDF files.
4. Ask a question in the main interface and click "Mistral Output" to get the answer.

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.
