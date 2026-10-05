# Document RAG Assistant

This project provides a Streamlit UI for asking questions about uploaded PDF and
TXT documents. It uses local Hugging Face embeddings, Chroma for similarity
search, and Groq for answer generation.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq key to `.env`:

   ```dotenv
   GROQ_API_KEY=your_key_here
   ```

4. Start the UI from this directory:

   ```bash
   streamlit run app.py
   ```

Upload text-based documents, click **Build index**, and then ask questions in
the chat. Answers include the retrieved source sections used as context.
