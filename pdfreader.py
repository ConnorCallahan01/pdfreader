# Necessary imports
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
import os
import tempfile
from langchain.document_loaders import TextLoader




@st.cache_data
def load_models():
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    return llm, embeddings, qa_chain, summarize_chain

llm, embeddings, qa_chain, summarize_chain = load_models()

def main():
    st.title("PDF Analyzer")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write the uploaded file to the temporary file
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        
        # Split the text into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(pages)

        # Create a vector store from the chunks
        docsearch = Chroma.from_documents(chunks, embeddings).as_retriever()

        # Display options to the user
        task = st.selectbox("What would you like to do?", ["Ask a question", "Generate a summary"])
        if task == "Ask a question":
            question = st.text_input("What's your question?")
            if question:
                docs = docsearch.get_relevant_documents(question)
                answer = qa_chain.run(input_documents=docs, question=question)
                st.write(answer)
        else:
            st.write("Not supported yet hehe")

if __name__ == "__main__":
    main()






