import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
import random
import time

# Load environment variables
load_dotenv()

# Set up LangSmith (optional, for tracking)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"

# CSS styles
css = '''
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Roboto', sans-serif;
}
header {
    background-color: #4CAF50;
    color: white;
    padding: 2rem;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
header h1 {
    margin: 0;
    font-size: 2.5rem;
}
header p {
    margin: 0.5rem 0 0;
    font-size: 1.2rem;
}
header .credit {
    margin-top: 1rem;
    font-size: 1rem;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.chat-message:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #fff;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
    font-size: 1.1em;
    line-height: 1.5;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    transition-duration: 0.4s;
    cursor: pointer;
    border-radius: 12px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.gif-container {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}
.stTextInput input {
    background-color: #e0e0e0;
    border: 2px solid #4CAF50;
    border-radius: 12px;
    padding: 10px;
    font-size: 1.1rem;
}
</style>
'''

# HTML templates
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.imgur.com/75rsm.gif">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.imgur.com/GB2Joel.gif">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = Ollama(model="mistral")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    start_time = time.time()  # Start the timer
    response = st.session_state.conversation({'question': user_question})
    end_time = time.time()  # End the timer
    
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    # Display a random GIF after bot's response
    gifs = [
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif",
        "https://media.giphy.com/media/l0HlRnAWXxn0MhKLK/giphy.gif",
        "https://media.giphy.com/media/3o7TKSha51ATTx9KzC/giphy.gif",
        "https://media.giphy.com/media/26FPJGjhefSJuaRhu/giphy.gif"
    ]
    st.markdown(f'<div class="gif-container"><img src="{random.choice(gifs)}" width="200"></div>', unsafe_allow_html=True)

    # Display the response time
    response_time = end_time - start_time
    st.write(f"Response time: {response_time:.2f} seconds")

def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":llama:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    st.markdown('''
    <header>
        <h1>📄 Chat with your data</h1>
        <p>Upload your PDF documents and ask any questions about their content.</p>
        <p class="credit">Made with ❤️ by Aman</p>
    </header>
    ''', unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat")
        st.markdown('<div class="bold-label">Ask a question about your documents:</div>', unsafe_allow_html=True)
        user_question = st.text_input("", key="user_input", placeholder="Type your question here...")
        if user_question:
            handle_userinput(user_question)

    with col2:
        st.header("📁 Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("🔮 Processing your documents..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(_vectorstore=vectorstore)
            
            st.success("✅ Processing complete!")
            st.balloons()

    st.markdown("---")

if __name__ == '__main__':
    main()

