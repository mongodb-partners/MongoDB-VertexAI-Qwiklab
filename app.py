import streamlit as st
from langchain.llms import VertexAI
from pymongo import MongoClient
from PyPDF2 import PdfReader
import certifi
from langchain.chat_models import ChatVertexAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from functools import lru_cache

st.title("🕵️‍Chatter")
# add your URI for MongoClient
client = MongoClient("replace with your URI here", tlsCAFile=certifi.where())
db = client['vertexaiApp']


def intro():
    import streamlit as st

    st.write("# Welcome 😊! ")
    st.sidebar.success("Select from the above options")

    st.markdown('''
            ##### This app is a Google VertexAI PAML-powered chatbot built using:
             [Streamlit](https://streamlit.io/) | [Google VertexAI](https://cloud.google.com/vertex-ai/docs) | [LangChain](https://python.langchain.com/) | [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
            ''')
    st.write('Created by [Ashwin Gangadhar](https://www.linkedin.com/in/ashwin-gangadhar-00b17046) and [Venkatesh Shanbhag](https://www.linkedin.com/in/venkatesh-shanbhag/)')


def get_embeddings_transformer():
    embeddings = VertexAIEmbeddings()
    return embeddings


@lru_cache(maxsize=1)
def get_vector_store():
    col = db['chat-vec']
    vs = MongoDBAtlasVectorSearch(collection=col, embedding=VertexAIEmbeddings(), index_name="default",
                                  embedding_key="vec", text_key="line")
    return vs


@lru_cache(maxsize=1)
def get_conversation_chain():
    llm = ChatVertexAI()
    retriever = get_vector_store().as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template="""
       Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
       {context}
       ##Question:{question} \n\
       ## Chat History: {chat_history}
       ##AI Assistant Response:\n""", input_variables=["context", "chat_history", "question"])}
    )

    return conversation_chain


def handle_userinput(user_question):
    conv = get_conversation_chain()
    response = conv({'question': user_question, "chat_history": st.session_state.chat_history})
    chat_history = [{"user": response["question"], "assistant": response["answer"]}]
    st.session_state.chat_history += chat_history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])


def search_docs():
    import streamlit as st

    with st.form('search_form'):
        text = st.text_area('Enter text:', '')
        submitted = st.form_submit_button('🔍 SEARCH')
        if submitted:
            llm = VertexAI()
            print(llm(text))
            st.info(llm(text))


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def upload_docs():
    import streamlit as st

    with st.form('load_form'):
        pdf = st.file_uploader(
            "Upload your PDFs", accept_multiple_files=False)
        submitted = st.form_submit_button('Submit')
        if submitted:

            with st.spinner("Processing"):
                vs = st.session_state.vectorstore
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = ""
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()

                    if raw_text:
                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        if len(text_chunks) > 500:
                            split = 100
                        else:
                            split = 10
                        for i in range(0, len(text_chunks), split):
                            batch_chunks = text_chunks[i:(i + split - 1)]
                            vs.add_texts(batch_chunks)
            st.write('Document added successfully')


st.session_state.vectorstore = get_vector_store()
page_names_to_funcs = {
    "Welcome": intro,
    "Search data": search_docs,
    "Upload data": upload_docs,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
