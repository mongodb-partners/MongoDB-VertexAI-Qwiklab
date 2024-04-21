import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain_community.llms import LlamaCpp, VertexAI
from langchain_google_vertexai import ChatVertexAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import PromptTemplate

import hashlib
from functools import lru_cache

from pymongo import MongoClient
import certifi

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "tmp/st/"
# Update MongoDB URI
client = MongoClient("mongodb+srv://username:password@host-name", tlsCAFile=certifi.where())
db = client["vertexaiApp"]

one_way_hash = lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()

CHAT_VERIFY_COL = "chat-vec-verify"
CHAT_APP_COL = "chat-vec"

PROMPT = PromptTemplate(template="""
       Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't be too descriptive and give point to point answer.
       {context}
       ##Question:{question} \n\
       ## Chat History: {chat_history}
       ##AI Assistant Response:\n""", input_variables=["context", "chat_history", "question"])


def check_doc_in_mdb(md5):
    if len(list(db[CHAT_VERIFY_COL].find({"md5": md5}))) > 0:
        return True
    else:
        return False


def insert_doc_verify_mdb(md5):
    db[CHAT_VERIFY_COL].insert_one({"md5": md5})


def get_pdf_data(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    md5 = one_way_hash(text)
    print(">>>>>>>>>>>>>>")
    print(md5)
    if check_doc_in_mdb(md5):
        return None, None
    else:
        return text, md5


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings_transformer():
    embeddings = VertexAIEmbeddings(model_name = "textembedding-gecko@001")
    return embeddings


@lru_cache(maxsize=1)
def get_vector_store():
    col = db[CHAT_APP_COL]
    vs = MongoDBAtlasVectorSearch(collection=col, embedding=get_embeddings_transformer(), index_name="vector_index",
                                  embedding_key="vec", text_key="line")
    return vs


@lru_cache(maxsize=1)
def get_conversation_chain():
    llm = ChatVertexAI(model_name="gemini-pro", convert_system_message_to_human=True,max_output_tokens=1000)
    retriever = get_vector_store().as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
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


st.set_page_config(page_title="Chat with multiple PDFs",
                   page_icon=":books:")
st.session_state.vectorstore = get_vector_store()
st.session_state.conv = get_conversation_chain()
tab1, tab2 = st.tabs(["Q & A", "ADD document"])
with tab1:
    st.markdown(
        """<img src="https://lh3.googleusercontent.com/I2_PSO0vMM8kLJxJ-OUIqtSBo3krzhmctqIkFv8Exgchm5X04h_MysTSB-8mELD6J_OIA1N2ExP_=e14-rj-sc0xffffff-h338-w600" class=" css-1lo3ubz" alt="MongoDB logo" style="height:200px;width:340px;align:center"> """,
        unsafe_allow_html=True)
    # st.title("""Assistant for any source powered by Atlas Vector Search and VertexAI""")

    chat_history_clear = st.button("Clear Chat History")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if ("chat_history" not in st.session_state) or chat_history_clear:
        st.session_state.chat_history = []

    st.header("Assistant for any source powered by MongoDB Atlas Vector Search and VertexAI")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

with tab2:
    st.subheader("Your documents")
    pdf = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
    b = st.button("Process")
    if b:
        vs = st.session_state.vectorstore
        with st.spinner("Processing"):
            # get pdf text
            raw_text, md5 = get_pdf_data(pdf)
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
                # insert to md5 once indexed
                insert_doc_verify_mdb(md5)
                st.write('Document added successfully')

    with st.sidebar:
        add_vertical_space(3)
        st.title("Process your PDFs and perform vector search")
        st.markdown('''
        ## About
        This app is a Google VertexAI PALM-powered chatbot built using:
        - [Google VertexAI](https://cloud.google.com/vertex-ai/docs)
        - [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        ''')
        add_vertical_space(5)
        st.write(
            'Made with ❤️  by [Ashwin Gangadhar](linkedin.com/in/ashwin-gangadhar-00b17046) and [Venkatesh Shanbhag](https://www.linkedin.com/in/venkatesh-shanbhag/) v1')
