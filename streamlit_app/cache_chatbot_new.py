# import libraries
import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# function to set up OpenAI and cache
@st.cache_resource
def get_openai_client():
    return ChatOpenAI(
        model_kwargs={'temperature': 0.2}
    )

# function to set up HuggingFace embeddings model and cache
@st.cache_resource
def get_embeddings_model():
    device = "cpu"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {
        "normalize_embeddings": False,
        "batch_size": 512
    }
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

# function to load the embeddings table for lecture content and cache
@st.cache_data
def load_content():
    content = pd.read_csv('content_vectors.csv')
    content['vectors'] = [eval(x) for x in content['vectors']]
    return content

# function to create the FAISS vector store from the embeddings table and cache
@st.cache_resource
def create_vector_store(content, _embeddings_model):
    # function to create the embeddings
    def vector_store_faiss(df, embeddings, _embeddings_model, pc_col=False, metadata_cols=False):
        metadata = df.to_dict(orient='records') if not metadata_cols else df[metadata_cols].to_dict(orient='records')
        texts = df['Concatenated'].tolist() if not pc_col else df[pc_col].tolist()
        text_embedding_pairs = zip(texts, embeddings)
        return FAISS.from_embeddings(text_embedding_pairs, _embeddings_model, metadatas=metadata)
    
    return vector_store_faiss(
        content.drop(columns=['vectors']), 
        content['vectors'].tolist(), 
        _embeddings_model, 
        pc_col='Chunk'
    )

# function to create prompt template sent to llm for context and cache
# includes llm's role, guidlines to follow, chat history, and how to format sources
@st.cache_resource
def get_qa_prompt():
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

You are an expert assistant for UIUC's CS 410: Text Information Systems class.
Your role is to help students understand text mining concepts and course materials.

Additional guidelines:
1. Always ground your answers in the provided context
2. If a concept appears in multiple weeks, mention the progression of the topic
3. Use specific examples from the course materials when possible
4. If a question is unclear, ask for clarification
5. Relate concepts to practical applications when relevant
6. Provide formulas (math equation) when user asks for it. PUT FORMULAS ON A NEW LINE & BOLD THEM!!!

Previous conversation context:
{chat_history}

Current context:
{context}

If answer not in context above, use your background knowledge to answer and provide RELEVANT SOURCES.

Internal Source Example -> Week: 1, Lesson: 1
External Source Example -> External Source: https://www.example.com

Question: {question}

Helpful Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=['context', 'chat_history', 'question']
    )

# function to create the chatbot chain using langchain 
@st.cache_resource
def create_qa_chain(_llm, _vector_store, _memory, _qa_prompt):
    return ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vector_store.as_retriever(search_kwargs={'k': 8}),
        memory=_memory,
        chain_type="stuff",  # Explicitly specify chain type
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": _qa_prompt},
    )

# function to handle chatbot's memory
def init_memory():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return st.session_state.memory

# function to handle message history and state
def init_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

# function to format sources and cache
@st.cache_data(ttl=3600)  # Cache responses for 1 hour
def format_response(response, sources):
    formatted_sources = "\n".join(list(set(sources)))
    return f"{response}\n\nSources:\n{formatted_sources}"

# main function for streamlit
# sets up UI, initializes and loads cached components, and processes user input
def main():
    # set up llm, embeddings, data, and chatbot chain
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(model_kwargs={'temperature': 0})
    embeddings_model = get_embeddings_model()
    content = load_content()
    vector_store = create_vector_store(content, embeddings_model)
    qa_prompt = get_qa_prompt()
    memory = init_memory()
    qa_chain = create_qa_chain(llm, vector_store, memory, qa_prompt)
    messages = init_messages()

    # styling
    st.markdown(
        f"""
        <style>
            .stApp {{ 
                margin-left: auto;
                margin-right: auto;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # main Streamlit UI
    st.title("💬 CS 410 Guardian")
    st.write(
        "RAG mi amor ❤️ ! To the rescue 🚀 !"
    )

    # display chat history
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # handle user input
    if prompt := st.chat_input("Ask your question..."):
        with st.spinner("Thinking..."):
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # get response from chatbot chain and sources
                result = qa_chain({"question": prompt})
                sources = [f"Week: {doc.metadata['Week']}, Lesson: {doc.metadata['Lesson']}" 
                          for doc in result['source_documents']]
                
                formatted_response = format_response(result['answer'], sources)

                with st.chat_message("assistant"):
                    st.markdown(formatted_response)
                
                messages.append({"role": "assistant", "content": formatted_response})
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try asking your question again.")

# run app
if __name__ == "__main__":
    main()