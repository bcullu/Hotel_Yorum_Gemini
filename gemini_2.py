import streamlit as st
from htmlTemplates import css, bot_template, user_template, sanitize_html
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from bs4 import BeautifulSoup

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

# Configure Google AI
genai.configure(api_key=google_api_key)

# --- Load Data (with Error Handling) ---
try:
    df = pd.read_pickle("/Users/batuhancullu/Documents/otel_yorum_scp/Hotel_RAG/PPAD_24/veri.pkl")
except FileNotFoundError:
    st.error("Error: Could not find veri.pkl. Please make sure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Utility Functions ---

def get_data(otel):
    filtered_reviews = df[df['otel_adi'] == str(otel)]
    if filtered_reviews.empty:
        return ""
    raw_text = '\n\n'.join(filtered_reviews['body'])
    return raw_text

def get_text_chunks(raw_text):
    if not raw_text:
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def initialize_vectorstore(chunks, force_refresh=False):
    if 'vectorstore' not in st.session_state or force_refresh:
        with st.spinner("Creating vector store..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                st.session_state['vectorstore'] = vectorstore
                st.success("Vector store created successfully!") 
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
    else:
        st.info("Using cached vectorstore")
    return st.session_state.get('vectorstore')


def create_conversation_chain(vectorstore):
    if vectorstore is None:
        st.error("Vector store is not initialized, cannot create conversation chain.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, convert_system_message_to_human=True)

        # --- IMPROVED PROMPT ENGINEERING ---
        template = """
        Sen seçilen otel hakkında yardımcı olan bir asistansın: {hotel_name}
        
        Sadece verilen bağlam bilgilerine dayanarak soruları yanıtla. 
        Eğer cevabı bilmiyorsan, sadece bilmediğini söyle. Bir cevap uydurmaya çalışma.
        Cevaplarını detaylı ve konuya odaklı tut.
        Otelin genel değerlendirmesi istenirse madde işaretli detaylı cevaplar ver.

        Bağlam:
        {context}

        Soru:
        {input}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # --- RETRIEVER with parameters ---
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 100}  # Retrieve more documents for better context
        )

        # --- DEBUGGABLE CHAIN ---
        def get_context_and_run_chain(input_data):
            # Explicitly get context first
            query = input_data["input"]
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Return the exact structure expected by the prompt
            return {
                "input": query,
                "context": context_text,
                "hotel_name": input_data["hotel_name"],
                "history": input_data["history"]
            }

        # Then simplify your chain
        chain = (
            RunnablePassthrough.assign(
                transformed_input=get_context_and_run_chain
            )
            | {
                "input": lambda x: x["transformed_input"]["input"],
                "context": lambda x: x["transformed_input"]["context"],
                "hotel_name": lambda x: x["transformed_input"]["hotel_name"],
                "history": lambda x: x["transformed_input"]["history"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.success("Conversation chain created successfully!")
        return chain

    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


# --- CONVERSATION MEMORY ---
def load_memory() -> List[dict]:
    """Loads the conversation history from Streamlit session state."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    return st.session_state['chat_history']


def save_memory(input_str: str, output_str: str):
    """Appends the current input and output to the conversation history."""
    st.session_state['chat_history'].append({"input": input_str, "output": output_str})


def display_chat_history():
    """Display the chat history using Streamlit's chat components"""
    # Create a container for the chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["input"])
            with st.chat_message("assistant"):
                st.write(message["output"])


def handle_userinput(user_question, chain):
    if chain:
        try:
            with st.spinner("Thinking..."):
                # Load conversation history
                history = load_memory()
                # Convert history to LangChain message format
                lc_messages = []
                for item in history:
                    lc_messages.append(HumanMessage(content=item["input"]))
                    lc_messages.append(AIMessage(content=item["output"]))

                # Get current hotel
                current_hotel = st.session_state.get('current_otel', '')
                
                # Log for debugging 
                st.session_state['debug_log'] = f"Processing question: {user_question} for hotel: {current_hotel}"
                
                # Invoke the chain with input, history, and hotel_name
                response = chain.invoke(
                    {
                        "input": user_question, 
                        "history": lc_messages,
                        "hotel_name": current_hotel
                    }
                )

                # Save the current interaction to memory
                save_memory(user_question, response)
                
                return response

        except Exception as e:
            st.error(f"Error during conversation: {e}")
            st.write(e)  # Print full exception for debugging
            st.session_state['error_log'] = str(e)
            return f"Error: {str(e)}"
    else:
        st.error("Conversation chain is not initialized. Please select a hotel first.")
        return "Please select a hotel first to start the conversation."


# --- Main App Logic ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Otel Yorumları Asistanı", page_icon="🏨")
    st.write(css, unsafe_allow_html=True)
    if 'current_otel' not in st.session_state:
        st.session_state['current_otel'] = None
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'current_otel' not in st.session_state:
        st.session_state['current_otel'] = None
    if 'debug_log' not in st.session_state:
        st.session_state['debug_log'] = ""
    if 'error_log' not in st.session_state:
        st.session_state['error_log'] = ""


    st.header("Otel Yorumları Asistanı 🏨")

    # Sidebar for hotel selection
    with st.sidebar:
        valid_otel = df['otel_adi'].unique().tolist()
        otel = st.selectbox(
            "Otel Seçin:",
            options=valid_otel,
            index=None,
            placeholder="Bir otel seçin"
        )

        if otel:
            if st.session_state['current_otel'] != otel or st.session_state['vectorstore'] is None:
                with st.spinner("Otel verileri yükleniyor..."):
                    raw_text = get_data(otel)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = initialize_vectorstore(text_chunks, force_refresh=True)
                        if vectorstore:
                            st.session_state['current_otel'] = otel
                            st.session_state['conversation'] = create_conversation_chain(vectorstore)
                            # Clear chat history when hotel changes
                            st.session_state['chat_history'] = []
                            st.success(f"'{otel}' oteli için veriler hazır!")
                        else:
                            st.session_state['current_otel'] = None
                            st.session_state['conversation'] = None
                            st.error("Vektör deposu oluşturulamadı.")

        # Add a button to clear chat history
        if st.button("Sohbet Geçmişini Temizle"):
            st.session_state['chat_history'] = []
            st.experimental_rerun()
            
        # Show current hotel selection
        if st.session_state['current_otel']:
            st.info(f"Seçili otel: {st.session_state['current_otel']}")
            
        # Debug information (toggle with a checkbox)
        if st.checkbox("Debug bilgilerini göster"):
            st.text("Debug Log:")
            st.code(st.session_state['debug_log'])
            if st.session_state['error_log']:
                st.text("Error Log:")
                st.code(st.session_state['error_log'])

    # Main chat area
    # Display chat history
    display_chat_history()
    
    # Chat input using Streamlit's chat_input
    if user_question := st.chat_input("Otel hakkında bir soru sorun:", key="user_question"):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_question)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if st.session_state['conversation'] is None:
                response = "Lütfen önce bir otel seçin."
                st.write(response)
                # Don't save this to chat history as it's just a prompt
            else:
                response = handle_userinput(user_question, st.session_state['conversation'])
                st.write(response)


if __name__ == '__main__':
    main()