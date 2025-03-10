import streamlit as st
from htmlTemplates import css, bot_template, user_template, sanitize_html  # Import sanitize_html
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferMemory # No longer directly used
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import for prompt
from langchain_core.runnables import RunnablePassthrough, RunnableConfig # For the chain
from langchain_core.output_parsers import StrOutputParser  # Parse LLM output
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
from typing import List  # Import List for type hinting
from langchain_core.messages import AIMessage, HumanMessage  # Import message types
from bs4 import BeautifulSoup #For HTML Parsing

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

def get_data(ship_name):
    filtered_reviews = df[df['otel_adi'] == str(ship_name)]
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
                time.sleep(2)  # Simulate loading (remove)
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
    else:
        st.info("Using cached vectorstore")
    return st.session_state.get('vectorstore')


def create_conversation_chain(vectorstore):
    if vectorstore is None:
        return None

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, convert_system_message_to_human=True)

        # --- PROMPT ENGINEERING ---
        template = """
        You are a helpful assistant for hotel customer support. Always be nice. Answer questions based only on the provided context.
        If you don't know the answer, just say that you don't know.  Don't try to make up an answer.
        Keep your answers detailed and to the point. If general evaluation of the hotel asked give detailed bulletpoint answers."

        Context:
        {context}

        Question:
        {input}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # --- RETRIEVER ---
        retriever = vectorstore.as_retriever()

        # --- CHAIN ---
        chain = (
            RunnablePassthrough.assign(context=retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)))
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain


    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


# --- CONVERSATION MEMORY (Custom Implementation) ---
def load_memory() -> List[dict]:
    """Loads the conversation history from Streamlit session state."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    return st.session_state['chat_history']


def save_memory(input_str: str, output_str: str):
    """Appends the current input and output to the conversation history."""
    st.session_state['chat_history'].append({"input": input_str, "output": output_str})



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

                # Invoke the chain with input and history
                response = chain.invoke(
                    {"input": user_question, "history": lc_messages},
                    config={"configurable": {"max_tokens": 256}}  # Example of using configurable
                )

                # Save the current interaction to memory
                save_memory(user_question, response)


            # Display the updated conversation history
            for item in st.session_state['chat_history']:
                st.write(sanitize_html(user_template.replace("{{MSG}}", item["input"])), unsafe_allow_html=True)
                st.write(sanitize_html(bot_template.replace("{{MSG}}", item["output"])), unsafe_allow_html=True)


        except Exception as e:
            st.error(f"Error during conversation: {e}")
            st.write(e)  # Print full exception for debugging

    else:
        st.error("Conversation chain is not initialized.")

# --- Main App Logic ---
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Customers", page_icon=":people:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state: # Used for storing the chain itself.
        st.session_state['conversation'] = None
    if 'chat_history' not in st.session_state:  # Used for storing the chat messages.
        st.session_state['chat_history'] = []
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'current_ship' not in st.session_state:
        st.session_state['current_ship'] = None


    st.header("Chat with Customers :people:")

    valid_ship_names = df['otel_adi'].unique().tolist()
    ship_name = st.selectbox(
        "Select a hotel:",
        options=valid_ship_names,
        index=None,
        placeholder="Choose an option"
    )

    if ship_name:
        if st.session_state['current_ship'] != ship_name or st.session_state['vectorstore'] is None:
            raw_text = get_data(ship_name)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vectorstore = initialize_vectorstore(text_chunks, force_refresh=True)
                if vectorstore:
                    st.session_state['current_ship'] = ship_name
                    st.session_state['conversation'] = create_conversation_chain(vectorstore) # Get the chain
            else:
                st.warning(f"No reviews found for {ship_name}.")
                st.session_state['current_ship'] = None
                st.session_state['conversation'] = None

        user_question = st.text_input("Ask a question to customers:", key="user_question")

        if user_question:
            if st.session_state['conversation'] is None:
                st.error("Please select a hotel with reviews to start the conversation.")
            else:
                handle_userinput(user_question, st.session_state['conversation']) #Pass the chain
if __name__ == '__main__':
    main()