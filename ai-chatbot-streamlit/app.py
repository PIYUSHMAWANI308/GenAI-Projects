import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OPENAI"

## prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpful assistant .  Respond clearly and concisely to user queries.") , 
        ("user" , "Question: {question}")
    ]
)

def generate_response(question , api_key , engine , temperature , max_token):
    openai.api_key = api_key

    llm = ChatOpenAI(model=engine ,
                     temperature=temperature ,
                     max_tokens = max_token
                     )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question" : question})
    return answer


## Title of the app
st.title("Enhanced Q&A ChatBot with OpenAI")

## Sidebar for setting
st.sidebar.title('Setting')
api_key = st.sidebar.text_input('Enter your Open AI API Key :' , type="password")

## Select Open AI Model
engine = st.sidebar.selectbox("Select your OpenAI Model" , ["gpt-4o" , "gpt-4-turbo" , "gpt-4"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and any type of question")
user_input = st.text_input("You:")

# Chat Memory Initialization (VERY IMPORTANT)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if user_input and api_key:
    try:
        # Store user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Generate response
        response = generate_response(
            user_input, api_key, engine, temperature, max_tokens
        )

        # Store bot response
        st.session_state["messages"].append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")

elif user_input:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.write(f"🧑 You: {msg['content']}")
    else:
        st.write(f"🤖 Bot: {msg['content']}")
