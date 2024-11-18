import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import time

def main():
    st.title("Defined Benefit Pension Bot")

    # Set up API keys (use environment variables)
    openai_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]


    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index('carmen')
    client = OpenAI(api_key=openai_key)

    def get_embedding(text, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        # Adjusted rate limiting: sleep for 20 seconds to comply with 3 requests/minute
        time.sleep(0.0001)
        response = client.embeddings.create(input=[text], model=model)
        #print("Response object type:", type(response))  # Debug: Check the type of response
        #print("Response content:", response)  # Debug: Print the response to understand its structure

        return client.embeddings.create(input=[text], model=model).data[0].embedding


    # Create four tabs tab3, tab4 
    tab1, tab2 = st.tabs([
        "CONTROL: GPT-4o",
        "TEST: GPT-4o + DFBRAG",
        #"SMART CONTROL: o1-preview",
        #"SMART TEST: o1-preview + DFBRAG"
    ])

    with tab1:
        st.header("CONTROL: GPT-4o")
        control_chat(client, model_name="gpt-4o")

    with tab2:
        st.header("TEST: GPT-4o + DFBRAG")
        test_chat(client, pinecone_index, model_name="gpt-4o")

    #with tab3:
        #st.header("SMART CONTROL: o1-preview")
        #control_chat(model_name="o1-preview")

   # with tab4:
        #st.header("SMART TEST: o1-preview + DFBRAG")
        #test_chat(client, pinecone_index, model_name="o1-preview")

def control_chat(client, model_name):
     # Use a unique key for each session state
    history_key = f'control_history_control_{model_name}'
    input_key = f'control_input_control_{model_name}'
    # Use a unique key for each session state
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    user_input = st.text_input("You:", key=input_key)
    if user_input:
        # Append user input to history
        st.session_state[history_key].append(("You", user_input))


        # Get response from OpenAI using augmented input
        completion = client.chat.completions.create(
          model=model_name,
          messages=[
            {"role": "system", "content": "You are an expert assistant helping actuaries create Defined Benefit Pension Plans."},
            {"role": "user", "content": user_input}
          ]
        )
        assistant_reply = completion.choices[0].message.content.strip()

        # Append assistant response to history
        st.session_state[history_key].append(("Assistant", assistant_reply))

    # Display chat history
    for speaker, message in st.session_state[history_key]:
        st.write(f"**{speaker}:** {message}")

def test_chat(client, pinecone_index, model_name):

    def get_embedding(text, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        # Adjusted rate limiting: sleep for 20 seconds to comply with 3 requests/minute
        time.sleep(0.0001)
        response = client.embeddings.create(input=[text], model=model)
        #print("Response object type:", type(response))  # Debug: Check the type of response
        #print("Response content:", response)  # Debug: Print the response to understand its structure

        return client.embeddings.create(input=[text], model=model).data[0].embedding
    
    # Use a unique key for each session state
    history_key = f'control_history_test_{model_name}'
    input_key = f'control_input_test_{model_name}'
    # Use a unique key for each session state
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    user_input = st.text_input("You:", key=input_key)
    if user_input:
        # Append user input to history
        st.session_state[history_key].append(("You", user_input))

        # Generate embedding for the user input
        embedding = get_embedding(user_input)

        # Query Pinecone index for similar documents
        query_results = pinecone_index.query(
            vector=embedding,
            top_k=5,
            #include_metadata=True
        )

        # Retrieve the most relevant document
        if query_results and 'matches' in query_results and query_results['matches']:
            top_match = query_results['matches'][0]
            reference_text = top_match['metadata'].get('text', '')
        else:
            reference_text = ''

        # Combine user input with reference text
        augmented_input = f"{user_input} /// here is a reference text related to this query: {reference_text}"

        # Get response from OpenAI using augmented input
        completion = client.chat.completions.create(
          model=model_name,
          messages=[
            {"role": "system", "content": "You are an expert assistant helping actuaries create Defined Benefit Pension Plans."},
            {"role": "user", "content": augmented_input}
          ]
        )
        assistant_reply = completion.choices[0].message.content.strip()

        # Append assistant response to history
        st.session_state[history_key].append(("Assistant", assistant_reply))

    # Display chat history
    for speaker, message in st.session_state[history_key]:
        st.write(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
