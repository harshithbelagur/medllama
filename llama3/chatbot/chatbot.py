import streamlit as st
import time
import requests

url = 'http://inference-service:5000/chat'

system_prompt = 'You are a helpful assistant helping individuals with their medical queries'
user_message = 'I have been having alot of catching ,pain and discomfort under my right rib.  If I twist to either side especially my right it feels like my rib actually catches on something and at times I have to stop try to catch my breath and wait for it to subside.  There are times if I am laughing too hard that it will do the same thing but normally its more so if I have twisted or moved  a certain way'

original_prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

    { user_message }<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

prompt_new = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{ system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>'''

total_string = prompt_new

# Streamed response emulator
def response_generator():
    global total_string
    for message in st.session_state.messages:
        if message["role"] == "user":
            total_string += f'''\n{message['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|> \n'''
        elif message["role"] == "assistant":
            total_string += f'''\n{message['content']}<|eot_id|><|start_header_id|>user<|end_header_id|> \n'''
    
    data = {'message': total_string}
    output = requests.post(url, json=data)
    output = output.json()
    # response = 'Hola amigo tester in Espanol'
   
    for word in output['response'].split():
        yield word + " "
        time.sleep(0.05)

st.title("MedLlama3")

    # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Accept user input
if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

        # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

print(st.session_state)