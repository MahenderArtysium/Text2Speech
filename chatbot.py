import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import speech_recognition as sr
from gtts import gTTS

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model
tokenizer, model = load_data()

def recordvoice():
    while True:
         r = sr.Recognizer()
         mic = sr.Microphone()
         print('start')
         with mic as source:
            audio = r.listen(source)
         print('end')
         text1 = r.recognize_google(audio)
         return text1


st.write("Welcome to the Avatar Chatbot. I am in development stage, please be patient....")
if st.button('Click on the Button to Speak'):
   text = recordvoice()
   st.write("Input From Audio: "+ str(text))
   input = text
else:
   input = st.text_input('User:')
   print(input, type(input))

if 'count' not in st.session_state or st.session_state.count == 6:
   st.session_state.count = 0 
   st.session_state.chat_history_ids = None
   st.session_state.old_response = ''
else:
   st.session_state.count += 1

new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors="pt")
bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if st.session_state.old_response == response:
   bot_input_ids = new_user_input_ids
 
   st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
   response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


st.write(f"Chatbot: {response}")
pyttsx3.speak(response)
st.session_state.old_response = response