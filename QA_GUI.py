
#Importing required packages
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import streamlit as st 
import torch



#title of the webpage
st.title("Telugu Question Answer System")


#text fields
context=st.text_input("పేరాను నమోదు చేయండి")
question=st.text_input("మీ ప్రశ్నను నమోదు చేయండి")


#caching the model for multiple requests
@st.cache(allow_output_mutation=True)
def load_model():

	tokenizer = AutoTokenizer.from_pretrained('teluguQAtrained',local_files_only=True)
	model = AutoModelForQuestionAnswering.from_pretrained('teluguQAtrained',local_files_only=True)
	
	return tokenizer,model


#loading the model and tokenizer
with st.spinner('Loading model and tokeniser into memory...'):
	tokenizer,model=load_model()


#if both textfields are given this executes
if(question and context):
	st.write("Response:")
	with st.spinner("predicting answer..."):

		#encoding context and question using our custom tokeniser
		encoding = tokenizer.encode_plus(question, context, return_tensors="pt")

		#fetching input_ids and attentions mask from ecodings
		input_ids = encoding["input_ids"]
		attention_mask = encoding["attention_mask"]

		#input above inputs to our trained model and predict start score and endscore(in ids form)
		start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

		#convert ids to token form
		all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

		#decode tokens 
		answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
		answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
		
		#displaying answer
		st.write("Answer:  "+answer)

	st.write("")


