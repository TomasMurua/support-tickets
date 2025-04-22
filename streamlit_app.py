import datetime
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI

# Load environment variables
load_dotenv()

# Suggestion questions
SUGGESTION_QUESTIONS = [
    "How long does shipping take?",
    "How do I update my shipping address?",
    "How do I access or update my account information?"
]

# Elasticsearch and OpenAI client configuration
@st.cache_resource
def get_elasticsearch_client():
    if "ES_API_KEY" not in os.environ:
        st.error("ES_API_KEY environment variable not found. AI assistant will not be available.")
        return None
    
    return Elasticsearch(
        os.environ["ES_ENDPOINT"],
        api_key=os.environ["ES_API_KEY"]
    )

@st.cache_resource
def get_openai_client():
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OPENAI_API_KEY environment variable not found. AI assistant will not be available.")
        return None
    
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

es_client = get_elasticsearch_client()
openai_client = get_openai_client()

# Source field definitions by index
index_source_fields = {
    ".kibana-observability-ai-assistant-kb-000001": [
        "conversation.title",
        "doc_id",
        "semantic_text",
        "text",
        "title"
    ]
}

# Elasticsearch and OpenAI integration functions
def get_elasticsearch_results(query):
    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "semantic": {
                        "field": "semantic_text",
                        "query": query
                    }
                }
            }
        },
        "highlight": {
            "fields": {
                "semantic_text": {
                    "type": "semantic",
                    "number_of_fragments": 2,
                    "order": "score"
                }
            }
        },
        "size": 10
    }
    result = es_client.search(index=".kibana-observability-ai-assistant-kb-000001", body=es_query)
    return result["hits"]["hits"]

def create_openai_prompt(results):
    context = ""
    for hit in results:
        # For semantic_text matches, extract text from the highlighted field
        if "highlight" in hit:
            highlighted_texts = []
            for values in hit["highlight"].values():
                highlighted_texts.extend(values)
            context += "\n --- \n".join(highlighted_texts)
        else:
            context_fields = index_source_fields.get(hit["_index"])
            for source_field in context_fields:
                hit_context = hit["_source"][source_field]
                if hit_context:
                    context += f"{source_field}: {hit_context}\n"
    prompt = f"""
  Instructions:
  
  - You are the UrbanStyle Support Assistant. Your goal is to deliver fast, precise and professional responses to customers using only the information available in your knowledge base. 
Always follow this exact structure when responding:
1. Title: A clear, informative headline summarizing the topic  
2. Summary: One or two concise sentences directly answering the question  
3. Reference: Direct URL to the relevant article in the knowledge base  
4. Additional Info: Optional short notes with useful tips or next steps  
5. Contact Support Note: Only if no relevant information is found or further human action is required  
Guidelines:
• Do not sound like a chatbot or engage in any casual conversation.  
• Never include greetings ("Hi", "Hello"), sign‑offs or apologies.  
• Do not use Markdown, asterisks, code blocks or JSON formatting.  
• Never fabricate information or links. Only return what exists in the knowledge base.  
• If you cannot find an article matching the request, reply exactly:
Title: Information Not Found  
Summary: We're unable to find information related to your request in the current knowledge base.  
Contact Support Note: Please contact our team directly at support@urbanstyle.com
  - Answer questions truthfully and factually using only the context presented.
  - If you don't know the answer, just say that you don't know, don't make up an answer.
  - You must always cite the document where the answer was extracted using inline academic citation style [], using the position.
  - Use markdown format for code examples.
  - You are correct, factual, precise, and reliable.
  
  Context:
  {context}
  
  """
    return prompt

def generate_openai_completion(user_prompt, question):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content

st.header("E‑commerce Support Assistant")
st.write("Describe your issue below. The assistant will try to answer using the knowledge base and show related articles.")

# Add suggestion pills
st.write("Common questions:")
cols = st.columns(3)
for i, question in enumerate(SUGGESTION_QUESTIONS):
    with cols[i % 3]:
        if st.button(
            question,
            key=f"suggestion_{i}",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.user_question = question
            st.session_state.auto_submit = True

with st.form("case_deflection_form"):
    issue = st.text_area("Describe the issue", value=st.session_state.get("user_question", ""))
    submitted = st.form_submit_button("Submit") or st.session_state.get("auto_submit", False)
    if st.session_state.get("auto_submit", False):
        st.session_state.auto_submit = False

if submitted and issue and es_client and openai_client:
    with st.spinner("Searching knowledge base..."):
        elasticsearch_results = get_elasticsearch_results(issue)
        if elasticsearch_results:
            context_prompt = create_openai_prompt(elasticsearch_results)
            with st.spinner("Generating response..."):
                openai_completion = generate_openai_completion(context_prompt, issue)
                import re
                def format_reference(text):
                    ref_match = re.search(r'Reference:\s*\[(.*?)\]\((.*?)\)', text)
                    if ref_match:
                        title, url = ref_match.groups()
                        link_html = f'<a href="{url}" target="_blank" style="color:#4ba3fa;">{title}</a>'
                        text = re.sub(r'Reference:\s*\[.*?\]\(.*?\)', f'Reference: {link_html}', text)
                    return text
                formatted_response = format_reference(openai_completion)
                st.markdown(
                    '<div style="background-color:#262730;padding:1.5rem 1.2rem;border-radius:12px;border:1px solid #444;margin-bottom:1.5rem;">'
                    '<h4 style="margin-top:0;margin-bottom:1rem;color:#fff;">Assistant Response</h4>'
                    f'<div style="color:#fff;white-space:pre-line;font-size:1.1rem;">{formatted_response}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            # Display related articles
            st.header("Related Articles")
            for hit in elasticsearch_results:
                with st.expander(f"Article: {hit['_source'].get('title', 'Untitled')}"):
                    st.write(hit["_source"].get("text", "No content available"))
        else:
            st.info("No relevant articles found in the knowledge base.")
elif submitted and not (es_client and openai_client):
    st.error("AI assistant is not available. Please check your environment variables for ES_API_KEY and OPENAI_API_KEY.")
