import datetime
import random
import os
from dotenv import load_dotenv

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Load environment variables
load_dotenv()

# Elasticsearch and OpenAI imports
from elasticsearch import Elasticsearch
from openai import OpenAI

# Show app title and description.
st.set_page_config(page_title="Support tickets", page_icon="🎫")
st.title("🎫 Support tickets")
st.write(
    """
    This app shows how you can build an internal tool with Streamlit.
    We're implementing a support ticket workflow and an AI assistant that can
    answer questions based on the knowledge base.
    """
)

# Elasticsearch and OpenAI client configuration
@st.cache_resource
def get_elasticsearch_client():
    if "ES_API_KEY" not in os.environ:
        st.error("ES_API_KEY environment variable not found. AI assistant will not be available.")
        return None
    
    return Elasticsearch(
        "https://general-search-d6fc37.es.us-east-1.aws.elastic.cloud:443",
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

# Create a random Pandas dataframe with existing tickets.
if "df" not in st.session_state:

    # Set seed for reproducibility.
    np.random.seed(42)

    # Make up some fake issue descriptions.
    issue_descriptions = [
        "Network connectivity issues in the office",
        "Software application crashing on startup",
        "Printer not responding to print commands",
        "Email server downtime",
        "Data backup failure",
        "Login authentication problems",
        "Website performance degradation",
        "Security vulnerability identified",
        "Hardware malfunction in the server room",
        "Employee unable to access shared files",
        "Database connection failure",
        "Mobile application not syncing data",
        "VoIP phone system issues",
        "VPN connection problems for remote employees",
        "System updates causing compatibility issues",
        "File server running out of storage space",
        "Intrusion detection system alerts",
        "Inventory management system errors",
        "Customer data not loading in CRM",
        "Collaboration tool not sending notifications",
    ]

    # Generate the dataframe with 100 rows/tickets.
    data = {
        "ID": [f"TICKET-{i}" for i in range(1100, 1000, -1)],
        "Issue": np.random.choice(issue_descriptions, size=100),
        "Status": np.random.choice(["Open", "In Progress", "Closed"], size=100),
        "Priority": np.random.choice(["High", "Medium", "Low"], size=100),
        "Date Submitted": [
            datetime.date(2023, 6, 1) + datetime.timedelta(days=random.randint(0, 182))
            for _ in range(100)
        ],
    }
    df = pd.DataFrame(data)

    # Save the dataframe in session state (a dictionary-like object that persists across
    # page runs). This ensures our data is persisted when the app updates.
    st.session_state.df = df

st.header("E‑commerce Case Deflection Demo")
st.write("Describe your issue below. The assistant will try to answer using the knowledge base. If no answer is found, a support ticket will be created automatically.")

with st.form("case_deflection_form"):
    issue = st.text_area("Describe the issue")
    submitted = st.form_submit_button("Submit")

if submitted and issue and es_client and openai_client:
    with st.spinner("Searching knowledge base..."):
        elasticsearch_results = get_elasticsearch_results(issue)
        if elasticsearch_results:
            context_prompt = create_openai_prompt(elasticsearch_results)
            with st.spinner("Generating response..."):
                openai_completion = generate_openai_completion(context_prompt, issue)
                st.markdown("### Assistant Response")
                st.markdown(openai_completion)
        else:
            # No results: create automatic support ticket
            recent_ticket_number = int(max(st.session_state.df.ID).split("-")[1]) if len(st.session_state.df) > 0 else 1001
            next_ticket_number = recent_ticket_number + 1
            today = datetime.datetime.now().strftime("%m-%d-%Y")
            auto_product = "Auto-generated"  # Demo: product is auto-generated
            df_auto = pd.DataFrame([{
                "ID": f"TICKET-{next_ticket_number}",
                "Order ID": f"ORDER-{next_ticket_number}",
                "Product": auto_product,
                "Issue": issue,
                "Status": "Open",
                "Priority": "Medium",
                "Date Submitted": today
            }])
            st.info("No answer found. A support ticket has been created for your request.")
            st.dataframe(df_auto, use_container_width=True, hide_index=True)
            st.session_state.df = pd.concat([df_auto, st.session_state.df], axis=0)
elif submitted and not (es_client and openai_client):
    st.error("AI assistant is not available. Please check your environment variables for ES_API_KEY and OPENAI_API_KEY.")

# Show section to view and edit existing tickets in a table.
st.header("Existing tickets")
st.write(f"Number of tickets: `{len(st.session_state.df)}`")

st.info(
    "You can edit the tickets by double clicking on a cell. Note how the plots below "
    "update automatically! You can also sort the table by clicking on the column headers.",
    icon="✍️",
)

# Show the tickets dataframe with `st.data_editor`. This lets the user edit the table
# cells. The edited data is returned as a new dataframe.
edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "Status",
            help="Ticket status",
            options=["Open", "In Progress", "Closed"],
            required=True,
        ),
        "Priority": st.column_config.SelectboxColumn(
            "Priority",
            help="Priority",
            options=["High", "Medium", "Low"],
            required=True,
        ),
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=["ID", "Date Submitted"],
)

# Show some metrics and charts about the ticket.
st.header("Statistics")

# Show metrics side by side using `st.columns` and `st.metric`.
col1, col2, col3 = st.columns(3)
num_open_tickets = len(st.session_state.df[st.session_state.df.Status == "Open"])
col1.metric(label="Number of open tickets", value=num_open_tickets, delta=10)
col2.metric(label="First response time (hours)", value=5.2, delta=-1.5)
col3.metric(label="Average resolution time (hours)", value=16, delta=2)

# Show two Altair charts using `st.altair_chart`.
st.write("")
st.write("##### Ticket status per month")
status_plot = (
    alt.Chart(edited_df)
    .mark_bar()
    .encode(
        x="month(Date Submitted):O",
        y="count():Q",
        xOffset="Status:N",
        color="Status:N",
    )
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

st.write("##### Current ticket priorities")
priority_plot = (
    alt.Chart(edited_df)
    .mark_arc()
    .encode(theta="count():Q", color="Priority:N")
    .properties(height=300)
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(priority_plot, use_container_width=True, theme="streamlit")
