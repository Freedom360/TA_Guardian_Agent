# TA_Guardian

## Project Overview
https://github.com/Freedom360/CS410_Guardian/blob/c522db2c1e2598b4de6a85e11ff88685c020ce3f/CS410%20Guardian%20Documentation.pdf

## Presentation/Demo: 
https://mediaspace.illinois.edu/media/t/1_x4k9b0rm

## How to Run: 
To use streamlit app, first make sure to create a separate a python or conda environmnet file and pip install the requirements file. To actually launch the streamlit app in localhost, cd to 'streamlit_app' in terminal and type 'streamlit run cache_chatbot_new.py' (the name of file with app. Make sure streamlit_app folder contains content_vectors.csv and lessons_vectors.csv (the embeddings tables).

## System Architecture

The TA Guardian Agent utilizes a lightweight, locally-stored Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based strictly on course materials.

```mermaid
flowchart TB
    %% Styling Definitions
    classDef offline fill:#f9f9fa,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef online fill:#eef5fc,stroke:#0056b3,stroke-width:2px;
    classDef data fill:#d4edda,stroke:#28a745,stroke-width:1px;
    classDef processing fill:#fff3cd,stroke:#ffc107,stroke-width:1px;
    classDef external fill:#f8d7da,stroke:#dc3545,stroke-width:1px;
    
    %% User Node
    User((Student / User))
    
    subgraph Offline_Pipeline [Offline Data Pipeline: Knowledge Base Generation]
        direction TB
        RawData[(Course Materials \n Folders: Week 1-12)]:::data
        ETL[generate_embeddings_table.ipynb \n Data Extraction & Text Chunking]:::processing
        EmbedAPI_1{{Embedding Model API \n e.g., OpenAI}}:::external
        
        CSV1[(content_vectors.csv)]:::data
        CSV2[(lessons_vectors.csv)]:::data
        
        RawData --> ETL
        ETL -->|Send Text Chunks| EmbedAPI_1
        EmbedAPI_1 -->|Return Vectors| CSV1 & CSV2
    end
    class Offline_Pipeline offline

    subgraph Online_Pipeline [Online Application: RAG Inference]
        direction TB
        UI[Streamlit Frontend \n streamlit_app]:::processing
        AppLogic[Application Logic \n cache_chatbot_new.py]:::processing
        EmbedAPI_2{{Embedding Model API}}:::external
        SimSearch{Vector Similarity Search \n e.g., Cosine/Euclidean}:::processing
        LLM{{Large Language Model API \n e.g., GPT-3.5/4}}:::external
        
        UI <-->|User Query / Response| AppLogic
        AppLogic -->|Embed Query| EmbedAPI_2
        EmbedAPI_2 -->|Return Query Vector| SimSearch
        SimSearch -->|Retrieve Top-K Context| AppLogic
        AppLogic -->|Construct Prompt \n Context + Query| LLM
        LLM -->|Generate Grounded Answer| AppLogic
    end
    class Online_Pipeline online

    %% Cross-Pipeline Connections
    User -->|Asks Question| UI
    UI -->|Displays Answer| User
    
    CSV1 -.->|Loaded into Memory \n on App Start| SimSearch
    CSV2 -.->|Loaded into Memory \n on App Start| SimSearch
