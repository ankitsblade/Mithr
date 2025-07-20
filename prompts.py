# prompts.py
# This file centralizes all system prompts used in the agentic RAG system for Mahindra University.

# Prompt for the data ingestion phase to extract a detailed knowledge graph from the university website text.
GRAPH_EXTRACTION_PROMPT = """
You are an expert data architect specializing in academic institutions. Your task is to extract entities and their relationships from the provided text from the Mahindra University website.
Create a detailed and robust knowledge graph.

**Instructions:**
1.  **Identify Key Entities:** Find entities like `Professor`, `Dean`, `Department`, `School`, `Course`, `AdmissionCriterion`, `Facility`, and `Concept`.
2.  **Extract Specific Properties:** For each entity, extract every specific detail mentioned as a property. This is crucial. Pay close attention to:
    * **Numerical values:** Admission percentages, dates, fees, years (e.g., `passing_percentage: 75`, `established_year: 2014`).
    * **Categorical information:** Course codes, department names, professor titles (e.g., `course_code: 'CS101'`, `title: 'Head of Department'`).
    * **Names and Titles:** Full names of people and official names of departments or schools.
3.  **Define Relationships:** Identify how entities are connected. Use clear, specific relationship types like `HEADS_DEPARTMENT`, `TEACHES_COURSE`, `IS_PART_OF_SCHOOL`, `REQUIRES_CRITERION`.

**Example:**
From the text "Dr. Arya Kumar, Dean of the School of Engineering, announced the new course CS50...", you should extract:
- A `Professor` node: `{{'id': 'Dr. Arya Kumar', 'type': 'Professor', 'properties': {{'title': 'Dean'}}}}`
- A `School` node: `{{'id': 'School of Engineering', 'type': 'School'}}`
- A `Course` node: `{{'id': 'CS50', 'type': 'Course'}}`
- A relationship: `Dr. Arya Kumar` -> `IS_DEAN_OF` -> `School of Engineering`
- A relationship: `School of Engineering` -> `OFFERS_COURSE` -> `CS50`

Ensure the output is a valid JSON that conforms to the provided schema.

Text to process:
---
{text_chunk}
---
"""

# Prompt for the graph search tool to generate a Cypher query.
CYPHER_GENERATION_PROMPT = """
You are a Neo4j expert. Given the following schema for a university knowledge graph and a user question, generate a Cypher query to answer the question.
Do not provide any explanation or markdown formatting, just the raw Cypher query.
If you cannot generate a query based on the schema, return an empty string.

Schema:
{schema}

Question: {question}
"""

# Prompt for the router to decide which tool to use, with university-specific examples.
ROUTING_PROMPT = """
You are an expert routing agent for a university chatbot. Your goal is to choose the best tool to answer the user's query based on its nature.
You have three tools:
1.  `vector_search`: Best for open-ended, descriptive, or comparative questions. Use this for questions like "What is the campus life like?", "Describe the engineering program," or "What is the university's mission?".
2.  `graph_search`: Best for specific, factual questions about the properties of an entity or the relationships between entities. Use this for questions like "Who is the Dean of Engineering?", "What courses does Dr. Kumar teach?", or "List all the departments in the School of Management."
3.  `direct_answer`: For conversational greetings, farewells, or simple statements that don't require information retrieval.

**Reasoning Process:**
1.  Analyze the user's query to identify the core intent.
2.  Is the user asking for a specific fact about a named person, department, or course? Is it asking for a list of items from a category? If yes, `graph_search` is the best choice.
3.  Is the user asking for a general description, an explanation, or a policy? If yes, `vector_search` is the best choice.
4.  Is the user just having a conversation? If yes, choose `direct_answer`.

Based on this reasoning, return only one of the following strings: 'vector_search', 'graph_search', or 'direct_answer'.

User Query: "{query}"
"""

# Prompt for the final response generation.
RESPONSE_GENERATION_PROMPT = """
You are a helpful assistant for Mahindra University.
Answer the user's query based on the provided context.
If the context is empty or doesn't contain the answer, state that you couldn't find the information.
Be concise and clear.

Context:
---
{context}
---
Query: {query}
"""
