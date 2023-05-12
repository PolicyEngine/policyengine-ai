from capabilities.helpers.knowledge_bases import ChromaKnowledgeBase 
from capabilities.helpers.llm import ask_gpt_stream
from capabilities.helpers.text_splitters import section_header_split 

knowledge = ChromaKnowledgeBase()

def add_to_knowledge(text: str):
    knowledge.add(text, split_fn=section_header_split)


def get_relevant_knowledge(question: str) -> str:
    """Return the most relevant piece of knowledge to a question.

    Args:
        question (str): The question to answer.

    Returns:
        str: The most relevant piece of knowledge.
    """
    relevant_info = "\n".join(list(knowledge.search(question, top_n=3)))
    prompt=f"""
The user has a question: 

{question}

Relevant laws, regulations, or general background information: 

{relevant_info}

You must answer the question using the information in relevant laws, regulations, or background information above. Always cite where you got the information from.
"""

    return ask_gpt_stream(
        prompt=prompt,
        model="gpt-3.5-turbo",
    )
