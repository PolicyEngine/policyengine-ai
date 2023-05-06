from .helpers import ask_gpt_stream, KnowledgeBase

knowledge = KnowledgeBase()


def add_to_knowledge(text: str):
    knowledge.add(text)


def get_relevant_knowledge(question: str) -> str:
    """Return the most relevant piece of knowledge to a question.

    Args:
        question (str): The question to answer.

    Returns:
        str: The most relevant piece of knowledge.
    """
    relevant_info = "\n".join(list(knowledge.search(question, top_n=3)))
    return ask_gpt_stream(
        prompt=f"""
        Relevant context: {relevant_info}
        The user has a question: {question}
        Answer the question.
        """,
        model="gpt-3.5-turbo",
    )
