from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_core.pydantic_v1 import SecretStr
except ImportError:
    from pydantic import SecretStr

from langchain_openai import ChatOpenAI

DEFAULT_SYSTEM_PROMPT = (
    "You are ResearchLite, a helpful assistant powered by a lightweight RAG system.\n"
    "Use the following pieces of retrieved context to answer the user's question.\n"
    "If the answer is not in the context, say that you don't know. Keep the answer concise.\n\n"
    "Context:\n{context}"
)


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        final_api_key: SecretStr | None = None
        if api_key is not None:
            final_api_key = SecretStr(api_key)

        self._llm: BaseChatModel = ChatOpenAI(
            model=model_name,
            api_key=final_api_key,
            base_url=base_url,
            temperature=temperature,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def generate_answer(self, query: str, context_documents: Iterable[Document]) -> str | None:
        """Generate an answer based on the query and retrieved documents."""
        context_text = "\n\n".join(
            f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
            for doc in context_documents
        )

        result = self._chain.invoke({"question": query, "context": context_text})
        if isinstance(result, str):
            return result
        return None
