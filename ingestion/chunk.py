from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size / chunk_overlap are measured in TOKENS here because we build the
# splitter from a tiktoken encoder. cl100k_base matches the OpenAI embedding
# and GPT-4 tokenizers.
_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=80,
)


def chunk_text(text: str) -> list[str]:
    """Split a section's text into overlapping, token-sized chunks."""
    return _splitter.split_text(text)