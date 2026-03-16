from nltk.tokenize import sent_tokenize

def production_chunk_text(text, chunk_size=500, overlap=80):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence

        else:
            chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:]

            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
