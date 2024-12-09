import pandas as pd
import re

def remove_non_text_fields(answers_str: str) -> str:
    # Remove all fields except 'text'
    cleaned = re.sub(r"'(?!text')\w+': [^,}]+(, )?", "", answers_str)
    # Clean up extra commas/spaces
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*\]", "]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def clean_html(raw_html: str) -> str:
    return re.sub(r'<[^>]*>', '', raw_html).strip()

def main():
    df = pd.read_csv("../../data/cooking_entries.csv")

    processed_rows = []
    for i, row in df.iterrows():
        question_html = str(row.get("question", ""))
        answers_str = str(row.get("answers", ""))
        metadata_str = str(row.get("metadata", "[]"))

        # Extract the link from metadata
        # Convert to double quotes if needed
        link_match = re.search(r'https?://[^"]*questions/\d+[^"]*', metadata_str.replace("'", '"'))
        question_link = link_match.group(0) if link_match else None

        # Clean the answers by removing non-text fields
        cleaned_answers = remove_non_text_fields(answers_str)

        # Clean question (remove HTML)
        question = clean_html(question_html)

        processed_rows.append({
            "question": question,
            "answers": cleaned_answers,
            "link": question_link
        })

    processed_df = pd.DataFrame(processed_rows, columns=["question", "answers", "link"])
    processed_df.to_csv("../../data/processed_cooking_data.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
