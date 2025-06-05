# app.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gradio as gr
import os

# Load dataset from the same folder
csv_path = os.path.join(os.path.dirname(__file__), "Mental_Health_FAQ.csv")
df = pd.read_csv(csv_path)

# Rename columns if needed
df.columns = ["Question_ID", "Questions", "Answers"]

questions = df["Questions"].tolist()
answers = df["Answers"].tolist()

# Load sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

def chatbot_response(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    top_match_idx = similarity_scores.argmax().item()
    return answers[top_match_idx]

app = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask a mental health question..."),
    outputs="text",
    title="🧠 Mental Health Support Chatbot",
    description="Ask me mental health-related questions like:\n- What are symptoms of depression?\n- Can anxiety be cured?\n- What to do if I feel low?"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=10000)
