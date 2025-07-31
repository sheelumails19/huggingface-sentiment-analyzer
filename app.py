from transformers import pipeline
import gradio as gr
1

# Load the Hugging Face model
classifier = pipeline("sentiment-analysis")

# Function to use in the web app
def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 2)
    return f"{label} (Confidence: {score})"

# Build Gradio Interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analyzer 🤖",
    description="Enter a sentence and see if it's Positive or Negative!"
)

# Run the web app locally
interface.launch()
