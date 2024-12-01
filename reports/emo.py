openai_api_key = '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'
llama_api_key = 'LA-509f5720bd7b444b81d58f272a21e3ff4a1c7b9674c14c878f4503b48fc401d0'

import json
from llamaapi import LlamaAPI

# Initialize the LLaMA API
llama = LlamaAPI(llama_api_key)

# Load text and emotion labels from the file
text_file = "/Users/ashishgatreddi/Desktop/face-emo/whisper_virtual_time/conversation_output_with_emotions.txt" # Replace with your text file's name

with open(text_file, "r") as file:
    data = file.readlines()

# Process each line in the text file
for line in data:
    # Extracting speaker, text, and emotion
    try:
        parts = line.strip().split(" (audio emotion detected - ")
        speaker_and_text = parts[0]
        emotion_label = parts[1].rstrip(")")

        speaker, text = speaker_and_text.split(": ", 1)
    except (ValueError, IndexError):
        print(f"Skipping malformed line: {line}")
        continue

    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Analyze the following dialogue from {speaker}:\n"
                    f"\"{text.strip()}\"\n"
                    f"The detected audio emotion is: {emotion_label.strip()}.\n"
                    "Provide a detailed analysis of the text and its emotional context. "
                    "Consider whether the detected emotion aligns with the text's sentiment."
                ),
            },
        ],
        "functions": [],
        "stream": False,
    }

    # Execute the Request
    try:
        response = llama.run(api_request_json)
        print(f"Analysis for {speaker}:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error processing line: {line.strip()}\nError: {str(e)}")

# import os
# from openai import OpenAI

# client = OpenAI(
#     api_key=openai_api_key,  # This is the default and can be omitted
# )

# # Function to send text and emotion to GPT
# def analyze_text_with_emotions(input_file, output_file):
#     with open(input_file, 'r') as file:
#         conversation_data = file.readlines()

#     # Prepare the analysis prompt with an example
#     example_analysis = """
# Example:
# Speaker: "I am feeling great today!" (Emotion: Happy)
# Analysis: The speaker seems to be in a positive mood, expressing enthusiasm or contentment.
# ---
# """

#     results = []
#     for line in conversation_data:
#         if line.strip():  # Skip empty lines
#             prompt = (
#                 f"{example_analysis}"
#                 f"Speaker dialogue with detected emotion:\n{line}\n"
#                 f"Provide an analysis of the speaker's intent, tone, and overall mood:"
#             )
#             response = client.chat.completions.create(
#                 model="gpt-4o",  # Replace with the desired model (e.g., gpt-4, gpt-3.5-turbo)
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             results.append(f"{line.strip()} Analysis: {response['choices'][0]['message']['content'].strip()}")

#     # Save the results
#     with open(output_file, 'w') as file:
#         file.write("\n\n".join(results))

#     print(f"Analysis complete! Results saved to {output_file}")

# # Input and output file paths
# input_text_file = "/Users/ashishgatreddi/Desktop/face-emo/whisper_virtual_time/conversation_output_with_emotions.txt"
# output_analysis_file = "conversation_analysis.txt"

# analyze_text_with_emotions(input_text_file, output_analysis_file)

