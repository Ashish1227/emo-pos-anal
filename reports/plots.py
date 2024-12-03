import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "/Users/ashishgatreddi/Desktop/emo-pos-anal/fer_deepface/relative_emotion_log.csv" # Replace with your actual file name
emotion_data = pd.read_csv(csv_file)

# Ensure timestamp column is sorted
emotion_data = emotion_data.sort_values(by='timestamp')

# Plot the emotions over time
plt.figure(figsize=(12, 6))

# Plot emotions using scatter for distinct points
plt.scatter(emotion_data['timestamp'], emotion_data['emotion'], color='blue', alpha=0.7, label='Emotion')

# Add titles and labels
plt.title("Emotion Variation Over Time")
plt.xlabel("Timestamp (seconds)")
plt.ylabel("Emotion")
plt.grid(True, linestyle='--', alpha=0.5)

# Show legend
plt.legend(loc='upper right')

# Display the plot
plt.show()
