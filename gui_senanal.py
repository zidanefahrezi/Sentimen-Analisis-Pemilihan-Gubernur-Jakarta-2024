import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import os

def select_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def add_paslon():
    name = paslon_name_entry.get()
    dataset = dataset_entry.get()
    image = image_path_entry.get()

    if not name or not dataset or not image:
        messagebox.showerror("Error", "Semua field harus diisi!")
        return

    paslon_list.insert("", "end", values=(name, dataset, image))
    paslon_name_entry.delete(0, tk.END)
    dataset_entry.delete(0, tk.END)
    image_path_entry.delete(0, tk.END)

def process_wordcloud(df, negative_lexicon_path, image_path, output_image_path):
    negative_lexicon = set(pd.read_csv(negative_lexicon_path, sep="\t", header=None)[0])

    df['cleaned_text'] = df['cleaned_text'].astype(str)

    def find_negative_words(text, negative_lexicon):
        return [word for word in text.split() if word in negative_lexicon]

    df['negative_words'] = df['cleaned_text'].apply(lambda x: find_negative_words(x, negative_lexicon))

    all_negative_words = [word for words in df['negative_words'] for word in words]
    word_counts = Counter(all_negative_words)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = np.where(image != 255, 0, 255).astype(np.uint8)
    processed_image[processed_image > 175] = 255
    processed_image[processed_image <= 175] = 0

    top_negative_words = dict(word_counts.most_common(500))

    wordcloud = WordCloud(
        width=processed_image.shape[1],
        height=processed_image.shape[0],
        background_color="black",
        mask=processed_image,
        contour_width=1,
        contour_color="white",
        colormap="Reds",
        prefer_horizontal=1.0,
        mode="RGB",
        random_state=42
    ).generate_from_frequencies(top_negative_words)

    wordcloud.to_file(output_image_path)

def process_data():
    items = paslon_list.get_children()
    if not items:
        messagebox.showerror("Error", "Tambahkan setidaknya satu paslon!")
        return

    model = model_path.get()
    vectorizer = vectorizer_path.get()
    negative_lexicon = negative_lexicon = "./negative.tsv" #filedialog.askopenfilename(title="Pilih Negative Lexicon")

    if not model or not vectorizer or not negative_lexicon:
        messagebox.showerror("Error", "Pilih semua path model, vectorizer, dan negative lexicon!")
        return

    try:
        clf = joblib.load(model)
        vec = joblib.load(vectorizer)
    except Exception as e:
        messagebox.showerror("Error", f"Gagal memuat model atau vectorizer: {e}")
        return

    results = []

    for item in items:
        name, dataset_path, image_path = paslon_list.item(item, "values")
        try:
            df = pd.read_csv(dataset_path)
            df['cleaned_text'] = df['cleaned_text'].astype(str).apply(word_tokenize).apply(lambda x: " ".join(x))
            X = vec.transform(df['cleaned_text'])
            df['predicted_label'] = clf.predict(X)

            sentiment_counts = df['predicted_label'].value_counts()
            sentiment_percentages = (sentiment_counts / len(df)) * 100

            labels = [
                f"{label}: {count} ({percentage:.1f}%)"
                for label, count, percentage in zip(
                    sentiment_counts.index, sentiment_counts, sentiment_percentages
                )
            ]

            plt.figure(figsize=(6, 6))
            plt.pie(
                sentiment_counts,
                labels=labels,
                startangle=140,
                colors=["green", "yellow", "red"]
            )
            plt.title(f"Sentiment Distribution for {name}")
            plt.show()

            wordcloud_output = os.path.join(os.path.dirname(dataset_path), f"{name}_wordcloud.png")
            process_wordcloud(df, negative_lexicon, image_path, wordcloud_output)
            results.append((name, sentiment_counts, wordcloud_output))

        except Exception as e:
            messagebox.showerror("Error", f"Error memproses paslon {name}: {e}")
            return

    most_positives = max(results, key=lambda x: x[1].get("Positif", 0))
    most_negatives = max(results, key=lambda x: x[1].get("Negatif", 0))

    result_text.set(
        f"Paslon dengan sentimen positif terbanyak: {most_positives[0]}\n"
        f"Paslon dengan sentimen negatif terbanyak: {most_negatives[0]}"
    )

    for _, _, wordcloud_path in results:
        img = plt.imread(wordcloud_path)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def delete_selected():
    selected_item = paslon_list.selection()
    if not selected_item:
        messagebox.showerror("Error", "Pilih paslon yang ingin dihapus!")
        return
    
    for item in selected_item:
        paslon_list.delete(item)


root = tk.Tk()
root.title("Analisis Sentimen Paslon")

input_frame = tk.Frame(root, padx=10, pady=10)
input_frame.pack(fill="x")

tk.Label(input_frame, text="Nama Paslon:").grid(row=0, column=0, sticky="w")
paslon_name_entry = tk.Entry(input_frame, width=30)
paslon_name_entry.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Dataset Path:").grid(row=1, column=0, sticky="w")
dataset_entry = tk.Entry(input_frame, width=30)
dataset_entry.grid(row=1, column=1, padx=5)
tk.Button(input_frame, text="Browse", command=lambda: select_file(dataset_entry)).grid(row=1, column=2, padx=5)

tk.Label(input_frame, text="Image Path:").grid(row=2, column=0, sticky="w")
image_path_entry = tk.Entry(input_frame, width=30)
image_path_entry.grid(row=2, column=1, padx=5)
tk.Button(input_frame, text="Browse", command=lambda: select_file(image_path_entry)).grid(row=2, column=2, padx=5)

tk.Button(input_frame, text="Add Paslon", command=add_paslon).grid(row=3, column=1, pady=10)

list_frame = tk.Frame(root, padx=10, pady=10)
list_frame.pack(fill="x")

paslon_list = ttk.Treeview(list_frame, columns=("Name", "Dataset", "Image"), show="headings")
paslon_list.heading("Name", text="Nama Paslon")
paslon_list.heading("Dataset", text="Dataset Path")
paslon_list.heading("Image", text="Image Path")
paslon_list.pack(fill="x")

# Tombol untuk menghapus paslon yang dipilih
tk.Button(list_frame, text="Delete Selected", command=delete_selected, bg="red", fg="white").pack(pady=10)

model_frame = tk.Frame(root, padx=10, pady=10)
model_frame.pack(fill="x")

tk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky="w")
model_path = tk.Entry(model_frame, width=40)
model_path.grid(row=0, column=1, padx=5)
tk.Button(model_frame, text="Browse", command=lambda: select_file(model_path)).grid(row=0, column=2, padx=5)

tk.Label(model_frame, text="Vectorizer Path:").grid(row=1, column=0, sticky="w")
vectorizer_path = tk.Entry(model_frame, width=40)
vectorizer_path.grid(row=1, column=1, padx=5)
tk.Button(model_frame, text="Browse", command=lambda: select_file(vectorizer_path)).grid(row=1, column=2, padx=5)

result_frame = tk.Frame(root, padx=10, pady=10)
result_frame.pack(fill="x")

result_text = tk.StringVar()
tk.Label(result_frame, textvariable=result_text, justify="left").pack()

tk.Button(root, text="Process Data", command=process_data, bg="blue", fg="white").pack(pady=10)

root.mainloop()
