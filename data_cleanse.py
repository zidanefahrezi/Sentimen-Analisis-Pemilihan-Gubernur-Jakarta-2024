import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tkinter import Tk, filedialog, Button, Label, messagebox
from tkinter.ttk import Progressbar
import threading

nltk.download('stopwords')
nltk.download('punkt')

# 1. Konfigurasi Pembersihan Data
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words("indonesian"))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Hilangkan URL
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Hapus karakter non-alfabet
    text = text.lower()  # Konversi ke huruf kecil
    text = word_tokenize(text)  # Tokenisasi
    text = [word for word in text if word not in stop_words]  # Hilangkan stopwords
    text = [stemmer.stem(word) for word in text]  # Stemming
    return " ".join(text)

# 2. Fungsi GUI
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        global df
        df = pd.read_csv(file_path)
        if "full_text" not in df.columns:
            messagebox.showerror("Error", "File CSV tidak memiliki kolom 'full_text'.")
            return
        label_status.config(text=f"File berhasil dimuat: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal memuat file: {str(e)}")

def clean_data():
    if 'df' not in globals():
        messagebox.showerror("Error", "Silakan muat file CSV terlebih dahulu.")
        return
    
    def process_cleaning():
        try:
            progress_bar["value"] = 0
            progress_bar["maximum"] = len(df)
            cleaned_data = []
            
            for i, text in enumerate(df["full_text"]):
                cleaned_data.append(clean_text(text))
                progress_bar["value"] = i + 1
                root.update_idletasks()  # Memperbarui GUI
            
            df["cleaned_text"] = cleaned_data
            label_status.config(text="Pembersihan data selesai.")
            messagebox.showinfo("Info", "Pembersihan data berhasil.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membersihkan data: {str(e)}")
        finally:
            progress_bar["value"] = 0

    # Gunakan threading agar GUI tidak hang
    threading.Thread(target=process_cleaning).start()

def save_file():
    if 'df' not in globals() or "cleaned_text" not in df.columns:
        messagebox.showerror("Error", "Tidak ada data yang bisa disimpan. Pastikan data telah dibersihkan.")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return

    try:
        df.to_csv(save_path, index=False)
        label_status.config(text=f"File berhasil disimpan: {save_path}")
        messagebox.showinfo("Info", "File berhasil disimpan.")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")

# 3. GUI Utama
root = Tk()
root.title("Pembersih Data CSV")

label_status = Label(root, text="Belum ada file yang dimuat.", fg="blue")
label_status.pack(pady=10)

btn_load = Button(root, text="Muat File CSV", command=load_file)
btn_load.pack(pady=5)

btn_clean = Button(root, text="Bersihkan Data", command=clean_data)
btn_clean.pack(pady=5)

progress_bar = Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

btn_save = Button(root, text="Simpan File", command=save_file)
btn_save.pack(pady=5)

root.mainloop()
