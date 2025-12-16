import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. KONFIGURASI & LOAD DATA
# ==========================================
file_input = "data_final.csv" 
df = pd.read_csv(file_input)

text_col = 'processed_text' 
df[text_col] = df[text_col].fillna('').astype(str)

# ==========================================
# 2. PERSIAPAN KAMUS (STEMMING OTOMATIS)
# ==========================================
print("Menyiapkan kamus dan melakukan stemming...")

# Init Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# A. DEFINISI FRASA (N-GRAM)
# Penting: Frasa ini harus diproses (stem + stopword removal) agar cocok dengan data
positive_phrases_raw = [
    "bebas dari keracunan", "bebas dari basi", "bebas dari korupsi",
    "tidak ada racun", "tidak basi", "sangat layak", "jauh dari kata basi",
    "kerja nyata", "sangat membantu"
]

negative_phrases_raw = [
    "kurang gizi", "tidak layak", "tidak enak", "tidak higienis", 
    "banyak lalat", "ada ulat", "bau anyep", "rasa hambar"
]

# Fungsi helper untuk menyamakan format frasa dengan data
def preprocess_match_key(text):
    # Lakukan stemming sederhana pada frasa
    # (Asumsi stopword sudah hilang di data, jadi kita ambil kata kuncinya saja)
    return stemmer.stem(text)

pos_phrases = [preprocess_match_key(p) for p in positive_phrases_raw]
neg_phrases = [preprocess_match_key(p) for p in negative_phrases_raw]

# B. DEFINISI KAMUS KATA (LEXICON)
pos_words_raw = [
    "sehat", "anak bangsa", "nutrisi",
    "mantap", "hebat", "dukung", "terima kasih", "lanjut", "gas", 
    "semangat", "optimis", "maju", "bangkit", "sejahtera", "makmur", 
    "berkah", "amanah", "nyata", "bukti", "komitmen", "solusi", 
    "terbantu", "menolong", "apresiasi", "salut", "best", "cinta", 
    "kenyang", "lezat", "sedap", "higienis", "bersih", "hangat", 
    "fresh", "segar", "lengkap", "mutu", "terjamin", "cerdas", 
    "pintar", "tumbuh", "kembang", "generasi", "emas", "penerus",
    "manfaat", "harapan", "memastikan", "merata", "berkualitas", 
    "mandiri", "berhasil", "kolaborasi", "sempurna", "ketahanan", 
    "bahagia", "meningkatkan", "terbaik", "pemerataan", "mendukung", 
    "menjamin", "strategis", "membangun", "unggul", "sukses"
]

neg_words_raw = [
    "basih", "ulatan", "keracun", "bushi", "ancur", "ilangg", 
    "badut", "dagelan", "wacana", "omdo", "pencitraan", "drama", 
    "lawak", "ngelawak", "topeng",
    "basi", "asam", "kecut", "bau", "anyep", "dingin", "keras", "alot", "mentah",
    "hambar", "kurang", "dikit", "sedikit", "pelit", "jelek", "buruk", "parah",
    "kotor", "jorok", "rambut", "lalat", "ulat", "belatung", "benda", "asing",
    "mual", "muntah", "sakit", "diare", "keracunan", "racun", "bahaya", "ngeri",
    "seram", "takut", "sampah", "plastik", "limbah", "kardus", "styrofoam", 
    "ompreng", "ngapain", "cape", "anjir", "modus", "waste", "kelaparan", "arogan", 
    "kroni", "rugi", "taik", "membahayakan", "bangsat", "mahal", "anjing", 
    "basah", "kocak", "masalah", "kasus", "gila", "bancakan", "sianida", 
    "menentang", "becus", "geng", "akalan", "keuntungan", "ketakutan", 
    "benerin", "hadeuh", "sumpah", "gak jelas", "gajelas", "harusnya",
    "korupsi", "maling", "tikus", "rampok", "garong", "sunat", "potong", "tilap",
    "anggaran", "dana", "duit", "uang", "pajak", "rakyat", "beban", "utang", "hutang",
    "bengkak", "boros", "hambur", "buang", "sia-sia", "percuma", "gagal", 
    "bohong", "palsu", "kampanye", "politik", "kepentingan",
    "bisnis", "cuan", "proyek", "tender", "bagi-bagi", "jatah", "oligarki", "kronik",
    "nepotisme", "keluarga", "dinasti", "impor", "asing", "swasta", "kapitalis",
    "kecewa", "sedih", "marah", "kesal", "benci", "muak", "lelah", "stres",
    "pusing", "bingung", "ribet", "susah", "sulit", "kacau", "rusak", "hancur",
    "antri", "lama", "lelet", "lambat", "telat", "batal", "stop", "hentikan",
    "tolak", "ganti", "hapus", "tarik", "aneh", "lucu", "suram", "bodoh", 
    "goblok", "tolol", "dungu", "mematikan", "merugikan", "korban", "tumbal"
]

# Stemming lexicon agar cocok dengan processed_text
# Gunakan set agar pencarian lebih cepat dan unik
pos_set = set([stemmer.stem(word) for word in pos_words_raw])
neg_set = set([stemmer.stem(word) for word in neg_words_raw])

print("Stemming lexicon selesai.")

# ==========================================
# 3. FUNGSI SENTIMEN
# ==========================================

def get_sentiment(text):
    if not text: return 'Netral'
    
    score = 0
    temp_text = text
    
    # 1. Cek N-gram / Frasa (Prioritas Tinggi)
    for phrase in pos_phrases:
        if phrase in temp_text:
            score += 2
            temp_text = temp_text.replace(phrase, "") # Hapus agar tidak double count
            
    for phrase in neg_phrases:
        if phrase in temp_text:
            score -= 2
            temp_text = temp_text.replace(phrase, "")
            
    # 2. Cek Kata per Kata (Lexicon)
    words = temp_text.split()
    found_pos = []
    found_neg = []
    
    for word in words:
        if word in pos_set:
            score += 1
            found_pos.append(word)
        elif word in neg_set:
            score -= 1
            found_neg.append(word)
            
    # 3. Logika Tie-Breaker (Jika skor 0)
    if score == 0:
        if len(found_neg) > 0:
            label = 'Negatif'
        elif len(found_pos) > 0:
            label = 'Positif'
        else:
            label = 'Netral'
    elif score > 0:
        label = 'Positif'
    else:
        label = 'Negatif'
        
    return label

# Terapkan fungsi
print("Sedang melakukan scoring sentimen...")
df['label_pred'] = df[text_col].apply(get_sentiment)

# Simpan hasil
output_csv = "hasil_sentimen_final.csv"
df.to_csv(output_csv, index=False)
print(f"Hasil disimpan di: {output_csv}")
print(df['label_pred'].value_counts())

# ==========================================
# 4. VISUALISASI
# ==========================================

# A. PIE CHART
plt.figure(figsize=(7, 7))
counts = df['label_pred'].value_counts()
colors = {'Positif': '#66b3ff', 'Negatif': '#ff9999', 'Netral': '#99ff99'}
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
        colors=[colors.get(x, '#cccccc') for x in counts.index], 
        startangle=90, shadow=True)
plt.title('Distribusi Sentimen (Data Final)')
plt.savefig('grafik_lingkaran_sentimen.png')
plt.show()

# B. WORDCLOUD
def generate_wordcloud(sentiment, colormap):
    text = " ".join(df[df['label_pred'] == sentiment][text_col])
    if not text.strip():
        print(f"Tidak ada kata untuk sentimen {sentiment}")
        return
        
    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud Sentimen {sentiment}')
    plt.savefig(f'wordcloud_{sentiment.lower()}.png')
    plt.show()

# Generate untuk Positif dan Negatif
generate_wordcloud('Positif', 'Greens')
generate_wordcloud('Negatif', 'Reds')


print("Visualisasi selesai dan disimpan.")
