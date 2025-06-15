# Gender Bias Analysis Tool

A powerful tool for analyzing and reducing gender bias in text content — particularly useful for job postings and professional communications.

## 🌟 Features

- **Gender Bias Detection**: Identifies masculine and feminine-coded words in text
- **Bias Scoring**: Calculates a friendliness score (0–10) indicating gender balance
- **Dimension Classification**: Classifies text as *Agentic*, *Communal*, or *Balanced*
- **AI Paraphrasing**: Uses an advanced model to generate gender-neutral text rewrites
- **Visual Highlighting**: Clearly shows gender-coded words in the interface

---

## 🚀 Local Setup & Usage

1. **Clone the repo**  
   ```bash
   git clone https://github.com/liyanzhuo2022/GenderAnalyzer.git
   cd GenderAnalyzer
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**  
   ```python
   import nltk
   nltk.download('wordnet')
   ```

4. **Start server** (unified FastAPI + frontend)  
   ```bash
   uvicorn backend.main:app --reload
   ```

5. **Visit in browser**  
   ```
   http://localhost:8000/
   ```

## 🌍 Sharing with Others

Use ngrok (free) to share your local instance:

```bash
ngrok http 8000
```

You'll get a temporary public URL like:
```
https://xxxx-xxxx-xxx.ngrok-free.app
```

Send it to others — they can access your tool instantly.

## 📂 Project Structure

```
GenderAnalyzer/
├── README.md
├── analysis/
│   ├── data_cleaning.py
│   ├── lightGBM_inference.py
│   └── lightGBM_training.py
├── backend/
│   ├── gender_bias_calculation.py
│   ├── main.py
│   └── utils.py
├── frontend/
│   └── New-Ad-Tool-VueAPI.html
└── requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.