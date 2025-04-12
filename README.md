---

# Substance Sense 🧠💊

**Substance Sense** is an intelligent chat interface that predicts overdose risk based on user input like age, gender, location, and substance. It leverages machine learning to provide insight into overdose likelihood and is tailored for public health awareness, research, and prevention.

## 🧩 Features

- 💬 Chat-based interface built with React + Framer Motion
- 🔮 Predictive backend using a trained Random Forest classifier
- 🌐 Flask API connected to a fine-tuned ML model
- 📊 Real-world population logic for adjusted overdose probabilities
- 📖 Beautifully formatted markdown responses
- ✨ Smooth animations with Framer Motion

## 🚀 Tech Stack

- **Frontend**: React, Vite, TailwindCSS, Framer Motion, React Markdown
- **Backend**: Flask, Scikit-learn, Pandas, NumPy
- **Model**: Random Forest classifier trained on real overdose dataset (~100K records)

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Hash650/darkdrive-v2.git
cd darkdrive-v2
```

### 2. Frontend Setup

Navigate to the `frontend` directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create a `.env` file inside `frontend/`:

```
VITE_BACKEND=http://localhost:5000
```

Run the development server:

```bash
npm run dev
```

### 3. Backend Setup

Navigate to the `backend` directory:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```
GEMINI_API_KEY=your_key_here
```

Then run the server:

```bash
python app.py
```

## 📸 Screenshots

> Coming soon...

## ✅ Example Prompt

```
What is the overdose probability for a 23 year old male in Fort Richmond using alcohol?
```
