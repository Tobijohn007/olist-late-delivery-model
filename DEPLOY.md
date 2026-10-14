# Run and publish the prediction app

## 1. Run it on your laptop

In the VS Code terminal, inside the `olist-project` folder (the same folder as `app.py`):

```
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

- `train_model.py` takes a minute or two. It saves the model to `results/model.joblib`. You only need to run it again if you change the model.
- Your browser opens at http://localhost:8501. If it does not, paste that address in.
- Keep the terminal open while you use the app. Press Ctrl+C to stop it.
- If you see "File does not exist: app.py", the terminal is in the wrong folder or `app.py` has not been copied in.

## 2. Put it on GitHub

Add these to your repo (drag and drop on the GitHub page works):

- `app.py`, `predictor.py`, `explain.py`, `chat.py`, `train_model.py`
- `requirements.txt` (the new one that `train_model.py` wrote, with your exact versions; it must list `groq`)
- the `results` folder, **including `model.joblib`** (it is small)
- the `charts` folder

Still do not upload `data/` or `deliveries.csv`.

## 3. Host it on Render (free)

1. Go to render.com and sign up with your GitHub account.
2. New, then **Web Service**, then pick your `olist-late-delivery-model` repo.
3. Fill in:
   - Language / runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Instance type: **Free**
4. If the build fails on a version error, add an environment variable `PYTHON_VERSION` set to the full version you use locally (run `python --version`, for example `3.13.5`).
5. Click **Create Web Service**. After a few minutes you get a public link like `https://olist-late-delivery-model.onrender.com`. Put it in your README and on your CV.

Free plan facts (from Render's documentation):

- The service goes to sleep after 15 minutes without visitors. The next visitor waits about a minute while it wakes up.
- You get 750 free hours a month.
- The disk is not permanent. This app does not save anything, so that is fine.

Tip: open the link yourself just before an interview or presentation so it is awake.

## 4. The chat assistant ("Ask the model" tab)

The app has a chat tab where you can ask things like "Why did this order get this score?" or "What could lower its risk?". A language model (Llama 3.3 70B, served by Groq) writes the answer.

**How it stays honest.** The assistant never sees your data or the model. It only gets a short fact sheet built by `explain.py`: the order on screen, the "Why this score?" numbers, the model's honest test results and the project findings. Its instructions tell it to use only those facts, to say "I do not know" otherwise, and to answer in plain words. The "Why this score?" table is calculated by the app, not by the chatbot: it changes one thing about the order to a typical value and shows how much the score drops. The chatbot only explains that table.

### Get a key (free account)
1. Sign up at console.groq.com and open **API Keys**, then **Create API Key**. Copy it once. It starts with `gsk_`.
2. I could not confirm Groq's current free limits or whether a card is needed. Check the limits page in your Groq account before you share the link widely.

### Use it on your laptop (Windows PowerShell)
In the same terminal window you use to start the app:

```
$env:GROQ_API_KEY="paste-your-key-here"
python -m streamlit run app.py
```

The key lasts only for that window. Without a key the app still works, and the chat tab says the chatbot is switched off.

### Use it on Render
1. Open your service on Render, then **Environment**, then **Add Environment Variable**.
2. Key: `GROQ_API_KEY`. Value: your key. Save. Render restarts the app.

### Keep the key safe
- Never put the key in a file, in the code or on GitHub. It goes only in the environment variable.
- If it ever ends up on GitHub, delete it in the Groq console and make a new one. Deleting the file from GitHub is not enough, because the history keeps it.

### Limits built in
- Each visitor gets 8 questions per visit (`MAX_QUESTIONS` in `app.py`).
- Questions are cut to 400 characters, answers to about 350 tokens, and only the last 6 messages are remembered.
- Anyone with your link can use the chat, so Groq's own limits on your key are the real cap.

### If the chat does not answer
- "Switched off, no GROQ_API_KEY": the variable is missing or has a typo in its name.
- "Could not answer right now": the full error is in the terminal (or the Render logs). A wrong or deleted key, a rate limit and a retired model name all land here.
- "Chat library is not installed": `groq` is missing from `requirements.txt`.
- To try another model, set `GROQ_MODEL` as an environment variable. The default is `llama-3.3-70b-versatile`.

### Text you can add to your README
> **Ask the model.** The app includes a chat assistant that explains why an order got its risk score. It is given only the order's numbers, a per-factor breakdown and the honest test results, and it is told not to guess.

## Other option

Streamlit also offers its own free hosting, called Streamlit Community Cloud, made for Streamlit apps. Render works for more kinds of apps, which is why it is a good skill to show.
