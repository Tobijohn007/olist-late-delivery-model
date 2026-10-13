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

- `app.py`, `predictor.py`, `train_model.py`
- `requirements.txt` (the new one that `train_model.py` wrote, with your exact versions)
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

## Other option

Streamlit also offers its own free hosting, called Streamlit Community Cloud, made for Streamlit apps. Render works for more kinds of apps, which is why it is a good skill to show.
