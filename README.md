# talk2data
You can talk to a SQL database containing a Insurance company's data like Customer, Policy, Quote, Sales, Commissions, Claims, Address, Agent


https://github.com/user-attachments/assets/933e18da-8fc5-4f4d-be9e-fc222a20bdde

## Deployment to Streamlit Cloud

1. **Push to GitHub**  
   Ensure your repository (including `Talk2DataINSUR.py`, `requirements.txt`, `Procfile`, and the `data/` folder with your SQLite databases) is pushed to GitHub.

2. **Create a Streamlit Cloud App**  
   - Go to https://streamlit.io/cloud and log in with your GitHub account.  
   - Click **New app**, select your repo and branch, and set the **Main file** to `Talk2DataINSUR.py`.

3. **Configure Secrets**  
   - In the app settings, go to **Secrets** and add your Google API key:  
     ```ini
     GOOGLE_API_KEY = "<your-google-api-key>"
     ```
   - Do *not* commit your actual key into version control.

4. **Verify Requirements & Config**  
   - `requirements.txt` lists all Python dependencies.  
   - `Procfile` instructs Streamlit Cloud to run `streamlit run Talk2DataINSUR.py`.  
   - `.streamlit/config.toml` is configured for headless mode.

5. **Deploy**  
   Once configured, click **Deploy**. The app will build using your `requirements.txt` and launch the Streamlit interface.

6. **Updates**  
   - To apply code changes, push commits to GitHub and Streamlit Cloud will automatically rebuild.  
   - For secret changes (e.g., rotating API keys), update them in the app settings.

