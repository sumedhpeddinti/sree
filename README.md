# MRI GPT-4 Medical Assistant

This Streamlit app allows uploading MRI images and a doctor's report to generate a unified diagnosis, risk assessment, treatment suggestions, follow-up Q&A, and PDF export.

## Files
- `app_mri_gpt4_final_multilang_env_file.py` — main Streamlit app
- `requirements.txt` — Python dependencies

## Local setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```
3. Run locally:
   ```bash
   streamlit run app_mri_gpt4_final_multilang_env_file.py
   ```

## Deploy to Streamlit Cloud
1. Create a GitHub repo and push this folder.
2. In your Streamlit Cloud app settings, set the secret `OPENAI_API_KEY`.
3. Add `app_mri_gpt4_final_multilang_env_file.py` as the app file.
4. Deploy.

## Notes
- The app supports English, Hindi, Spanish, and French output.
- Keep `OPENAI_API_KEY` secret and do not commit `.env` to public repos.
