How to Run a proactive-failure-dashboard

Step 1: Install Python (if not already installed)
1. Visit https://www.python.org/downloads/
2. Download the latest version for your operating system.
3. Run the installer:
   - On Windows: Check the box that says 'Add Python to PATH' before clicking 'Install Now'.
   - On macOS/Linux: Follow the installer instructions or use a package manager like Homebrew (macOS): brew install python
4. Verify installation by running `python --version` in your terminal or command prompt.

Step 2: Install Required Python Libraries
Open a terminal or command prompt and run the following command:
pip install streamlit pandas numpy matplotlib plotly seaborn openpyxl requests

Step 3: Prepare Your Project Folder
Create a folder (e.g., server_dashboard) and place the following files inside:
- dashboard_app.py (your Streamlit script)
- Image_1.jpg (logo image)
- Hardware_Eventlog.xlsx (Excel file with event logs), Nj2Serverissue.xlsx and Nj2Serverissue.xlsx files.

Step 4: Create .streamlit/secrets.toml for Secure Webhook
Inside your project folder, create a subfolder named `.streamlit`.
Inside that, create a file named `secrets.toml` with the following content:

teams_webhook_url = "https://your-actual-webhook-url"

Replace the URL with your actual Microsoft Teams webhook.

Step 5: Run the Streamlit App
Navigate to your project folder in the terminal and run:
streamlit run dashboard_app.py

This will launch the dashboard in your default web browser.

Step 6: Troubleshooting
- Ensure all required files are in the same directory.
- Check for typos in the script or secrets.toml.
- If the browser doesn't open automatically, visit http://localhost:8501 manually.
