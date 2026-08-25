# N_test Prediction GUI

Engineering dashboard for predicting `N_test` with the exported RF, ETR, and DTR tree models. The app performs inference only; it does not retrain or modify the models.

## Setup

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. Open PowerShell in this folder.
3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Keep `gui_export_bundle.json` in the same folder as `app.py`.
5. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

The eight inputs are constrained to the min/max values exported in the bundle. Predictions are blocked outside that validated domain. The current interface intentionally contains only the predictor and one-factor variable impact analysis; UQ, explainability, metrics, exports, and other extensions are not part of this version.