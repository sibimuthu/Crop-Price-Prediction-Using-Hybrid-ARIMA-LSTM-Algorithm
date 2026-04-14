# Crop-Price-Prediction-Using-Hybrid-ARIMA-LSTM-Algorithm

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-blue.svg)
![Razorpay](https://img.shields.io/badge/Razorpay-Payment%20Gateway-blue.svg)

AgroFuture is a comprehensive, AI-driven agricultural platform designed to empower farmers and stakeholders. It seamlessly integrates a predictive analytics dashboard for crop price forecasting (using hybrid Deep Learning architectures) with an intuitive e-commerce marketplace equipped with secure payment gateways and OAuth authentication.

## 🌟 Key Features

* **Advanced Predictive Analytics:** 
  * Leverages a Hybrid Model (LSTM + SARIMA) to forecast daily prices for **Tomato, Onion, Potato**.
  * Incorporates **What-If Analysis** to gauge the impact of environmental factors (Rainfall, Temperature) on market prices.
* **Intelligent eCommerce & Orders:**
  * Fully integrated shopping experience with **Razorpay** payment gateway for seamless transactions.
  * Real-time order tracking timeline (Confirmed → Shipped → Out for Delivery → Delivered).
* **Robust Authentication:**
  * Secure user sign-up/login using Scrypt password hashing.
  * Single Sign-On (SSO) integration via **Google OAuth 2.0**.
* **Modern Database Infrastructure:**
  * Managed via **Supabase (PostgreSQL)**, ensuring highly scalable and available data storage for users and orders.
* **Asset & Data Management:**
  * Automated background removal utility for catalog images.
  * Synthetic data generators to handle historical data scarcity for accurate model training.

## 🛠️ Technology Stack

* **Backend:** Python (Flask), SQLAlchemy, Authlib
* **Machine Learning:** PyTorch, Scikit-Learn, Statsmodels (SARIMA)
* **Database:** PostgreSQL (hosted on Supabase)
* **Payments:** Razorpay API
* **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Templates
* **Data Processing:** Pandas, NumPy

## ⚙️ Prerequisites

* **Python 3.9+** installed on your local machine.
* **Git** installed.
* A **Supabase** account (for PostgreSQL database).
* A **Razorpay** Sandbox/Live account (for payment processing).
* **Google Cloud Console** project (for OAuth credentials).

## 🚀 Getting Started

### 1. Clone the Repository (or Initialize)

If not already cloned:
```bash
git clone <your-repository-url>
cd "Final year proj"
```

### 2. Install Dependencies

It is recommended to use a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Ensure you replace the configuration variables in `app.py` or export them to your environment (it is recommended to use `.env` files in production):
* `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
* `db_password` (Supabase Postgres Password)
* Razorpay `RAZORPAY_KEY_ID` & `RAZORPAY_KEY_SECRET`

### 4. Data Preparation & Model Training

Generate synthetic historical data and train the AI models:
```bash
python scripts/generate_synthetic.py
python src/models/lstm_model.py
```
*(This will generate the necessary `.pth` model files in the `reports/figures/` directory).*

### 5. Database Initialization

The app uses SQLAlchemy to automatically create tables (`User`, `Order`) upon running. You can test the connection explicitly:
```bash
python test_db_conn.py
```

### 6. Run the Application

Launch the Flask development server:
```bash
python app.py
```
Or use the provided batch script on Windows:
```cmd
run_project.bat
```

Access the dashboard at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## 💡 Utilities

### Automated Background Removal
For processing e-commerce product images:
```bash
pip install rembg[gpu] onnxruntime
python scripts/remove_bg.py <input_image_path> <output_image_path>
```

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your enhancements.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
