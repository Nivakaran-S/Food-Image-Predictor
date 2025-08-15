# 🍏 EcoHarvest – AI-Powered Food Image Classification API & Web App

EcoHarvest is an **AI-driven food recognition platform** capable of classifying images into **101 unique food categories** with high accuracy.  
It offers:

- A **FastAPI backend** for programmatic image classification and category retrieval.  
- A **Streamlit web interface** for an interactive, user-friendly experience.  
- A **pre-trained EfficientNet-B0 model** fine-tuned for food image recognition.  

> This project blends computer vision, deep learning, and modern web frameworks to create a practical, production-ready ML application.

---

## 🚀 Features

- **Deep Learning Model**: EfficientNet-B0 fine-tuned on a 101-class food dataset.
- **Dual Interface**:
  - **API**: Built with FastAPI for scalable programmatic access.
  - **Web App**: Built with Streamlit for end-user accessibility.
- **Image Processing**:
  - Accepts `.jpg`, `.jpeg`, `.png` images (max 5MB).
  - Automatically preprocesses and resizes images for prediction.
  - Returns predictions with annotated images.
- **Food Categorization**:
  - Groups recognized dishes into categories like *Meals*, *Desserts*, *Seafood*, etc.
- **Portable Deployment**:
  - Runs locally or in the cloud.
  - Model is separate from code for easy updates.

---

## 📂 Project Structure
```bash

├── app.py # FastAPI backend service
├── app2.py # Streamlit frontend application
├── core/ 
|      ├── predict.py
├── model/ # Folder to store downloaded model (.pth file)
|      ├── model.pth
├── .env
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```


---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ecoharvest.git
cd ecoharvest

```

### 2️⃣ Create and Activate Virtual Environment
```bash

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

```

### 3️⃣ Install Dependencies

```bash

pip install --upgrade pip
pip install -r requirements.txt

```

---

## 📥 Download the Pre-Trained Model

The EfficientNet-B0 model (best_model.pth) is not stored in this repository due to size constraints.

1. Download the model from:
📎 Download best_model.pth['https://drive.google.com/file/d/1wU7xL4goWLo4iNEKvhZ9ZMvXaNlchBru/view?usp=sharing']

2. Create a model folder in the project root:

```bash 

mkdir model

```

3. Move the file into the model folder:

```bash

mv /path/to/best_model.pth model/

```

The final path should be:

```bash

model/best_model.pth

```

## ▶️ Running the Applications
### Option 1: Run the FastAPI Backend

```bash

uvicorn app:app --host 0.0.0.0 --port 5000 --reload

```

The API will be available at:
http://localhost:5000

Key Endpoints:
- POST /predict – Upload an image and get the predicted food label & category.
- GET /food/categories – Retrieve all 101 food classes with their categories.
- GET / – API welcome message & endpoint guide.

### Example curl request:

```bash

curl -X POST "http://localhost:5000/predict" \
  -F "file=@test.jpg"

```

### Option 2: Run the Streamlit Web App

```bash

streamlit run app2.py

```

The web interface will open in your browser, allowing you to:
- Upload an image
- View predictions
- See food categories

--- 

## 📊 Supported Food Categories

EcoHarvest supports 101 distinct dishes organized into:
- 🥘 Meals & Main Courses
- 🥐 Baked Goods & Pastries
- 🥗 Appetizers & Side Dishes
- 🐟 Meat & Seafood
- 🍦 Dairy Products & Desserts
- 🍜 Rice, Grains & Noodles
- 🥑 Sauces, Condiments & Seasonings

(See app.py or app2.py for full mappings.)

---

## 🧠 Model Details

- Architecture: EfficientNet-B0 (from torchvision.models)
- Input Size: 224×224 RGB
- Framework: PyTorch
- Output: 101 food classes
- Preprocessing:
    - Resize → Normalize → Tensor Conversion
- Annotation: OpenCV-based text overlay on predicted images.

### Create a .env file in the root:

```bash

MODEL_PATH=./model/best_model.pth
HOST=0.0.0.0
PORT=5000

```

---

## 🧪 Example Prediction Flow

1. Upload pizza.jpg via API or web app.
2. Model predicts:

```bash

Label: pizza
Category: Meals & Main Courses

```

3. Annotated output image is returned/displayed.

---

## 📌 Future Improvements

- Add Docker support for containerized deployment.
- Implement real-time webcam food detection.
- Deploy API to AWS/GCP/Azure for public access.
- Add nutritional information API integration.

---

## 🤝 Contributing
Contributions are welcome!
Feel free to fork the repo, create a feature branch, and submit a PR.

---

## 💡 Author’s Note:
This project demonstrates practical ML deployment – from training a vision model to building full-stack interfaces.

It’s designed to be both technically solid and easy to use, making it ideal for real-world adoption.