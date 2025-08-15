import streamlit as st
import os
import io
import base64
from PIL import Image, UnidentifiedImageError
from typing import Optional
import logging
from dotenv import load_dotenv
from core.predict import ImageClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.getcwd(), "model", "best_model.pth"))

# Validate environment variables
if not MODEL_PATH:
    logger.error("Missing MODEL_PATH environment variable")
    raise RuntimeError("Environment variables not set")

# Food classification class names
class_name = {
    0: 'apple_pie', 1: 'baby_back_ribs', 2: 'baklava', 3: 'beef_carpaccio', 4: 'beef_tartare',
    5: 'beet_salad', 6: 'beignets', 7: 'bibimbap', 8: 'bread_pudding', 9: 'breakfast_burrito',
    10: 'bruschetta', 11: 'caesar_salad', 12: 'cannoli', 13: 'caprese_salad', 14: 'carrot_cake',
    15: 'ceviche', 16: 'cheese_plate', 17: 'cheesecake', 18: 'chicken_curry', 19: 'chicken_quesadilla',
    20: 'chicken_wings', 21: 'chocolate_cake', 22: 'chocolate_mousse', 23: 'churros', 24: 'clam_chowder',
    25: 'club_sandwich', 26: 'crab_cakes', 27: 'creme_brulee', 28: 'croque_madame', 29: 'cup_cakes',
    30: 'deviled_eggs', 31: 'donuts', 32: 'dumplings', 33: 'edamame', 34: 'eggs_benedict',
    35: 'escargots', 36: 'falafel', 37: 'filet_mignon', 38: 'fish_and_chips', 39: 'foie_gras',
    40: 'french_fries', 41: 'french_onion_soup', 42: 'french_toast', 43: 'fried_calamari', 44: 'fried_rice',
    45: 'frozen_yogurt', 46: 'garlic_bread', 47: 'gnocchi', 48: 'greek_salad', 49: 'grilled_cheese_sandwich',
    50: 'grilled_salmon', 51: 'guacamole', 52: 'gyoza', 53: 'hamburger', 54: 'hot_and_sour_soup',
    55: 'hot_dog', 56: 'huevos_rancheros', 57: 'hummus', 58: 'ice_cream', 59: 'lasagna',
    60: 'lobster_bisque', 61: 'lobster_roll_sandwich', 62: 'macaroni_and_cheese', 63: 'macarons',
    64: 'miso_soup', 65: 'mussels', 66: 'nachos', 67: 'omelette', 68: 'onion_rings', 69: 'oysters',
    70: 'pad_thai', 71: 'paella', 72: 'pancakes', 73: 'panna_cotta', 74: 'peking_duck',
    75: 'pho', 76: 'pizza', 77: 'pork_chop', 78: 'poutine', 79: 'prime_rib', 80: 'pulled_pork_sandwich',
    81: 'ramen', 82: 'ravioli', 83: 'red_velvet_cake', 84: 'risotto', 85: 'samosa', 86: 'sashimi',
    87: 'scallops', 88: 'seaweed_salad', 89: 'shrimp_and_grits', 90: 'spaghetti_bolognese',
    91: 'spaghetti_carbonara', 92: 'spring_rolls', 93: 'steak', 94: 'strawberry_shortcake',
    95: 'sushi', 96: 'tacos', 97: 'takoyaki', 98: 'tiramisu', 99: 'tuna_tartare', 100: 'waffles'
}

# Food categorization dictionary
food_categories = {
    "Meals & Main Courses": [
        'bibimbap', 'breakfast_burrito', 'chicken_curry', 'chicken_quesadilla', 'clam_chowder',
        'club_sandwich', 'croque_madame', 'dumplings', 'eggs_benedict', 'filet_mignon', 'fish_and_chips',
        'french_onion_soup', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
        'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'miso_soup',
        'omelette', 'pad_thai', 'paella', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'prime_rib',
        'pulled_pork_sandwich', 'spaghetti_bolognese', 'spaghetti_carbonara', 'tacos', 'takoyaki'
    ],
    "Baked Goods & Pastries": [
        'apple_pie', 'baklava', 'beignets', 'bread_pudding', 'cannoli', 'carrot_cake', 'chocolate_cake',
        'churros', 'cup_cakes', 'donuts', 'french_toast', 'macarons', 'pancakes', 'red_velvet_cake',
        'strawberry_shortcake', 'waffles'
    ],
    "Appetizer & Side Dishes": [
        'beet_salad', 'bruschetta', 'caesar_salad', 'caprese_salad', 'ceviche', 'deviled_eggs', 'edamame',
        'falafel', 'french_fries', 'fried_calamari', 'garlic_bread', 'greek_salad', 'grilled_cheese_sandwich',
        'hummus', 'nachos', 'onion_rings', 'poutine', 'samosa', 'seaweed_salad', 'spring_rolls'
    ],
    "Meat & Seafood": [
        'baby_back_ribs', 'beef_carpaccio', 'beef_tartare', 'chicken_wings', 'crab_cakes', 'escargots',
        'foie_gras', 'grilled_salmon', 'mussels', 'oysters', 'sashimi', 'scallops', 'shrimp_and_grits',
        'steak', 'sushi', 'tuna_tartare'
    ],
    "Dairy Products & Desserts": [
        'cheese_plate', 'cheesecake', 'chocolate_mousse', 'creme_brulee', 'frozen_yogurt', 'ice_cream',
        'panna_cotta', 'tiramisu'
    ],
    "Rice Grains & Noodles": [
        'fried_rice', 'gnocchi', 'ravioli', 'risotto', 'ramen'
    ],
    "Beverages": [],
    "Fruits & Vegetables": [],
    "Sauce Condiments and Seasonings": ['guacamole'],
}

# Initialize image classifier
try:
    classifier = ImageClassifier(model_path=MODEL_PATH, class_name=class_name)
    logger.info("Image classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to load image classifier model: {str(e)}")
    st.error("Failed to initialize image classifier")
    st.stop()

# Streamlit app
st.title("EcoHarvest Food Classifier")
st.write("Upload a food image (JPG/PNG, max 5MB) to classify it into one of 101 food categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Validate file size
    contents = uploaded_file.read()
    if len(contents) > 5 * 1024 * 1024:
        st.error("File size exceeds 5MB")
    else:
        try:
            # Process image
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Save to temporary file for prediction
            with open("temp_image.jpg", "wb") as f:
                f.write(contents)
            
            # Predict
            label, output_image_path = classifier.predict("temp_image.jpg")
            category = next((cat for cat, foods in food_categories.items() if label in foods), "Uncategorized")

            # Display results
            st.success(f"Prediction: {label} (Category: {category})")

            # Display output image if available
            if output_image_path and os.path.exists(output_image_path):
                with open(output_image_path, "rb") as f:
                    output_image = base64.b64encode(f.read()).decode("utf-8")
                st.image(f"data:image/jpeg;base64,{output_image}", caption="Processed Image", use_container_width=True)

            # Clean up
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
            if output_image_path and os.path.exists(output_image_path):
                os.remove(output_image_path)

        except UnidentifiedImageError:
            st.error("Invalid image file")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Food categories section
st.subheader("Food Categories")
if st.button("Show Food Categories"):
    for label in class_name.values():
        category = next((cat for cat, foods in food_categories.items() if label in foods), "Uncategorized")
        st.write(f"{label}: {category}")