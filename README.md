# amazon-ml-hackathon-
This notebook builds a multimodal regression model to predict product prices using:

📝 Text data → from catalog descriptions (via SentenceTransformer embeddings)

🖼️ Image data → from product images (via EfficientNetB0 CNN features)

It then combines both types of features and trains a neural network regressor using TensorFlow/Keras.

🔹 Step-by-Step Breakdown
1. Import Libraries

Imports all the required packages:

os, math, random, requests → file I/O, randomness, web requests.

PIL.Image, BytesIO → image loading and resizing.

numpy, pandas → data manipulation.

matplotlib, seaborn → visualization.

scikit-learn → preprocessing, train/test splitting.

tensorflow.keras → deep learning.

SentenceTransformer → to generate semantic text embeddings.

2. Configuration

Defines constants:

CSV_PATH = "/content/Copy of train.csv"
IMAGES_DIR = "./images_cache"
MAX_SAMPLES = 1000
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 3


Sets random seeds for reproducibility and creates image storage directory.

3. Load Dataset

Loads the training CSV file:

df = pd.read_csv(CSV_PATH)
df = df[['sample_id', 'catalog_content', 'image_link', 'price']].dropna()


Keeps only necessary columns and drops missing data.
Then shuffles rows and optionally limits to 1000 samples for quick experimentation.

4. Download & Cache Images

For each image link in the dataset:

Downloads the image via requests

Converts it to RGB, resizes to 224×224, and saves as .jpg

Keeps track of valid image paths in the DataFrame

This ensures that images are downloaded once and cached locally.

5. Text Embeddings

Uses SentenceTransformer (“all-MiniLM-L6-v2”) to encode the text descriptions:

embeddings = sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True)


Each catalog description becomes a 384-dimensional dense vector capturing its meaning.

6. Image Feature Extraction

Uses EfficientNetB0 (pretrained on ImageNet) without the top classification layer:

cnn = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')


Each image is preprocessed and passed through the CNN to extract a 1280-dimensional feature vector.

7. Combine Text + Image Features

Scales text and image embeddings using StandardScaler

Scales target price

Concatenates the normalized text and image features:

X = np.concatenate([X_text_s, X_img_s], axis=1)


→ The final input vector per product = 384 (text) + 1280 (image) = 1664 dimensions.

8. Train-Validation-Test Split

Splits the combined dataset:

80% for training

10% for validation

10% for testing

9. Build Regression Model

Creates a fully-connected neural network:

Input → Dense(512, relu) → Dropout(0.3)
      → Dense(128, relu) → Dense(64, relu)
      → Dense(1)  # output price


Compiled with Adam optimizer and MSE loss.

10. Train Model

Trains the regressor:

Monitors validation loss with early stopping (patience=3)

Saves best weights automatically.

11. Plot Training Curves

Plots training vs validation loss to visualize convergence and overfitting.

12. Evaluate on Test Data

Evaluates model performance:

Reports MSE and RMSE on scaled data

Converts predictions back to original price scale

Computes real RMSE and MAE in rupees/dollars.

13. Save Model and Scalers

Saves:

Trained model as multimodal_price_model.keras

Fitted StandardScalers for reuse via joblib.dump

14. Predict on Test Data & Generate Submission

For unseen test data:

Downloads test images

Encodes text with SentenceTransformer

Extracts CNN image features

Applies saved scalers

Combines both and predicts prices

Saves final output as submission.csv

15. Download Submission File

Uses Colab’s file API to download the generated CSV.

⚙️ Summary of Data Flow
train.csv
   ├── catalog_content → SentenceTransformer → text embeddings
   ├── image_link → EfficientNetB0 → image embeddings
   └── price → scaled target

[text embeddings | image embeddings] → Combined → Neural Net → Predicted price
