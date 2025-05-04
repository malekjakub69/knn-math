from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS
import os
import pickle
import torch
from PIL import Image
from werkzeug.utils import secure_filename
import logging
import sys

# Add the app directory to Python path to import neural network modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.neural_network.model import LatexOCRModel
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get the absolute path to the uploads directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'checkpoints', 'best_model_by_accuracy.pth')
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'idx2word.pkl')

# Global variables for model and vocabulary
model = None
idx2word = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the model and vocabulary at server startup"""
    global model, idx2word
    
    logger.info("Loading vocabulary...")
    with open(VOCAB_PATH, "rb") as f:
        idx2word = pickle.load(f)
    
    logger.info("Loading model checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    vocab_size = len(idx2word)
    
    # Initialize model with same parameters as during training
    model = LatexOCRModel(
        encoder_dim=128,
        vocab_size=vocab_size,
        embedding_dim=64,
        dropout=0.65,
        num_transformer_layers=4
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded successfully")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image through the model and return LaTeX equation"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # First resize to fixed height while preserving aspect ratio
        target_height = 80  # Same as model's height parameter
        aspect_ratio = image.size[0] / image.size[1]
        new_width = min(int(target_height * aspect_ratio), 2048)  # Cap width at 2048
        image = image.resize((new_width, target_height), Image.Resampling.BICUBIC)
        
        # Ensure width is divisible by patch_size (8)
        patch_size = 8
        if new_width % patch_size != 0:
            padding_width = patch_size - (new_width % patch_size)
            # Create new image with padding
            padded_image = Image.new("RGB", (new_width + padding_width, target_height), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            image = padded_image
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move to device
        image = image.to(device)
        
        # Generate prediction
        with torch.no_grad():
            predicted_tokens = model.predict(image, beam_size=3)
        
        # Convert tokens to LaTeX string
        latex = ""
        for token in predicted_tokens:
            if token == 2:  # END token
                break
            if token in idx2word:
                latex += idx2word[token]
        
        return latex
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return "Error processing image"

@app.route("/")
def index():
    logger.info(f"Accessing index page from {request.remote_addr}")
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    logger.info(f"Upload request from {request.remote_addr}")
    if "file" not in request.files:
        logger.warning("No file part in request")
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        logger.warning("No selected file")
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logger.info(f"File saved as {filepath}")

        # Process the image and get LaTeX equation
        latex_equation = process_image(filepath)
        logger.info(f"Generated LaTeX: {latex_equation}")

        # Return URL relative to server root
        return {"image_path": f"/uploads/{filename}", "latex_equation": latex_equation}

    logger.warning(f"Invalid file type: {file.filename}")
    return "Invalid file type", 400

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    logger.info(f"Serving uploaded file: {filename} from {app.config['UPLOAD_FOLDER']}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    # Load model before starting server
    load_model()
    app.run(port=8000, debug=True)
