from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path):
    # TODO: Implement actual model processing here
    return "E = mc^2"  # Placeholder return


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
        logger.info(f"File saved as {filename}")

        # Process the image and get LaTeX equation
        latex_equation = process_image(filepath)

        return {"image_path": f"/uploads/{filename}", "latex_equation": latex_equation}

    logger.warning(f"Invalid file type: {file.filename}")
    return "Invalid file type", 400


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    logger.info(f"Serving uploaded file: {filename}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(port=8000, debug=True)
