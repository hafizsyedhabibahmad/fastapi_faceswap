```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from retry import retry
from PIL import Image, ImageEnhance
import os
import uuid
import hashlib
import tempfile
import logging
from cachetools import TTLCache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the output directory to serve static files
app.mount("/output", StaticFiles(directory="output"), name="output")

# Configuration
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cache setup (TTL: 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path: str) -> bool:
    return os.path.exists(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

def compress_image(content: bytes, max_size: int = 512) -> bytes:  # Reduced max_size to 512 to save memory
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_in:
            temp_in.write(content)
            temp_in.flush()
            img = Image.open(temp_in.name)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_out:
                img.save(temp_out.name, "PNG", optimize=True, quality=85)
                with open(temp_out.name, "rb") as f:
                    return f.read()
    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        return content

def enhance_image(image_path: str) -> None:
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(2.0)
        img_enhanced.save(image_path, "PNG")
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")

def save_output_image(result_path: str, output_dir: str, output_name: str) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        img = Image.open(result_path)
        img = img.convert("RGB")
        img.save(output_path, "PNG")
        enhance_image(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error saving output image: {str(e)}")
        return ""

@retry(tries=3, delay=2, backoff=2)
async def face_swap(source_image: str, dest_image: str, source_face_idx: int = 1, dest_face_idx: int = 1) -> str:
    try:
        logger.info("Validating input files")
        if not all([validate_file(source_image), validate_file(dest_image)]):
            logger.error("Invalid input files")
            return "Invalid input files"

        logger.info("Connecting to Gradio client")
        client = Client("Dentro/face-swap")
        logger.info("Sending face swap request to Gradio")
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result):
            unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
            final_path = save_output_image(result, OUTPUT_FOLDER, unique_filename)
            if final_path:
                logger.info(f"Face swap successful, saved to: {final_path}")
                return final_path
            logger.error("Failed to save output")
            return "Failed to save output"
        logger.error("Face swap failed, no result returned")
        return "Face swap failed"
    except Exception as e:
        logger.error(f"Error in face_swap: {str(e)}")
        return f"Error: {str(e)}"

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "API is running"}

@app.post("/swap")
async def swap_faces(source_image: UploadFile = File(...), dest_image: UploadFile = File(...)):
    logger.info("Received request to /swap")
    if not source_image.filename or not dest_image.filename:
        logger.error("No file selected")
        raise HTTPException(status_code=400, detail="No file selected")

    if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
        logger.error("Invalid file format")
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, JPEG allowed")

    source_content = await source_image.read()
    dest_content = await dest_image.read()
    logger.info("Images read successfully")
    source_content = compress_image(source_content)
    dest_content = compress_image(dest_content)
    logger.info("Images compressed")

    cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"
    if cache_key in cache:
        result_url = cache[cache_key]
        logger.info(f"Cache hit: {result_url}")
        return {"result_image_url": f"/output/{os.path.basename(result_url)}"}

    with tempfile.TemporaryDirectory() as temp_dir:
        source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
        dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
        source_path = os.path.join(temp_dir, source_filename)
        dest_path = os.path.join(temp_dir, dest_filename)

        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(dest_path, "wb") as f:
            f.write(dest_content)
        logger.info("Temporary files saved")

        result = await face_swap(source_path, dest_path)
        if result.startswith("Error") or result == "Invalid input files" or result == "Failed to save output":
            logger.error(f"Face swap failed: {result}")
            raise HTTPException(status_code=500, detail=result)

        cache[cache_key] = result
        logger.info(f"Cache updated with: {result}")
        return {"result_image_url": f"/output/{os.path.basename(result)}"}
```
