# 🪪 Persian National Card Scanner

An end-to-end document API that automatically extracts and validates information from Iranian national ID cards. Powered by FastAPI, YOLO, and Hezar OCR for real-world document processing. I have designed this application to be suitable for edge devices and fast enough. So far, I have achieved 900ms per card on CPU.

Preview:  

https://github.com/user-attachments/assets/3e801184-7ba1-4b10-bbdf-3ec79e6b3bf2

# 🚀 What Problem This Solves

Manual data entry from identity documents is:

- **Time-consuming** – 3–5 minutes per card  
- **Error-prone** – 15% error rate in manual transcription  
- **Inconsistent** – varies by operator skill level  
- **Not scalable** – doesn't work for mobile/remote applications  

This solution automates the entire process with 90%+ accuracy in seconds.

# 🏗️ Real-World Applications

- 🏦 Banking: KYC verification and account opening  
- 🏛️ Government: Digital citizen services  
- 🏥 Healthcare: Patient registration  
- 📱 Tech: User onboarding and verification  
- 🚚 Logistics: Sender/recipient identity validation  

# 🤖 Pipeline Architecture

This system processes the document in a multi-stage pipeline:

`Input Image (using FastAPI) → 1. Orientation → 2. Detection → 3. Recognition → Validated JSON`

## Orientation Correction (Orientation → run.py)

- A lightweight YOLO nano pose estimation model predicts the 4 corners of the card.  
- Image rotation is applied.  
- The perspective method in OpenCV flattens the image.

## Region Detection (Detection → run.py)

- A YOLO nano detection model is used to detect ROIs.  
- Detects key regions: `national_id`, `first_name`, `last_name`, `birth_date`, `father_name`, `expire_date`.  
- Returns a dictionary of cropped regions with confidence scores.

## Text Recognition & Validation (Recognition → OCR.py)

- HezarOCR for state-of-the-art Persian text extraction.  
- Rule-based validation for data integrity (e.g., checking ID format).  
- Outputs structured JSON with validated fields.

# 📊 Results

Example Output:

```json
{
  "national_id": "0987654321",
  "first_name": "حسین",
  "last_name": "محمدی",
  "father_name": "علی",
  "birth_date": "1379/01/15",
  "expire_date": "1406/02/04"
} 
```

## 🛠️ Installation
```cmd
# 1. Clone the repository
git clone https://github.com/your-username/Persian-National-Card-Scanner.git
cd Persian-National-Card-Scanner

# 2. Install dependencies
pip install -r requirements.txt

```

## 🎯 Usage
```python
python run.py
```
The FastAPI server waits on localhost:8000 for image POST requests.

# 🚧 Limitations & Future Work

Current Limitations:

- OCR model makes wronge predictions on cropped images which contain mixed letters and numbers. 

- Moderate accuracy on very low-quality or blurry images.

- Requires further enhancement of validation rules.

## Planned Improvements:

- Fine tuning HezarOCR on related texts.

- Fine tuning the Pose and Det models on more data

- Quantization and performance improvement for edge devices

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Hezar AI for their excellent Persian OCR capabilities.

Ultralytics for the YOLOv8 implementation.

The OpenCV community.

⭐ Star this repository if you find it useful!
