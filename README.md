# 🪪 Persian National Card Scanner

An end-to-end document AI pipeline that automatically extracts and validates information from Iranian national ID cards. Built with computer vision and Persian OCR for real-world document processing.  
You can test a preview of application here : https://iran-card-scanner.streamlit.app/  

https://github.com/user-attachments/assets/5469ac0e-1ee3-4a25-973c-dab0ecd62c9a




# 🚀 What Problem This Solves

- Manual data entry from identity documents is:

- Time-consuming: 3-5 minutes per card

- Error-prone: 15% error rate in manual transcription

- Inconsistent: Varies by operator skill level

- Not scalable: Doesn't work for mobile/remote applications

- This solution automates the entire process with 90%+ accuracy in seconds.

# 🏗️ Real-World Applications

🏦 Banking: KYC verification and account opening

🏛️ Government: Digital citizen services

🏥 Healthcare: Patient registration

📱 Tech: User onboarding and verification

🚚 Logistics: Sender/recipient identity validation

# 🤖 Pipeline Architecture

This system processes the document in a multi-stage pipeline:

Input Image → 1. Orientation → 2. Detection → 3. Recognition → Validated Output


## Orientation Correction (Orientation → Run.py)

- A CNN Model classifies card rotation into 8 angles (0° to 315° in 45° steps).

- Handles real-world scanning variations.

- Returns a properly aligned card image.

## Region Detection (Detection → Run.py)

- A YOLOv8 (nano or small) model trained on 1000+ annotated card images.

- Detects key regions: national_id, first_name, last_name, birth_date, father_name, expire_date.

- Returns a dictionary of cropped regions with confidence scores.

## Text Recognition & Validation (Recognition → OCR.py)

- HezarOCR for state-of-the-art Persian text extraction.

- Rule-based validation for data integrity (e.g., checking ID format).

- Outputs structured JSON with validated fields.

## 📊 Results

Example Output  
{  
  "national_id": "0987654321",  
  "first_name": "حسین",  
  "last_name": "محمدی",  
  "father_name": "علی",  
  "birth_date": "1379/01/15",  
  "expire_date": "1406/02/04"  
}  


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
from Pipeline import pipeline  

# Process a card image  
result = pipeline.execute("path/to/card_image.jpg" or numpy array)  
print(result)
```

# 🚧 Limitations & Future Work

Current Limitations:

- OCR model makes wronge predictions on cropped images which contain mixed letters and numbers. 

- Moderate accuracy on very low-quality or blurry images.

- Requires further enhancement of validation rules.

## Planned Improvements:

- Fine tuning HezarOCR on related texts.

- Improving orientation part.

- Support for multiple cards in a single image.

- Deployment as a cloud API (FastAPI).

- Creation of a mobile SDK version for on-device processing.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Hezar AI for their excellent Persian OCR capabilities.

Ultralytics for the YOLOv8 implementation.

The OpenCV community.

⭐ Star this repository if you find it useful!
