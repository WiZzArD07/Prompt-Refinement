# 🔒 Jailbreak Detection System

A comprehensive security system that detects adversarial content in both images and text prompts. This project helps protect AI systems from potential misuse and harmful content.

## Features

### Image Analysis
- Detects adversarial images using a trained neural network
- Supports common image formats (JPG, JPEG, PNG)
- Provides confidence scores for predictions
- Offers prompt refinement suggestions

### Text Prompt Analysis
- Real-time analysis of text prompts for security risks
- Detects multiple types of adversarial patterns:
  - Dangerous content (weapons, explosives, etc.)
  - Bypass attempts
  - Code injection attempts
  - Social engineering
  - Suspicious instruction patterns
- Risk level classification (High, Medium, Low, Safe)
- Detailed issue reporting and recommendations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jailbreak_detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file:
- Place the trained model file (`best_model.pth`) in the `models` directory
- If you don't have a trained model, you can train one using the provided scripts

## Project Structure

```
jailbreak_detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── models/               # Directory for model files
├── data/                 # Training data directory
│   ├── clean/           # Clean images
│   └── adversarial/     # Adversarial images
├── scripts/             # Utility scripts
│   ├── train_model.py   # Model training script
│   └── generate_sample_data.py  # Sample data generation
└── utils/               # Utility modules
    ├── model.py         # Model architecture
    └── prompt_analyzer.py  # Text prompt analysis
```

## Usage

### Running the Web Interface

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

### Using the Image Analysis

1. Click on the "Image Analysis" tab
2. Upload an image using the file uploader
3. Click "Detect Adversarial" to analyze the image
4. Use "Refine Prompt" to get suggestions for improving the image

### Using the Prompt Analysis

1. Click on the "Prompt Analysis" tab
2. Enter your text prompt in the text area
3. Click "Analyze Prompt" to check for security risks
4. Review the risk level and recommendations

## Training Your Own Model

1. Generate sample data:
```bash
python scripts/generate_sample_data.py
```

2. Train the model:
```bash
python scripts/train_model.py
```

The trained model will be saved in the `models` directory.

## Security Features

The system includes multiple layers of security:

1. **Image Analysis**
   - Neural network-based detection
   - Confidence scoring
   - Prompt refinement suggestions

2. **Text Analysis**
   - Pattern matching for dangerous content
   - Multiple risk level classification
   - Suspicious combination detection
   - Code injection prevention
   - Social engineering detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web interface framework
- PyTorch for the neural network implementation
- All contributors who have helped improve the project

## Support

If you encounter any issues or have questions, please open an issue in the repository. 