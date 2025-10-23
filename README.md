cat << EOF > README.md
# Fireworks.ai Vision Demo: Image KYC Processor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository demonstrates how to use [Fireworks.ai](https://fireworks.ai/) for fast, efficient access to AI vision processing models. Specifically, it extracts structured **KYC (Know Your Customer)** data—such as name, address, date of birth, ID number, and expiration date—from document images (e.g., passports, driver's licenses, or ID cards).

Fireworks.ai provides easy self-service access to powerful vision models like Llama 3.2 Vision or similar multimodal LLMs, making it ideal for rapid prototyping and production-grade applications.

> **Quick Start**: Sign up for a free account at [Fireworks.ai](https://fireworks.ai/) and add your API key to run the demo.

## Features
- **Vision-Powered Extraction**: Uses AI models to parse unstructured document images and output JSON-structured KYC data.
- **Modular Design**: Simple Python script for easy integration into larger workflows.
- **Fast Inference**: Leverages Fireworks.ai's optimized infrastructure for low-latency processing.
- **Extensible**: Easily swap models or add custom prompts for different document types.

## Prerequisites
- Python 3.8 or higher
- A Fireworks.ai account and API key (free tier available)
- Optional: \`Pillow\` for image handling (if processing local files)

## Installation
1. **Clone the Repository**:
   \`\`\`
   git clone https://github.com/databased/fireworks-ai-vision-demo-image-kyc_processor.git
   cd fireworks-ai-vision-demo-image-kyc_processor
   \`\`\`

2. **Install Dependencies**:
   Create a virtual environment (recommended) and install required packages:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   pip install fireworks-ai pillow
   \`\`\`
   - \`fireworks-ai\`: Official SDK for Fireworks.ai API.
   - \`pillow\`: For loading and validating images.

3. **Set Up Your API Key**:
   Create a \`.env\` file in the root directory (or use environment variables directly):
   \`\`\`
   FIREWORKS_API_KEY=your_api_key_here
   \`\`\`
   Replace \`your_api_key_here\` with your actual key from [Fireworks.ai dashboard](https://app.fireworks.ai/).

   Load it in your script using \`python-dotenv\` (install with \`pip install python-dotenv\` if needed).

## Usage
### Running the Demo
The core functionality is in \`kyc_processor.py\` (assumed based on project structure). This script takes an image path or URL as input, sends it to Fireworks.ai's vision model, and extracts KYC fields via a custom prompt.

1. **Basic Command-Line Run**:
   \`\`\`
   python kyc_processor.py --image_path /path/to/your/document.jpg
   \`\`\`
   - Output: JSON with extracted data, e.g.:
     \`\`\`json
     {
       "name": "John Doe",
       "address": "123 Main St, Anytown, USA",
       "dob": "01/01/1990",
       "id_number": "123-45-6789",
       "expiration_date": "12/31/2030"
     }
     \`\`\`

2. **Example Code Snippet** (from \`kyc_processor.py\`):
   \`\`\`python
   import os
   from fireworks.client import Fireworks
   from PIL import Image
   import base64
   from dotenv import load_dotenv

   load_dotenv()

   def extract_kyc_from_image(image_path: str) -> dict:
       # Initialize Fireworks client
       fw = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

       # Load and encode image
       with open(image_path, "rb") as img_file:
           img_data = base64.b64encode(img_file.read()).decode('utf-8')

       # Custom prompt for KYC extraction
       prompt = """
       Analyze this document image and extract the following KYC fields in JSON format:
       - name: Full name
       - address: Full address
       - dob: Date of birth (MM/DD/YYYY)
       - id_number: ID or passport number
       - expiration_date: Expiration date (MM/DD/YYYY)
       Only respond with valid JSON.
       """

       # Generate with vision model
       response = fw.chat.completions.create(
           model="llama-3.2-vision:11b",  # Or another vision model
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": prompt},
                       {
                           "type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                       }
                   ]
               }
           ],
           max_tokens=300
       )

       # Parse JSON from response
       extracted_data = response.choices[0].message.content.strip()
       return json.loads(extracted_data)  # Requires import json

   # Example usage
   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument("--image_path", required=True)
       args = parser.parse_args()
       result = extract_kyc_from_image(args.image_path)
       print(result)
   \`\`\`

   Customize the \`prompt\` for accuracy on specific document types.

### Testing with Sample Images
- Add sample KYC document images (e.g., \`samples/passport.jpg\`) to a \`samples/\` directory.
- Run: \`python kyc_processor.py --image_path samples/passport.jpg\`

## Configuration
- **Model Selection**: Change the \`model\` parameter in the API call (e.g., to \`qwen-2-vl:7b\` for better multilingual support).
- **Prompt Engineering**: Tweak the prompt for higher precision or additional fields.
- **Batch Processing**: Extend the script to handle multiple images via a loop.

## Contributing
Contributions are welcome! Please:
1. Fork the repo.
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`).
3. Commit changes (\`git commit -m 'Add amazing feature'\`).
4. Push to the branch (\`git push origin feature/amazing-feature\`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Fireworks.ai](https://fireworks.ai/) for the awesome API and models.
- Built with ❤️ by [databased](https://github.com/databased).

---

*Questions? Open an issue or reach out! 🚀*
EOF
