## Project Title
**AI-Powered Image Analysis Tool with Real-Time Object Detection**

## Project Description

Developed a sophisticated desktop application that leverages state-of-the-art AI models to automatically analyze images, generate scene descriptions, and detect objects with high precision.

### Key Features:
- **Intelligent Scene Understanding**: Utilizes BLIP (Bootstrapping Language-Image Pre-training) model to generate natural language descriptions of images
- **Advanced Object Detection**: Implements DETR (DEtection TRansformer) with 80+ object categories from the COCO dataset
- **Interactive Visualization**: Real-time hover effects that highlight detected objects with bounding boxes and confidence scores
- **Multilingual Support**: Automatic translation from English to French using Google Translate API
- **Professional Reporting**: Exports comprehensive analysis reports to Excel with embedded images and detection data
- **Modern UI/UX**: Clean, dark-themed interface with smooth animations and responsive design

### Technical Stack:
- **AI/ML**: Transformers (Hugging Face), PyTorch, BLIP, DETR
- **UI Framework**: Tkinter with custom rounded components
- **Image Processing**: PIL (Pillow)
- **Data Export**: OpenPyXL for Excel generation
- **Translation**: Google Translate API
- **Threading**: Asynchronous model loading and analysis

### Technical Highlights:
- Implemented confidence threshold filtering (80%) for accurate detections
- Created custom UI components with rounded corners and fade-in animations
- Optimized performance with multithreaded processing
- Built responsive canvas system with dynamic image scaling
- Designed bilingual label mapping for 80 COCO object categories

### Performance Metrics:
- Processes images in real-time with simultaneous description and detection
- Handles high-resolution images with automatic scaling
- Confidence-based filtering ensures 80%+ accuracy
- Supports JPEG, PNG formats

### Use Cases:
- Image content analysis for accessibility
- Automated inventory detection
- Security and surveillance applications
- Educational tools for computer vision
- Content moderation assistance

### Project Impact:
This application demonstrates proficiency in integrating cutting-edge AI models into practical desktop applications, combining deep learning, computer vision, and software engineering principles to create a user-friendly tool for automated image understanding.
