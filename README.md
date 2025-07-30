# Employee Salary Prediction 💼

A machine learning-powered web application that predicts employee salaries based on various professional and personal factors. Built with Streamlit and scikit-learn, this tool provides detailed salary analysis and career growth recommendations.

## Features 🌟

- **Salary Prediction**: Uses an MLP Classifier to predict salary ranges
- **Detailed Analysis**: Breaks down the factors influencing the salary prediction
- **Interactive Interface**: User-friendly form with comprehensive input options
- **Career Recommendations**: Personalized suggestions for career growth
- **Real-time Processing**: Instant predictions with confidence scores
- **Visual Insights**: Progress bars, metrics, and organized data presentation

- Dataset and notebook file is added for reference

## Installation 📦

1. Clone the repository:
   ```bash
   git clone https://github.com/Manjit345/Employee-Salary-Prediction.git
   cd Employee-Salary-Prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## How It Works 🔄

1. **Data Collection**: Users input various employee details including:
   - Personal Information (age, gender, race)
   - Professional Details (occupation, education, work class)
   - Financial Information (capital gain/loss, hours worked)
   - Geographic Information (native country)

2. **Processing**:
   - Converts categorical variables to numerical values
   - Scales numerical features
   - Applies the trained ML model
   - Calculates detailed salary components

3. **Output**:
   - Estimated salary range
   - Confidence score
   - Factor-wise breakdown
   - Career growth recommendations

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Built as part of Edunet Internship Project
