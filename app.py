import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .subheader {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open('MLP_model.pkl', 'rb'))

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    model_loaded = False

st.markdown("<h1 class='main-header'>💼 Employee Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Predict salary ranges using advanced machine learning</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### How it works
    This tool uses machine learning to predict the amount of salary of an employee salary based on various 
    professional and personal factors. Fill in the form with employee details to get a prediction.
    """)
    
    with st.form("prediction_form"):
        st.subheader("📋 Employee Information")

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            age = st.number_input("Age", min_value=17, max_value=75, value=30,
                                help="Age of the employee")
            workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 
                            'Self-emp-inc', 'Federal-gov', 'Others']
            workclass = st.selectbox("Work Class", workclass_options,
                                help="Type of employer")
            education_num = st.number_input("Education Years (1-16)", min_value=1, max_value=16, value=10,
                                        help="Total years of formal education")
            marital_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 
                          'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
            marital_status = st.selectbox("Marital Status", marital_options,
                                        help="Current marital status")
            occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical',
                            'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving',
                            'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv',
                            'Armed-Forces', 'Others']
            occupation = st.selectbox("Occupation", occupation_options,
                                    help="Primary occupation category")

        with form_col2:
            relationship_options = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']
            relationship = st.selectbox("Relationship", relationship_options,
                                    help="Relationship status in family")
            race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
            race = st.selectbox("Race", race_options,
                            help="Race/ethnicity of the employee")
            gender = st.selectbox("Gender", ['Male', 'Female'],
                                help="Gender of the employee")
            
            st.markdown("### 📊 Financial & Work Information")
            capital_gain = st.number_input("Capital Gain ($)", min_value=0, value=0,
                                        help="Capital gains in USD")
            capital_loss = st.number_input("Capital Loss ($)", min_value=0, value=0,
                                        help="Capital losses in USD")
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40,
                                        help="Average working hours per week")

        st.markdown("### 🌎 Geographic Information")
        country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico',
                      'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy',
                      'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia',
                      'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece',
                      'France', 'Ecuador', 'Ireland', 'Hong', 'Cambodia', 'Trinadad&Tobago',
                      'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary',
                      'Scotland', 'Holand-Netherlands', 'Not Listed']
        native_country = st.selectbox("Native Country", country_options,
                                    help="Country of origin")

        submitted = st.form_submit_button("🔍 Predict Salary")

with col2:
    if model_loaded:
        st.markdown("""
        ### 💡 Salary Insights
        Understanding the factors that influence salary predictions:
        """)
        
        with st.expander("📚 Education Impact"):
            st.write("""
            - Higher education years typically correlate with higher salaries
            - Professional specialization often leads to better compensation
            - Continuous learning can lead to salary growth
            """)
        
        with st.expander("💼 Work Experience"):
            st.write("""
            - Work class affects salary potential
            - Hours worked per week can influence earnings
            - Occupation type is a major salary determinant
            """)
        
        with st.expander("📊 Financial Factors"):
            st.write("""
            - Capital gains/losses reflect financial management
            - Additional income sources affect total earnings
            - Investment decisions can impact wealth building
            """)

if submitted and model_loaded:
    le = LabelEncoder()
    scaler = MinMaxScaler()
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    
    with st.container():
        st.subheader("📋 Input Summary")
        display_df = pd.DataFrame([{
            "Age": age,
            "Work Class": workclass,
            "Education Years": education_num,
            "Occupation": occupation,
            "Hours/Week": hours_per_week
        }])
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        with st.spinner('Analyzing employee data...'):
            categorical_columns = ['workclass', 'marital-status', 'occupation', 
                                'relationship', 'race', 'gender', 'native-country']
            
            for col in categorical_columns:
                input_data[col] = le.fit_transform(input_data[col])

            input_scaled = scaler.fit_transform(input_data)
            
            prediction = model.predict(input_scaled)
        categorical_columns = ['workclass', 'marital-status', 'occupation', 
                             'relationship', 'race', 'gender', 'native-country']
        
        for col in categorical_columns:
            input_data[col] = le.fit_transform(input_data[col])
        
        input_scaled = scaler.fit_transform(input_data)
        
        prediction = model.predict(input_scaled)
        
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.subheader("🎯 Salary Analysis")

        pred_proba = model.predict_proba(input_scaled)[0]
        confidence = max(pred_proba) * 100

        base_salary = 30000
        
        edu_factor = (education_num / 16) * 20000
        
        exp_factor = ((age - 17) / (75 - 17)) * 15000
        
        hours_factor = (hours_per_week / 100) * 10000
        
        occupation_tiers = {
            'Exec-managerial': 1.5,
            'Prof-specialty': 1.4,
            'Tech-support': 1.3,
            'Sales': 1.2,
            'Adm-clerical': 1.1,
            'Other-service': 1.0,
            'Craft-repair': 1.0,
            'Machine-op-inspct': 1.0,
            'Transport-moving': 0.9,
            'Handlers-cleaners': 0.9,
            'Farming-fishing': 0.9,
            'Protective-serv': 1.1,
            'Armed-Forces': 1.1,
            'Others': 1.0
        }
        
        occupation_multiplier = occupation_tiers.get(occupation, 1.0)
        
        estimated_salary = (base_salary + edu_factor + exp_factor + hours_factor) * occupation_multiplier
        
        if capital_gain > 0:
            estimated_salary *= 1.1
        if capital_loss > 0:
            estimated_salary *= 0.95
            
        salary_range_low = max(25000, estimated_salary * 0.85)
        salary_range_high = estimated_salary * 1.15
        
        st.markdown(f"### Estimated Salary Range")
        
        st.markdown(f"""
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
            <div style="background-color: {'#28a745' if confidence > 70 else '#ffc107'}; 
                      width: {confidence}%; height: 100%; border-radius: 10px;">
            </div>
        </div>
        <p style="text-align: center; margin-top: 5px;">Confidence: {confidence:.1f}%</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lower Estimate", f"${salary_range_low:,.0f}")
        with col2:
            st.metric("Upper Estimate", f"${salary_range_high:,.0f}")
        
        st.markdown("### 📊 Salary Factors Analysis")
        
        factors_df = pd.DataFrame({
            'Factor': ['Education', 'Experience', 'Work Hours', 'Occupation Level'],
            'Impact': [
                f"${edu_factor:,.0f}",
                f"${exp_factor:,.0f}",
                f"${hours_factor:,.0f}",
                f"{occupation_multiplier:.1f}x multiplier"
            ]
        })
        st.dataframe(factors_df, hide_index=True, use_container_width=True)
        
        st.markdown("### 💡 Career Growth Recommendations")
        recommendations = []
        
        if education_num < 12:
            recommendations.append("• Consider completing higher education to increase earning potential")
        if hours_per_week < 35:
            recommendations.append("• Explore full-time employment opportunities")
        if occupation_multiplier < 1.2:
            recommendations.append("• Look into upskilling for higher-paying roles")
        if capital_gain == 0:
            recommendations.append("• Consider investment opportunities for additional income")
            
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.caption(f"Prediction made on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    Employee Salary Prediction Tool | Developed with Streamlit | Data-driven HR solutions
</div>
""", unsafe_allow_html=True)
