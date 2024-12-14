import streamlit as st
import pandas as pd
import pickle as pk

# Load the pre-trained model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Streamlit app header
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title('Loan Prediction App')
st.write("Easily predict loan approval and estimate repayment amounts.")

# Layout: Two columns for input and results
col1, col2 = st.columns(2)

# Input Section
with col1:
    st.header('User Details')

    no_of_dep = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1, value=0)
    grad = st.radio('Education Level', ['Graduated', 'Not Graduated'], index=0)
    self_emp = st.radio('Are you Self-Employed?', ['Yes', 'No'], index=1)

    st.header('Financial Details')

    Annual_Income = st.number_input('Annual Income (in INR)', min_value=0, step=10000, value=500000)
    Loan_Amount = st.number_input('Loan Amount Requested (in INR)', min_value=0, step=10000, value=500000)
    Loan_Dur = st.number_input('Loan Duration (in years)', min_value=1, max_value=30, step=1, value=10)
    Cibil = st.number_input('CIBIL Score', min_value=300, max_value=900, step=1, value=700)
    Assets = st.number_input('Value of Assets (in INR)', min_value=0, step=10000, value=1000000)

# Convert categorical inputs to numerical values
if grad == 'Graduated':
    grad_s = 0
else:
    grad_s = 1

if self_emp == 'No':
    emp_s = 0
else:
    emp_s = 1

# Prediction Section
with col2:
    st.header('Loan Prediction Result')
    if st.button("Predict Loan Status"):
        # Prepare input data
        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        
        pred_data = scaler.transform(pred_data)
        prediction = model.predict(pred_data)

        if prediction[0] == 1:
            st.success('Congratulations! Your loan is approved.')

            # Loan Repayment Section
            st.header('Loan Repayment Calculator')
            interest_rate = 8.5  # Example SBI interest rate

            if Loan_Amount > 0 and Loan_Dur > 0:
                monthly_rate = interest_rate / 12 / 100
                months = Loan_Dur * 12

                # Calculate EMI using the formula: EMI = [P * r * (1 + r)^n] / [(1 + r)^n - 1]
                emi = Loan_Amount * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
                total_repayment = emi * months

                st.write(f"### Loan Amount: ₹{Loan_Amount:,.2f}")
                st.write(f"### Interest Rate: {interest_rate}% p.a.")
                st.write(f"### Loan Tenure: {Loan_Dur} years")
                st.write(f"### Monthly EMI: ₹{emi:,.2f}")
                st.write(f"### Total Repayment Amount: ₹{total_repayment:,.2f}")
        else:
            st.error('Sorry, your loan application was rejected.')
