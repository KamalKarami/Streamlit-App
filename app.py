import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from absenteeism_module import absenteeism_model

def Input_Output():
    data = st.file_uploader("Please Upload Your File Here", type={"csv", "txt"})
    if data is not None:
        df = pd.read_csv(data)
        st.write("داده های آپلود شده:")
        st.write(df)
        model = absenteeism_model('model', 'scaler')
        try:
            model.load_and_clean_data(df) # استفاده از DataFrame آپلود شده
            result = ""
            if st.button("برای پیش‌بینی کلیک کنید"):
                result = model.predicted_outputs()
                st.balloons()
                st.success('خروجی به شرح زیر است:')
                st.write(result)
        except Exception as e:
            st.error(f"خطا در پردازش داده‌ها: {e}")

if __name__ == '__main__':
    Input_Output()
