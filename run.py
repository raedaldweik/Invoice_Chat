import os
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI

# 1) Load .env and set API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 2) Convert Excel to CSV (utf-8 with BOM) if needed
EXCEL_PATH = "Invoice.xlsx"
CSV_PATH = "Invoice.csv"
if not os.path.exists(CSV_PATH):
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# 3) Initialize the CSV agent (with dangerous code allowed)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_csv_agent(
    llm,
    CSV_PATH,
    pandas_kwargs={"encoding": "utf-8-sig"},
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True,
)

# 4) Your data dictionary for context
data_dictionary = """
| Column Name                        | Description                                                                                          |
|------------------------------------|------------------------------------------------------------------------------------------------------|
| اسم الشركة                          | The legal name of the company                                                                        |
| نوع الشركة                          | Type of the company (e.g., مساهمة for a joint-stock company)                                         |
| القطاع                              | Industry sector (e.g., النقل for Transportation)                                                     |
| رقم السجل التجاري                  | Commercial registration number of the company                                                         |
| العنوان                            | Full postal address, including street and suite                                                      |
| المدينة                             | City where the company is located (e.g., الدوحة)                                                     |
| المنطقة                             | Administrative region within the city (e.g., المنطقة 91)                                              |
| رقم الهاتف                          | Contact phone number                                                                                  |
| البريد الإلكتروني                   | Contact email address                                                                                 |
| خط العرض                            | Latitude coordinate in decimal degrees                                                               |
| خط الطول                            | Longitude coordinate in decimal degrees                                                              |
| رقم الفاتورة                        | Invoice identifier (e.g., INV-1000)                                                                  |
| تاريخ الفاتورة                      | Date when the invoice was issued (YYYY-MM-DD)                                                        |
| تاريخ الاستحقاق                    | Invoice due date (YYYY-MM-DD)                                                                        |
| نوع الفاتورة                        | Category of the invoice (e.g., فاتورة أجور, فاتورة خدمات)                                              |
| المبلغ (ر.ق)                       | Invoice amount in Qatari Riyal before tax                                                            |
| الضريبة (5%)                       | Tax amount at 5% of the invoice value                                                                 |
| المبلغ الإجمالي (ر.ق)              | Total amount in Qatari Riyal including tax                                                            |
| حالة الدفع                         | Payment status (مدفوعة for paid, غير مدفوعة for unpaid)                                               |
| طريقة الدفع                         | Payment method (e.g., بطاقة, تحويل بنكي, نقداً)                                                      |
| تقييم مخاطر الفاتورة                | Risk rating of the invoice (منخفض for Low, متوسط for Medium, مرتفع for High)                         |
| فجوة الإيرادات/الضريبة المحتملة (ر.ق) | Potential revenue/tax gap in Qatari Riyal if the invoice remains unpaid                              |
"""

# 5) Streamlit UI using chat primitives
st.title("Digital Assistant")
st.write("Ask me anything about your invoices!")

# initialize history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# render existing messages
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# chat input pinned at bottom
if prompt := st.chat_input("You:"):
    full_prompt = (
        f"Refer to the following data dictionary for context:\n\n"
        f"{data_dictionary}\n\n"
        f"{prompt}"
    )
    response = agent_executor.invoke({"input": full_prompt})["output"]

    # append to history
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", response))

    # display immediately
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(response)
