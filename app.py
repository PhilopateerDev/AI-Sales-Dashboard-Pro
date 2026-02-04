import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import io

# إعداد الصفحة
st.set_page_config(page_title="محلل البيانات الذكي", layout="wide")

st.title("📊 محلل المبيعات الذكي (ارفع ملفك وحلل بياناتك)")
st.markdown("قم برفع ملف المبيعات الخاص بك، وسيقوم النظام بتنظيفه، تحليله، والتنبؤ بالمستقبل.")

# --- 1. مرحلة رفع الملفات ---
st.sidebar.header("📁 مدخلات البيانات")
uploaded_file = st.sidebar.file_uploader("اختر ملف Excel أو CSV", type=['csv', 'xlsx'])

# دالة لتوليد بيانات افتراضية لو العميل مرفعش ملف
def load_default_data():
    data = {
        'Order_Date': pd.date_range(start='2025-01-01', periods=12, freq='M'),
        'Category': ['Electronics', 'Furniture'] * 6,
        'Quantity': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'Unit_Price': [100, 200] * 6,
        'Total_Sales': [1000, 3000, 2000, 5000, 3000, 7000, 4000, 9000, 5000, 11000, 6000, 13000]
    }
    return pd.DataFrame(data)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ تم رفع ملفك بنجاح!")
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")
        df = load_default_data()
else:
    st.info("💡 تعرض الآن بيانات تجريبية. ارفع ملفك من القائمة الجانبية لتحليل بياناتك الخاصة.")
    df = load_default_data()

# --- 2. معالجة وتجهيز البيانات ---
df.columns = [c.strip().title() for c in df.columns]
if 'Order_Date' in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df = df.sort_values("Order_Date")
df["Month_Num"] = range(1, len(df) + 1)

# --- 3. الذكاء الاصطناعي (التنبؤ) ---
X = df[['Month_Num']]
y = df['Total_Sales']
model = LinearRegression().fit(X, y)
next_month = np.array([[len(df) + 1]])
prediction = model.predict(next_month)[0]

# --- 4. عرض المؤشرات (Metrics) ---
col1, col2, col3 = st.columns(3)
col1.metric("إجمالي المبيعات", f"${df['Total_Sales'].sum():,.0f}")
col2.metric("عدد العمليات", f"{len(df)}")
col3.metric("توقع الشهر القادم", f"${prediction:,.2f}")

# --- 5. الرسوم البيانية ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    if 'Category' in df.columns:
        st.subheader("توزيع المبيعات حسب الفئة")
        fig1, ax1 = plt.subplots()
        df.groupby('Category')['Total_Sales'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        st.pyplot(fig1)

with c2:
    st.subheader("اتجاه المبيعات وتوقعات AI")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='blue', label='بيانات فعلية')
    ax2.plot(X, model.predict(X), color='red', linestyle='--', label='خط الاتجاه')
    ax2.scatter(next_month, [prediction], color='green', marker='*', s=200, label='توقع مستقبلي')
    ax2.legend()
    st.pyplot(fig2)

# --- 6. تحميل البيانات النظيفة ---
st.divider()
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Clean_Data')
st.download_button(
    label="📥 تحميل البيانات المعالجة (Excel)",
    data=buffer.getvalue(),
    file_name="Processed_Sales_Data.xlsx",
    mime="application/vnd.ms-excel"
)

if st.checkbox("عرض جدول البيانات"):
    st.dataframe(df)
