import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re

# Sayfa başlığı
st.set_page_config(page_title='TEAM NUMBER 1 CÜMLE ANALİZİ', page_icon='🔍')
st.title('🔍 TEAM NUMBER 1 CÜMLE ANALİZİ')

# Sayaçları ve analiz sonuçlarını session state içinde başlatma
if "positive_count" not in st.session_state:
    st.session_state.positive_count = 0
if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0
if "neutral_count" not in st.session_state:
    st.session_state.neutral_count = 0
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame(columns=["İSİM", "CÜMLE", "TAHMİN", "GERİ BİLDİRİM"])

# Kullanıcı geçmişi ve profili için session state
if "user_history" not in st.session_state:
    st.session_state.user_history = {}

# Model ve vektörleştiriciyi session state içinde saklama
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

# Yan menü
menu = st.sidebar.selectbox("Menü", ["Uygulama Hakkında", "Bizler Hakkında", "Cümle Analizi", "Sonuçlar", "Kullanıcı Profili"])

if menu == "Uygulama Hakkında":
    st.markdown('Bu Uygulama Ne Yapabilir?')
    st.info('Bu uygulama kullanıcılara cümle analizi yapma imkanı sunar. Eğitilmiş model ve vektörleştirici dosyaları yükleyerek cümlelerin pozitif, nötr veya negatif olduğunu analiz edebilirsiniz.')
    st.markdown('Nasıl Kullanılır?')
    st.warning('Model ve vektörleştirici dosyalarını yükleyin, ardından analiz etmek istediğiniz cümleyi girin.')

elif menu == "Bizler Hakkında":
    st.markdown('Bu Projemizde Neyi Amaçlıyoruz?')
    st.warning('Biz, yapay zeka ve veri mühendisliği alanında eğitim alan dört kişilik bir ekip olarak, cümle tahmini üzerine çalışan Twitter ekonomi başlığı altında olan verilerin duygu analizi projesine odaklanıyoruz. Yüksek kaliteli veri analizi ve makine öğrenimi tekniklerini kullanarak, kullanıcıların cümlelerini tahmin etmeye yönelik yenilikçi çözümler geliştirmeyi hedefliyoruz. Amacımız, doğal dil işleme alanında öncü bir rol oynamak ve kullanıcı deneyimini geliştirmek için teknolojiyi en iyi şekilde kullanmaktır.')

elif menu == "Cümle Analizi":
    st.header('CÜMLE ANALİZİ')

    if st.session_state.model is None or st.session_state.vectorizer is None:
        uploaded_model = st.file_uploader("Lütfen Eğitilmiş Modeli Yükleyiniz.Örnek model için https://linksharing.samsungcloud.com/p4sxPQ0j7p10  dosyayı indiriniz.", type=["joblib8"])
        uploaded_vectorizer = st.file_uploader("Lütfen Vektörleştiriciyi Yükleyiniz.Örnek vektörleşirici için https://linksharing.samsungcloud.com/xfvNwZ2hpyKZ dosyayı indiriniz.", type=["joblib8"])

        if uploaded_model and uploaded_vectorizer:
            st.session_state.model = joblib.load(uploaded_model)
            st.session_state.vectorizer = joblib.load(uploaded_vectorizer)
            st.success("Model ve Vektörleştirici başarıyla yüklendi.")
    else:
        st.success("Model ve Vektörleştirici daha önce yüklendi.")

    if st.session_state.model and st.session_state.vectorizer:
        user_name = st.text_input("Lütfen Kullanıcı İsminizi Giriniz:")
        input_sentence = st.text_input("Lütfen Cümlenizi Giriniz:")

        if user_name and input_sentence:
            # Emojileri kaldırma butonu
            if st.button("Emojileri Kaldır"):
                input_sentence = re.sub(r'[^\w\s,]', '', input_sentence)
                st.write("Güncellenmiş Cümle:", input_sentence)

            # Noktalama işaretlerini kaldırma butonu
            if st.button("Noktalama İşaretlerini Kaldır"):
                input_sentence = re.sub(r'[^\w\s]', '', input_sentence)
                st.write("Güncellenmiş Cümle:", input_sentence)

            input_data = st.session_state.vectorizer.transform([input_sentence])
            prediction = st.session_state.model.predict(input_data)[0]
            st.write(f"{user_name}, tahmin edilen duygu: {prediction}")

            # Kullanıcı geri bildirimi
            feedback = st.radio(
                "Bu tahmini doğru buluyor musunuz?",
                ('Evet', 'Hayır')
            )

            # Sayaçları güncelle
            if prediction == 'pozitif':
                st.session_state.positive_count += 1
            elif prediction == 'negatif':
                st.session_state.negative_count += 1
            elif prediction == 'nötr':
                st.session_state.neutral_count += 1

            # Analiz sonucunu kaydet
            new_entry = pd.DataFrame({"İSİM": [user_name], "CÜMLE": [input_sentence], "TAHMİN": [prediction], "GERİ BİLDİRİM": [feedback]})
            st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, new_entry], ignore_index=True)

            # Kullanıcı geçmişini güncelle
            if user_name in st.session_state.user_history:
                st.session_state.user_history[user_name] = pd.concat([st.session_state.user_history[user_name], new_entry], ignore_index=True)
            else:
                st.session_state.user_history[user_name] = new_entry

            # Analiz sonucunu bir dosyaya kaydet
            st.session_state.analysis_results.to_csv("analysis_results.csv", index=False)
            st.success("Analiz sonucu başarıyla kaydedildi.")

        # Sayaçları göster
        st.write(f"Pozitif Yorum Sayısı: {st.session_state.positive_count}")
        st.write(f"Negatif Yorum Sayısı: {st.session_state.negative_count}")
        st.write(f"Nötr Yorum Sayısı: {st.session_state.neutral_count}")

elif menu == "Sonuçlar":
    st.header('Sonuçlar')

    if st.session_state.positive_count + st.session_state.negative_count + st.session_state.neutral_count > 0:
        st.write("Pozitif Yorum Sayısı:", st.session_state.positive_count)
        st.write("Negatif Yorum Sayısı:", st.session_state.negative_count)
        st.write("Nötr Yorum Sayısı:", st.session_state.neutral_count)

        # Yüzdeleri göster
        total = st.session_state.positive_count + st.session_state.negative_count + st.session_state.neutral_count
        positive_percentage = (st.session_state.positive_count / total) * 100
        negative_percentage = (st.session_state.negative_count / total) * 100
        neutral_percentage = (st.session_state.neutral_count / total) * 100

        st.write(f"Pozitif Yorum Yüzdesi: {positive_percentage:.2f}%")
        st.write(f"Negatif Yorum Yüzdesi: {negative_percentage:.2f}%")
        st.write(f"Nötr Yorum Yüzdesi: {neutral_percentage:.2f}%")

        # Grafik göster
        labels = ['Pozitif', 'Negatif', 'Nötr']
        counts = [st.session_state.positive_count, st.session_state.negative_count, st.session_state.neutral_count]

        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Eşit görünüm oranı pastanın bir daire olarak çizilmesini sağlar.

        st.pyplot(fig)

        # Geri bildirimleri analiz et
        feedback_counts = st.session_state.analysis_results["GERİ BİLDİRİM"].value_counts()
        feedback_labels = feedback_counts.index.tolist()
        feedback_values = feedback_counts.values.tolist()

        # Geri bildirim yüzdeleri
        feedback_total = sum(feedback_values)
        feedback_percentages = [(value / feedback_total) * 100 for value in feedback_values]

        st.write("Geri Bildirim Yüzdeleri:")
        for label, percentage in zip(feedback_labels, feedback_percentages):
            st.write(f"{label}: {percentage:.2f}%")

        # Geri bildirim grafiği
        fig, ax = plt.subplots()
        ax.pie(feedback_values, labels=feedback_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Eşit görünüm oranı pastanın bir daire olarak çizilmesini sağlar.

        st.pyplot(fig)

    else:
        st.write("Henüz herhangi bir analiz yapılmadı.")

elif menu == "Kullanıcı Profili":
    st.header('Kullanıcı Profili ve Geçmişi')

    user_name = st.text_input("Kullanıcı İsmini Giriniz:")

    if user_name:
        if user_name in st.session_state.user_history:
            user_data = st.session_state
