import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, rand_score

data = pd.read_csv('smoke (1).csv')
data=data.drop(columns=["UTC"])
X = data.drop('Fire Alarm_Yes', axis=1)
y = data['Fire Alarm_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_models():
    model1=pickle.load(open('SVM.pkl', 'rb'))
    model2=pickle.load(open('kmeans.pkl', 'rb'))
    model4=pickle.load(open('GradientBoosting.pkl', 'rb'))
    model5=pickle.load(open('Bagging.pkl', 'rb'))
    model3=pickle.load(open('Stacking.pkl', 'rb'))
    model6 = load_model('Neural.h5')
    return model1, model2, model3, model4, model5, model6

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f1f3f6;
}
h1 {
    color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)


# Функции для каждой страницы
def page_developer_info():
    st.title("Информация о разработчике")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Сагалбаев Дамир Амангельдыевич")
        st.write("Номер учебной группы: ФИТ-222")
    with col2:
        st.header("Фотография")
        st.image("zrxUc--UfHk.jpg", width=300)  

    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")


def page_dataset_info():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Описание Датасета smoke_detector
    **Файл датасета:** `smoke.csv`

    **Описание:**
    Данный датасет содержит табличные данные с различными показаниями сенсоров. Включает следующие столбцы:

    - **UTC**: Временная метка в секундах с начала эпохи.
    - **Temperature[C]**: Температура в градусах Цельсия.
    - **Humidity[%]**: Влажность в процентах.
    - **TVOC[ppb]**: Общие летучие органические соединения в частях на миллиард.
    - **eCO2[ppm]**: Эквивалент диоксида углерода в частях на миллион.
    - **Raw H2**: Сырое считывание датчика водорода.
    - **Raw Ethanol**: Сырое считывание датчика этанола.
    - **Pressure[hPa]**: Атмосферное давление в гектопаскалях.
    - **PM1.0**: Частицы с диаметром 1,0 микрометра.
    - **PM2.5**: Частицы с диаметром 2,5 микрометра.
    - **NC0.5**: Концентрация частиц с диаметром более 0,5 микрометра.
    - **NC1.0**: Концентрация частиц с диаметром более 1,0 микрометра.
    - **NC2.5**: Концентрация частиц с диаметром более 2,5 микрометра.
    - **CNT**: Количество частиц.
    - **Fire Alarm_Yes**: Бинарный индикатор (0 или 1) наличия пожарной сигнализации.

    **Особенности предобработки данных:**
    - Удаление лишних столбцов, например, 'UTC'.
    - Обработка пропущенных значений.
    - Кодирование категориальных переменных.
    - нельзя удалять выбросы, потому что так мы теряем важную информацию
    """)


def page_data_visualization():
    st.title("Визуализации данных smoke")

    fig, ax = plt.subplots()

    sns.boxplot(y='Temperature[C]',data=data)
    ax.set_title('Выбросы температуры')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data['Pressure[hPa]'], kde=True, color="blue", label='Pressure[hPa]', ax=ax)
    sns.histplot(data['Humidity[%]'], kde=True, color="red", label='Humidity[%]', ax=ax)
    ax.set_title('Распределение Humidity[%] и Pressure[hPa]')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    correlation_matrix= data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".1f")
    ax.set_title('Матрица корреляций')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()

    sns.histplot(data['Humidity[%]'], kde=True, color="blue", label='Humidity[%]', ax=ax)
    sns.histplot(data['Temperature[C]'], kde=True, color="red", label='Temperature[C]', ax=ax)
    ax.set_title('Humidity[%] И Temperature[C]')
    ax.legend()
    st.pyplot(fig)



def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        feature_names = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]',
                         'PM1.0','PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=10.0)

        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model1,model2, model3, model4, model5, model6 = load_models()

            input_df = pd.DataFrame([input_data])

            # Проверяем, что данные изменились
            st.write("Входные данные:", input_df)

            # Делаем предсказания
            prediction_ml1 = model1.predict(input_df)
            prediction_ml3 = model3.predict(input_df)
            prediction_ml4 = model4.predict(input_df)
            prediction_ml5 = model5.predict(input_df)
            prediction_ml6 = (model6.predict(input_df) > 0.5).astype(int)  # Для нейронной сети

            # Вывод результатов
            st.success(f"Результат предсказания SVC: {prediction_ml1}")
            st.success(f"Результат предсказания Stacking: {prediction_ml3}")
            st.success(f"Результат предсказания GradientBoosting: {prediction_ml4}")
            st.success(f"Результат предсказания Bagging: {prediction_ml5}")
            st.success(f"Результат предсказания Neural: {prediction_ml6}")
    else:
        try:
            model1=pickle.load(open('SVM.pkl', 'rb'))
            model2=pickle.load(open('kmeans.pkl', 'rb'))
            model4=pickle.load(open('GradientBoosting.pkl', 'rb'))
            model5=pickle.load(open('Bagging.pkl', 'rb'))
            model3=pickle.load(open('Stacking.pkl', 'rb'))
            model6 = load_model('Neural.h5')

            # Подготовка тестовых данных
            # (здесь предполагается, что X_test и y_test уже подготовлены)

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model1.predict(X_test)
            prediction_ml2  = model2.fit_predict(X_test)
            predictions_ml4 = model4.predict(X_test)
            predictions_ml5 = model5.predict(X_test)
            predictions_ml3 = model3.predict(X_test)
            predictions_ml6 = model6.predict(X_test).round()  # Округление для нейронной сети

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = round(rand_score(y_test, prediction_ml2))
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность SVC: {accuracy_ml1}")
            st.success(f"Точность Kmeans: {accuracy_ml2}")
            st.success(f"Точность Stacking: {accuracy_ml3}")
            st.success(f"Точность GradienBoosting: {accuracy_ml4}")
            st.success(f"Точность Bagging: {accuracy_ml5}")
            st.success(f"Точность Neural: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")


if page == "Информация о разработчике":
    page_developer_info()
elif page == "Информация о наборе данных":
    page_dataset_info()
elif page == "Визуализации данных":
    page_data_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()
