import pandas as pd
import streamlit as st


df = pd.read_csv("movies_prepared.csv")

# Cписок эмоций
emotions = ["Positive", "Neutral", "Negative"]


st.title("Приложение для аннотации эмоций в диалоге")
# Просим пользователя выбрать фильм
movies = df["movie_title"].unique()
movie = st.selectbox("Выберите фильм", movies, key='movie_select')
# Фильтруем данные по выбранному фильму
data = df[df["movie_title"] == movie].copy()
# Создаем новую колонку для выбранных эмоций
data["emotions"] = [[] for _ in range(len(data))]
# Отображаем диалоги для выбранного фильма
for i, row in data.iterrows():
    names = eval(row["names"])
    dialog = eval(row["dialog"])
    dialogues = [f"{name}: {replic}" for name,
                 replic in zip(names, dialog)]
    dialogue_text = "\n\n".join(dialogues)
    st.write(f"**Dialogue {i + 1}** \n\n {dialogue_text}")

    # Просим пользователя выбрать эмоции для диалога
    emotions_selected = []
    for j, replic in enumerate(dialog):
        emotion = st.multiselect(f"Выберите эмоции для реплик(и) {j+1}",
                                 emotions, key=f'emotions_select_{i}_{j}')
        emotions_selected.append(emotion)

    # Отображаем выбранные эмоции
    st.write("Выбранные эмоции:", emotions_selected)

    # Добавляем выбранные эмоции в датафрейм
    data.at[i, "emotions"] = emotions_selected

    # Добавляем разделитель
    st.markdown("---")

# Добавляем кнопку для загрузки размеченных данных
if st.button("Сохранить размеченные данные"):
    # Сохраняем размеченные данные в csv-файл
    data.to_csv("my_movies_prepared.csv", index=False)
