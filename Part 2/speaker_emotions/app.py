import pandas as pd
import streamlit as st
import random
pd.options.mode.chained_assignment = None


df = pd.read_csv('movies_prepared.csv')

# Список эмоций
labels = ["Positive", "Negative", "Neutral"]

df.dialog = df.dialog.apply(lambda x: eval(x) if isinstance(x, str) else x)
df.order = df.order.apply(lambda x: eval(x) if isinstance(x, str) else x)
df.characters = df.characters.apply(
    lambda x: eval(x) if isinstance(x, str) else x)
df.genres = df.genres.apply(lambda x: eval(x) if isinstance(x, str) else x)
df.names = df.names.apply(lambda x: eval(x) if isinstance(x, str) else x)


mapping = {i: k for i, k in enumerate(labels)}

st.sidebar.header('Выбор фильма и персонажа')
movie_title = st.sidebar.selectbox("Название фильма",
                                   df.movie_title.unique().tolist())
character = st.sidebar.selectbox("Персонаж",
                                 [' '] + df[
                                  df.movie_title == movie_title]
                                 .names.explode().unique().tolist(),
                                 disabled=False)
movie_id_ = df.query("movie_title == @movie_title").movie_id.unique().item()


def stat_for_genre(movie_id, character=None):
    """Функция принимает id фильма и имя персонажа в качестве аргументов,
    возвращает фрейм данных с частотами эмоций для каждого персонажа в фильме.
    """
    df_final = df.query("movie_id == @movie_id")
    df_final["emotion"] = df_final.dialog.apply(
        lambda x: [random.randint(0, 2) for _ in x])
    df_final["emotion"] = df_final.emotion.apply(
        lambda x: [mapping[i] for i in x])
    df_final.reset_index(inplace=True, drop=True)
    df_final = df_final[["names", "emotion"]]
    df_final = df_final.explode(["names", "emotion"])
    df_final = df_final.groupby("names")["emotion"].value_counts(
        normalize=True).to_frame("frequency")
    # проверяем персонажа, если это не None, то находим во фрейме данных
    if character is not None:
        df_to_return = df_final.loc[character]
        df_to_return.index.names = ["Эмоция"]
        return df_to_return.rename(columns={
            "frequency": f"Соотношение для персонажа {character}"})
    else:
        df_final.index.names = ["Персонаж", "Эмоция"]
        return df_final.rename(
            columns={"frequency": "Соотношение"})


st.header(f"Выбранный фильм: {movie_title}")
if character == ' ':
    st.subheader("Персонаж не выбран")
    st.write(stat_for_genre(movie_id_))
else:
    st.subheader(f"Выбранный персонаж: {character}")
    st.write(stat_for_genre(movie_id_, character))
