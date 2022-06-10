import streamlit as st
import pandas as pd
import re
import pymorphy2
from pyaspeller import YandexSpeller
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer


SMALL_SIZE = 6
speller = YandexSpeller()
morph = pymorphy2.MorphAnalyzer()
SMALL_SIZE = 6
plt.rc('font', size=SMALL_SIZE)



# загрузка датасета
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/strangledzelda/thesis/main/df_with_stats.csv'
    # data = pd.read_csv('C:\\Users\\dasha\\Downloads\\df_with_stats.csv')
    data = pd.read_csv(url)
    return data


# загрузка списка стоп-слов
@st.cache
def load_stopwords():
    sw = pd.read_csv('https://raw.githubusercontent.com/strangledzelda/thesis/main/all_stopwords.csv', encoding="utf-8")
    stopwords = sw.values.tolist()
    all_stopwords = []
    for word in stopwords:
        all_stopwords.append(word[0])
    return all_stopwords


def text_preprocessing(text, correction=False):
    # уберём все символы кроме кириллицы и пробелов
    text = re.sub(r'[^-:,;.А-Яа-я\s:]', '', text)

    text = re.sub(r'\n', '', text)
    # приведём к нижнему регистру
    text = text.lower()
    # убираем слова меньше 3 символов
    text = re.sub(r'\W*\b\w{1,2}\b', '', text)
    # удалим повторяющиеся подряд буквы (3 и больше)
    text = re.sub("(.)\\1{2,}", "\\1", text)

    # сделаем так, чтобы разделение на слова происходило не только через пробелы,
    # но и через дефисы
    text = re.split(' |-|:|,|;|\\.', text)
    # удалим пустые строки из списка
    text = list(filter(None, text))

    stopwords_counter = 0
    # применим лемматизатор, удалим стоп-слова, исправим опечатки
    lemlist = ['']
    for word in text:
        if correction is True:
            word = speller.spelled(word)
        else:
            word = word
        if morph.parse(word)[0].normal_form not in all_stopwords:
            lemlist.append(morph.parse(word)[0].normal_form)
        else:
            stopwords_counter += 1

    lemlist = lemlist[1:]

    # удалим повторяющиеся слова в предложении
    clean_txt = sorted(set(lemlist), key=lemlist.index)

    return ' '.join(clean_txt)


def str_to_list(text):
    return list(text.split(" "))


def piechart():
    values = [data['toxic'].value_counts()[0], data['toxic'].value_counts()[1]]
    labels = ['non-toxic', 'toxic']
    colors = sns.color_palette('bright')[2:4]
    fig = plt.figure(figsize=(3, 2))
    plt.pie(values, labels=labels, colors=colors, autopct='%.0f%%')
    st.pyplot(fig)


def plot_confusion_matrix(y_test, y_pred, ax):
    cf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')

    ax.set_title('Матрица ошибок\n\n');
    ax.set_xlabel('\nПредсказанные значения')
    ax.set_ylabel('Истинные значения ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    st.pyplot(fig)


def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    lw = 2
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='darkgreen', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('\nFalse Positive Rate')
    ax.set_ylabel('\nTrue Positive Rate')
    ax.set_title('ROC-кривая\n\n')
    ax.legend(loc="lower right")
    st.pyplot(fig)


def model_results(classifier):
    classifier.fit(X_train_cv, y_train)
    y_predicted = classifier.predict(X_test_cv)
    proba = classifier.predict_proba(X_test_cv)
    y_proba = proba[:, 1]
    matrix = confusion_matrix(y_test, y_predicted)
    acc0, acc1 = matrix.diagonal() / matrix.sum(axis=1)
    metrics = {'Общая точность': accuracy_score(y_test, y_predicted),
               'Сбалансированная точность': balanced_accuracy_score(y_test, y_predicted),
               'Точность на классе 0': acc0, 'Точность на классе 1': acc1,
               'F1-мера': f1_score(y_test, y_predicted),
               'ROC AUC': roc_auc_score(y_test, y_proba)}
    metrics_df = pd.DataFrame(metrics, index=[0])
    st.table(metrics_df.transpose())
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    draw_roc_curve(y_test, y_proba, ax1)
    plot_confusion_matrix(y_test, y_predicted, ax=ax2)


def model_results_gauss(classifier):
    classifier.fit(X_train_cv.toarray(), y_train)
    y_predicted = classifier.predict(X_test_cv.toarray())
    proba = classifier.predict_proba(X_test_cv.toarray())
    y_proba = proba[:, 1]
    matrix = confusion_matrix(y_test, y_predicted)
    acc0, acc1 = matrix.diagonal() / matrix.sum(axis=1)
    metrics = {'Общая точность': accuracy_score(y_test, y_predicted),
               'Сбалансированная точность': balanced_accuracy_score(y_test, y_predicted),
               'Точность на классе 0': acc0, 'Точность на классе 1': acc1,
               'F1-мера': f1_score(y_test, y_predicted),
               'ROC AUC': roc_auc_score(y_test, y_proba)}
    metrics_df = pd.DataFrame(metrics, index=[0])
    st.table(metrics_df.transpose())
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    draw_roc_curve(y_test, y_proba, ax1)
    plot_confusion_matrix(y_test, y_predicted, ax=ax2)

def main():
    page = st.sidebar.selectbox(
        "Выберите страницу",
        [
            "Главная",
            "Графики и статистика",
            "Word2vec",
            "Визуализация эмбеддингов",
            "Модель ML",
            "Нейронная сеть",
            "Как это работает?"
        ]
    )

    if page == "Главная":
        homepage()

    elif page == "Графики и статистика":
        graphics()

    elif page == "Word2vec":
        word2vec()

    elif page == "Модель ML":
        ml_model()

    elif page == "Нейронная сеть":
        neural_network()

    elif page == "Как это работает?":
        description()

    elif page == "Визуализация векторных представлений":
        print(1)
        # embedding_viz()



def homepage():
    st.title('Классификация текстовых сообщений с форумов 2ch и Pikabu')
    # st.dataframe(data)
    st.subheader('Взглянем на данные')
    row_num = st.number_input('Выберите строку датасета', 0, 14411, 4562)
    st.write(data.iloc[row_num, 0])
    own_txt = st.checkbox('Хотите ввести свой текст?')
    if own_txt:
        text = st.text_area("Текст для анализа:")
    else:
        text = data.iloc[row_num, 0]
    preprocess_type = st.radio('Выберите тип предобработки:', ['обычная', 'с коррекцией ошибок'])
    if preprocess_type == 'обычная':
        correct = False
    else:
        correct = True
    if st.button('Нажмите, чтобы увидеть предобработанный текст'):
        st.write(text_preprocessing(text, correct))
    if st.button('Анализировать комментарий'):
        mdl = MultinomialNB(alpha=0.8).fit(X_train_cv, y_train)
        prediction = mdl.predict(cv.transform([text_preprocessing(text, correct)]))
        st.subheader(f'Ответ модели - {int(prediction[0])}.')
        prob = mdl.predict_proba(cv.transform([text_preprocessing(text, correct)]))
        st.subheader(f'Степень уверенности - {round(np.max(prob) * 100, 1)}%.')
    # выведем ещё степень уверенности..?


def graphics():
    st.header("Графики и статистические данные")
    st.subheader('Облако самых частых слов *до* удаления стоп-слов:')
    st.image('https://github.com/strangledzelda/thesis/blob/main/wordcloud_before.png')
    st.subheader('Облако самых частых слов *после* удаления стоп-слов:')
    st.image('https://github.com/strangledzelda/thesis/blob/main/wordcloud_after.png')
    st.subheader('Распределение по классам:')
    st.image('https://github.com/strangledzelda/thesis/blob/main/pie.png')
    # piechart()
    st.subheader('Статистические данные')
    st.write('Рассмотрим следующие искусственно созданные статистические показатели корпуса текстов:\n\n* число '
             'символов в одном '
             'комментарии, \n\n* число слов в каждом комментарии,\n\n* число стоп-слов в комментарии. \n\nВ среднем '
             'каждый комментарий состоит из ~175.5 символов, ~27 слов, содержит 13 стоп-слов. В квантиль q=0.95 '
             'входят комментарии с числом слов до 62 и числом символов до 527. Эту информацию мы сможем использовать, '
             'когда будем определять параметры для максимальной длины последовательностей, подающихся на входной слой '
             'нейронной сети.')
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), ax=ax, cmap="Blues", annot=True)
    st.write('Как видно из тепловой карты корреляции, ни одно из искусственно созданных значений не коррелирует хоть '
             'сколько-нибудь значительно с целевым признаком toxic.')
    st.pyplot(fig)


def neural_network():
    st.header('Нейронные сети')
    models = ['Полносвязная', 'Свёрточная', 'Рекуррентная']
    model_choice = st.selectbox('Выберите архитектуру НС', models)
    if model_choice == 'Полносвязная':
        st.subheader('Полносвязная НС')
        st.write('В полносвязной нейронной сети прямого распространения каждый нейрон связан со всеми остальными '
                 'нейронами, находящимися в соседних слоях. Поскольку стоит задача бинарной классификации, '
                 'на выходном слое достаточно одного нейрона с сигмоидальной функцией.')
        st.image('https://github.com/strangledzelda/thesis/blob/main/dense.png')
        st.latex(r'''f(x)=\frac{1}{1+e^{-x}}''')
        if st.checkbox('1 скрытый слой, 16 нейронов, текст в формате OHE'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-1-16.png')
            st.write('Точность на тестовой выборке - 55%')
        if st.checkbox('1 скрытый слой, 16 нейронов, текст после Embedding-слоя'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-1-16-emb.png')
            st.write('Точность на тестовой выборке - 83%')
            st.write('Очевидно, модель переобучилась. После добавления слоя Dropout точность повышается на 2-3% (до '
                     '83-84%).')
        if st.checkbox('2 скрытых слоя, 16 + 8 нейронов, Embedding + Dropout (0.5)'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-2-16-8.png')
            st.write('Точность на тестовой выборке - 83%')
        if st.checkbox('2 скрытых слоя, 64 + 32 нейронов, Embedding + Dropout (0.5)'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-2-64-32.png')
            st.write('Точность на тестовой выборке - 82%')
        if st.checkbox('3 скрытых слоя, 32 + 16 + 8 нейронов, Embedding + Dropout (0.5)'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-3-32-16-8.png')
            st.write('Точность на тестовой выборке - 83%')
        if st.checkbox('3 скрытых слоя, 128 + 64 + 32 нейронов, Embedding + Dropout (0.5)'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-3-128-64-32.png')
            st.write('Точность на тестовой выборке - 83%')
        if st.checkbox('4 скрытых слоя, 128 + 64 + 32 + 16 нейронов, Embedding + Dropout (0.5)'):
            st.image('https://github.com/strangledzelda/thesis/blob/main/dense-4-128-64-32-16.png')
            st.write('Точность на тестовой выборке - 83%')
    if model_choice == 'Свёрточная':
        st.write('Как правило, сверточную (двуслойную) архитектуру нейронных сетей используют для работы с '
                 'изображениями, однако однослойные нейронные сети также широко применяются для задачи классификации '
                 'текста. Свёрточные нейронные сети работают по принципу рецептивных полей. В отличие от полносвязных '
                 'нейронных сетей, в свёрточных нет связей между всеми нейронами соседних слоёв, каждый нейрон '
                 '«смотрит» на небольшой кусочек предыдущего слоя и обобщает информацию с него для передачи дальше. ')
        st.write('Обычно используется не классический MaxPooling, MaxOverTimePooling / GlobalMaxPooling - MaxPooling, '
                 'применённый ко всей последовательности сразу, то есть ширина его окна равна ширине всей матрицы.')
        st.image('https://github.com/strangledzelda/thesis/blob/main/cnn-for-text.png')
        st.write('\n\n\n\n')
        st.write('\n\n **1 свёрточный слой, 64 нейрона, ядро свёртки = 3**')
        st.image('https://github.com/strangledzelda/thesis/blob/main/cnn-64.png')
        st.write('Точность на тестовой выборке - 87%')
        st.write('**2 свёрточных слоя 62 + 32, ядро свёртки = 3**')
        st.image('https://github.com/strangledzelda/thesis/blob/main/cnn-32-16.png')
        st.write('Точность на тестовой выборке - 82%')
    if model_choice == 'Рекуррентная':
        st.write('Рекуррентные сети (RNN) используются для лингвистических задач, когда нужно «помнить» '
                 'синтакcическую структуру предложения. В RNN разрешены циклы, выход нейрона может быть соединён со '
                 'входом. ')
        st.image('https://github.com/strangledzelda/thesis/blob/main/rnn.png')
        st.write('1 слой, 12 нейронов')
        st.image('https://github.com/strangledzelda/thesis/blob/main/rnn-1-12.png')
        st.write('Точность на тестовой выборке - 74%')


def ml_model():
    st.header('Модели машинного обучения')
    models = ['Полиномиальная', 'Бернулли', 'Гаусса']
    model_choice = st.selectbox('Выберите модель алгоритма наивного Байеса', models)
    if model_choice == 'Полиномиальная':
        st.subheader('Наивный Байес (полиномиальная модель)')
        alpha = st.slider('alpha', 0.1, 0.9, 0.8, 0.1, '%f')
        model_results(MultinomialNB(alpha=alpha))
    if model_choice == 'Бернулли':
        st.subheader('Наивный Байес (модель Бернулли)')
        alpha = st.slider('alpha_bernoulli', 0.1, 0.9, 0.8, 0.1, '%f')
        model_results(BernoulliNB(alpha=alpha))
    if model_choice == 'Гаусса':
        st.subheader('Наивный Байес (модель Гаусса)')
        model_results_gauss(GaussianNB())

def word2vec():
    st.header('Word2vec')
    st.write('word2vec - способ построения сжатого пространства векторов слов, использующий нейронные сети. Векторное '
             'представление слова основывается на контекстной близости: если слова встречаются рядом с одинаковыми '
             'словами и, следовательно, имеют высокую степень синонимичности, в векторном представлении обладают '
             'высоким косинусным сходством (*cosine similarity*).')
    st.write('Формально задачу можно представить так: необходимо составить такие векторные представления, '
             'чтобы косинусная близость слов, появляющихся в похожих контекстах, была максимальна, а не появляющихся '
             '- минимальной.')
    st.latex(r'''similarity(A,B)=\frac{{A}\cdot{B}}{|A|*|B|}''')
    st.write('Где А, В - векторы признаков, A∙B - их скалярное произведение, |A| - длина вектора.')
    st.write('Косинусная мера близости располагается в диапазоне  [0;1], поскольку частота встречаемости слова не '
             'может быть отрицательной; угол между векторами частоты, таким образом, лежит в пределах от 0° до 90°.')
    st.write('Нейронная сеть Word2Vec имеет две реализации: **Skip-gram** и **CBOW** (Continuous bag-of-words). В '
             'модель слова подаются в формате one-hot encoding. На выходном слое с N нейронами (где N - размер '
             'словаря) используется слой softmax или его вариации, так что на выходе получается распределение '
             'вероятности каждого слова.')
    st.write('У нас нет размеченных данных, но нам нужно обучить модель, поэтому мы искусственно создаём задание '
             'предсказание слова, побочным эффектом которого будет векторное представление слов. Вся разница между '
             'алгоритмами skip-gram и CBOW - в этом искусственном задании.')
    st.write('skip-граммам достаточно небольшого корпуса слов, тогда как алгоритм CBOW хорошо обучается, когда объём '
             'корпуса превышает 100млн слов.')
    if st.checkbox('Показать топологию нейронной сети для модели skip-gram.'):
        st.write('Skip-gram получает на вход одно слово и предсказывает подходящий контекст. Контекст - это ближайшие '
                 'слова, образованные в зависимости от размера контекстного окна.')
        st.image('https://github.com/strangledzelda/thesis/blob/main/skipgram1.png')
    if st.checkbox('Показать топологию нейронной сети для модели CBOW'):
        st.write('CBOW - обычная модель мешка слов с учётом ближайших соседей (контекста). CBOW пытается угадать '
                 'слово исходя из окружающего контекста. Интуитивно понятно, что задача CBOW '
                 'намного проще, и действительно, алгоритм сходится быстрее, чем skip-граммы.')

        st.image('https://github.com/strangledzelda/thesis/blob/main/cbow1.png')
    st.write('В нейронных сетях плотное векторное представление слов определяется в процессе обучения. На первом '
             'этапе элементы вектором инициализируются случайными числами, изменений значений векторов происходит '
             'итерационно с помощью метода обратного распространения ошибки. ')
    st.write('Для ускорения обучения моделей используются модификации softmax (например, иерархический softmax или '
             'негативный сэмплинг), позволяющие вычислять распределение вероятностей быстрее, чем за линейное время '
             'от размера словаря. ')
    if st.checkbox('Совсем подробно'):
        st.write('* читается корпус, рассчитывается встречаемость каждого слова в корпусе,\n\n * массив слов '
                 'сортируется по частоте (слова сохраняются в хэш-таблице), редкие слова удаляются,\n\n * cтроится '
                 'дерево Хаффмана,\n\n * проходим окном заданного размера по предложению.')
    w2v = gensim.models.Word2Vec.load('https://github.com/strangledzelda/thesis/blob/main/word2vec.model')
    word = st.text_input("Введите слово (безопасные: родитель, учитель, школа ребёнок, мама, Москва, дорогой, город, расход, заработок)")
    if word != '':
        similar = w2v.wv.most_similar(positive=[word])
        st.write('Наиболее близкие слова:')
        i=1
        for el in similar:
            st.write(f'{i}) {el[0]}, {round(el[1],4)}')
            i +=1
    if word != '':
        st.write('Векторное представление:')
        st.write(w2v.wv[word])


def description():
    st.header("Как это работает?")
    st.subheader('1. Предобработка текстовых данных.')
    st.write('Сначала текстовые данные необходимо предобработать. В предобработку входит: \n* приведение к нижнему '
             'регистру, \n* удаление цифр, специальных символов и латиницы, \n* удаление повторяющихся букв (> 3) и '
             'слишком коротких слов (< 3), \n* удаление стоп-слов, \n* исправление ошибок и опечаток, '
             '\n* лемматизация/стемминг.')

    st.write('Все перечисленные выше операции нужны для уменьшения словаря и сокращения вычислительной нагрузки.')
    if st.checkbox('Подробнее про коррекцию ошибок'):
        st.write('Для исправления ошибок и нахождения ближайшего похожего слова используется редакционное расстояние, '
                 'или расстояние Левенштейна, которое вычисляется как минимальное количество операций вставки, '
                 'удаления, замены одного символа, необходимых для превращения одной строки в другую. \n\n '
                 'Яндекс.Спеллер для обнаружения ошибок и подбора замены использует библиотеку CatBoost. В '
                 'документации разработчики подчёркивают, что Спеллер не придирается к новым словам, ещё не попавшим '
                 'в словари, и умеет учитывать контекст.')
        word_for_correction = st.text_input('Введите текст с ошибкой')
        if st.button('Исправить с помощью Яндекс.Спеллера'):
            st.write(speller.spelled(word_for_correction))
    if st.checkbox('Подробнее про лемматизацию'):
        st.write('Лемматизация - это приведение слова к его нормальной морфологической (словарной) форме - лемме. '
                 '\n\n Я использую библиотеку pymorphy2, она обучена на корпусе OpenCorpora и определяет по написанию '
                 'слова его морфологические характеристики (часть речи, род, число, падеж и т.д.)')
        word_for_lemmatization = st.text_input('Введите слово для приведения к нормальной форме')
        if st.button('Привести к нормальной форме'):
            st.write(morph.parse(word_for_lemmatization)[0].normal_form)
    if st.checkbox('Подробнее про стемминг'):
        st.write('Стемминг отсекает от слова окончания и суффиксы так, чтобы оставшаяся часть (корень, *stem*) была '
                 'одинакова для всех грамматических форм слова. \n\nК сожалению, как правило, при обработке '
                 'русскоязычных текстов этот подход не даёт слишком хороших результатов за счёт таких феноменов как '
                 'беглые гласные корня. ')
        word_for_stemming = st.text_input('Введите слово')
        snow_stemmer = SnowballStemmer(language='russian')
        if st.button('Отрезать основу'):
            st.write(snow_stemmer.stem(word_for_stemming))
    st.subheader('2. Представление текста в числовом виде (векторизация).')
    st.write('Поскольку алгоритмы машинного обучения не могут работать напрямую с необработанным текстом, '
             'его необходимо преобразовать в векторы. Цель этого этапа - сделать так, чтобы числовые векторы как '
             'можно лучше отражали лингвистические свойства текста.')
    st.write('К простым видам векторизации относятся методы, основанные на характеристике частотности: \n* Прямое '
             'кодирование (One-Hot Encoding), \n* Bag-of-Words (CountVectorizer, TF-IDF).')
    st.write('Все перечисленные методы векторизации текста имеют общую проблему: они не учитывают контекст и порядок '
             'слов в документах. С этим успешно справляется метод представления текста в виде плотных векторов (*word '
             'embeddings*). Подробнее про них - во вкладке word2vec.')
    if st.checkbox('Больше про методы векторного представления текстов'):
        sent1 = st.text_input('Введите первое предложение')
        sent2 = st.text_input('Введите второе предложение')
        samples = {sent1, sent2}
        token_index = {}
        counter = 0
        for sample in samples:
            for considered_word in sample.split():
                if considered_word not in token_index:
                    token_index.update({considered_word: counter + 1})
                    counter = counter + 1
        if st.button('Словарь'):
            st.write(token_index)
        if st.checkbox('Прямое кодирование (OHE)'):
            st.write('Каждый токен - бинарное значение, если токен есть - 1, если нет - 0.')
            max_length = 6
            if sent1 != '' and sent2 != '':
                results = np.zeros(shape=(len(samples),
                                          max_length,
                                          max(token_index.values()) + 1))
                for i, sample in enumerate(samples):
                    for j, considered_word in list(enumerate(sample.split())):
                        index = token_index.get(considered_word)
                        results[i, j, index] = 1.

                st.write(results)
            else:
                st.error('Заполните поля с тестовыми предложениями')
            st.write(
                '**Очевидная проблема такого метода - размерность.** Получаются разреженные матрицы с огромным числом '
                'нулей, к тому же чем больше словарь, тем больше будет матрица.')
        if st.checkbox('Классический мешок слов'):
            st.write('Bag-of-Words принимает во внимание только две параметра: **словарь** (список уникальных слов в '
                     'корпусе) и **меру присутствия** таких слов в тексте.')
            if sent1 != '' and sent2 != '':
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform([sent1, sent2])
                st.dataframe(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))
            else:
                st.error('Заполните поля с тестовыми предложениями')
            st.write('Такой подход решает проблему размерности по одной оси, однако не учитывает важность '
                     'каждого токена, к примеру, повторяющегося несколько раз в разных документах.')
        if st.checkbox('TF-IDF'):
            st.write('Проблема с оценкой частоты слов в классической модели мешка слов заключается в том, что часто '
                     'встречающиеся слова не несут столько семантической нагрузки, сколько более редкие, '
                     'специфические для предметной области выражения. \n\nПодход TF-IDF заключается в том, '
                     'что чем чаще в документах встречается слово, тем больше его "штрафуют".')
            st.latex(r'''w_{t,d} = tf(t,d) * idf(t)''')
            st.latex(r'''tf(t,d)=\frac{n_{t}}{\sum_{k}n_{k}}''')
            st.latex(r'''idf(t)=log(\frac{D}{df(t)})''')
            st.write(
                'Где t (*term*) - слово, d - документ, *tf* (*term frequency*) - вероятность найти слово в документе - отношение '
                'числа вхождений слова *t* в документ *d* к общему числу слов документа. \n\nIDF (*inverse '
                'document frequency*) оценивает, насколько редко слово встречается в документе, и считается как '
                'логарифм отношения общего числа документов *D* на число документов, в которых встречается '
                'термин *t*. ')

            if sent1 != '' and sent2 != '':
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform([sent1, sent2])
                st.dataframe(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))
            else:
                st.error('Заполните поля с тестовыми предложениями')
    st.subheader('3. Обучение модели, предсказание.')
    st.write('Весь набор данных разделяется на **обучающую** и **тестовую** выборки. После того, как выбранная модель '
             'обучилась, с помощью неё предсказывают метки классов для тестового набора данных и сравнивают их с '
             'истинными метками для оценки качества работы модели.')
    if st.checkbox('Подробнее про алгоритм наивного Байеса'):
        st.write('Наивный байесовский классификатор основан на применении теоремы Байеса, используемой для вычисления '
                 'условных вероятностей. Условная вероятность показывает вероятность наступления одного события при '
                 'условии, что другое событие уже произошло.')
        st.latex(r'''P(A|B)=\frac{P(B|A)*P(A)}{P(B)}''')
        st.write('Где P(A|B) - апостериорная оценка вероятности гипотезы А при условии наступления события В, '
                 '\n\nP(A) - априорная вероятность гипотезы А, \n\nP(B|A) - вероятностнь наступления события В при '
                 'истинности гипотезы А, \n\nP(B) - вероятность наступления события В.')
        st.write('Байесовский подход к вероятности предполагает, что вероятность оценивается как степень уверенности. '
                 'Основная идея теоремы Байеса состоит в том, что можно более точно оценить вероятность события путём '
                 'учёта дополнительных данных. Теорема позволяет по известному факту произошедшего события вычислить '
                 'вероятность того, что оно было вызвано данной причиной.')
        st.write("Вероятностная модель для классификатора с n набором признаков и целевым y:")
        st.latex(r'''P(y, x_{1},...x_{n})=\frac{P(x_{1},...x_{n}|y)*P(y)}{P(x_{1},...x_{n})}''')
        st.write("Алгоритм носит такое название, поскольку предполагает 'наивные' допущения: каждая характеристика "
                 "вносит **независимый** и **равный** вклад в результат. \n\nПоскольку мы исходим из независимости "
                 "признаков, верно выражение:")
        st.latex(r'''P(x_{i}|y,x_1,...,x_{i-1},x_{i+1},...,x_n)=P(x_i|y)''')
        st.latex(r'''P(y|x_{1},...x_{n})=\frac{P(y)\prod_{i=1}^{n}{P(x_i|y)}}{P(x_{1},...x_{n})}''')
        st.write('Тогда наивный байесковский классификатор сводится к:')
        st.latex(r'''\^{y}=argmax(P(y)*\prod_{i=1}^{n}{P(x_i|y}))''')
        st.write('Существуют три типа наивного байесовского классификатора: \n\n* **полиномиальный**: векторы '
                 'признаков представляют собой частоты, с которыми события генерируются полиномиальной моделью')
        st.latex(r'''(p_1,...,p_n)''')
        st.write('Где p_i - вероятность события i')
        st.latex(r'''p(x|y)=\frac{(\sum_{i=1}^{n}{x_i})!}{\prod_{i=1}^{n}{x_i!}}\prod_{i=1}^{n}{p_{k_i}^{x_i}}''')
        st.write('В признак *x_i* записывается частота вхождения слова в каждый образец данных (документ).')
        st.write('Обычно такая модель событий используется для классификации документов.')
        st.write('* **Бернулли**: характеристики входных данных описываются логическими '
                 '(двоичными) значениями; модель применяется в тех задачах классификации документов, где используется '
                 'не частота встречаемости слова в документе, а бинарные характеристики встречаемости каждого слова *x_i*('
                 '0 – слово не встретилось, 1 – встретилось).')
        st.latex(r'''p(x|y)=\prod_{i=1}^{n}{p_{k_i}^{x_i}}(1-p_{k_i})^{(1-x_i)}''')
        st.write('* **Гаусса**: непрерывные значения характеристики имеют распределение Гаусса. ')
        st.latex(r'''P(x=v|y)=\frac{1}{\sqrt{2\pi\sigma_{y}^2}}e[-\frac{(v-\mu_y)^2}{2\sigma_y^2}]''')
        st.write('Если во время выполнения расчётов алгоритму встретится новое слово, которого не было на этапе '
                 'обучения системы, это приведёт к тому, что оценка будет равна нулю и документ нельзя будет отнести '
                 'ни к одному классу. Нельзя обучить модель всем возможным словам, но можно применить **сглаживание '
                 'Лапласа** (аддитивное сглаживание) с параметром альфа. \n\nИдея заключается в том, что мы решаем '
                 'проблему нулевой вероятности прибавлением единицы к частоте каждого слова.')
    st.subheader('4. Оценка качества работы модели.')
    st.write('Качество предсказаний алгоритма оценивается путём сравнения истинных меток с ответами модели.')
    st.image('https://github.com/strangledzelda/thesis/blob/main/matrix.png')
    st.write('Для оценки я использовала следующие метрики:')
    st.write(
        '* **Общая точность** (*accuracy*) показывает долю объектов, для которых алгоритм выдал правильные ответы. '
        'Не отражает реальную ситуацию, если алгоритм отлично научился распознавать один класс, но с другим '
        'справляется плохо.')
    st.latex(r'''accuracy=\frac{TP+TN}{TP+TN+FP+FN}''')
    st.write('* **Сбалансированная точность** и точность на каждом классе.')
    st.latex(r'''accuracy_{balanced} = 0.5 * (\frac{TP}{TP+FN}+\frac{TN}{TN+FP})''')
    st.write('* **F1-мера** - среднее гармоническое метрик recall (полнота) и precision (точность)')
    st.latex(r'''F1=2*\frac{precision*recall}{precision*recall}''')
    st.write('*Recall* показывает способность алгоритма обнаруживать положительный класс, т.е. какую долю объектов '
             'положительного класса из всех объектов положительного класса мы правильно классифицировали.')
    st.latex(r'''recall=\frac{TP}{TP+FN}''')
    st.write('*Precision* показывает способность алгоритма отличать положительный класс от других классов, т.е. долю '
             'объектов, определённых классификатором как положительные и на самом деле являющихся положительными.')
    st.latex(r'''precision=\frac{TP}{TP+FP}''')
    st.write('* **ROC AUC**, или площадь под кривой ошибок.')
    st.write('ROC-кривая показывает зависимость чувствительности (=recall, TPR) модели от её специфичности. Под '
             '**специфичностью** понимается доля истинно отрицательных наблюдений в общем числе отрицательных (TNR).')
    st.latex(r'''specificity=\frac{TN}{TN+FP}''')
    st.write('По вертикальной оси графика ROC-кривой располагается чувствительность, а по горизонтальной – False '
             'Positive Rate')
    st.latex(r'''FPR=1-specificity=\frac{FP}{FP+TN}''')
    st.write('С помощью ROC-кривой можно сравнить, как меняется чувствительность модели на разных порогах отсечения, '
             'модель опирается на порог отсечения, чтобы принимать решения и относить объекты к положительному '
             'классу. ROC-кривая отражает связь между вероятностью «ложной тревоги» и вероятностью верного '
             'обнаружения, с ростом чувствительности растёт надёжность распознавания положительных наблюдений, '
             'но при этом растёт и вероятность ложной тревоги. \n\nROC AUC (area under curve) – площадь под кривой '
             'ошибок. Чем ближе к единице, тем лучше.')


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


data = load_data()
X = data.clean_comment
y = data.toxic
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
cv = CountVectorizer(ngram_range=(1, 1))
tfidf = TfidfVectorizer()
tfidf.fit(X_train.values.astype('U'))
X_train_cv = cv.fit_transform(X_train.values.astype('U'))
X_test_cv = cv.transform(X_test.values.astype('U'))
all_stopwords = load_stopwords()
X_for_w2v = data_for_w2v.processed
main()
