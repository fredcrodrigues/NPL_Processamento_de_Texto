import emoji
import nltk
import re
import sklearn
import collections
from nltk.util import ngrams
import unicodedata2 as uni
from emot.emo_unicode import EMOTICONS
from googletrans import Translator
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from num2words import num2words

# Atenção!!! - Baixar esses recursos na primeira execução - Resources from ntlk
nltk.download('punkt')  # Necessário para a opção 1
nltk.download('wordnet')  # Necessário para a opção 2
nltk.download('stopwords')  # Necessário para a opção 2 b)


# QUESTÃO 6 d)
def bigramas_e_trigramas(texto):
    # Cleaning and Tokenization
    r_urls = remover_url(texto)
    r_html = remover_html(r_urls)
    r_emojins = remover_emojis(r_html)
    r_emticons = remove_emoticons(r_emojins)
    r_tokens = tokenizar(r_emticons)
    corpus = remove_pontuacao(r_tokens)

    # bigrams
    bigrams = ngrams(corpus, 2)
    bigrams_freq = collections.Counter(bigrams)
    print("Bigramas: ", bigrams_freq.most_common())

    # trigrams
    trigrams = ngrams(corpus, 3)
    trigrams_freq = collections.Counter(trigrams)
    print("\nTrigramas: ", trigrams_freq.most_common())


# QUESTÃO 6 c)
def term_frequence(texto):
    # PRÉ PROCESSAMENTO
    r_urls = remover_url(texto)
    r_html = remover_html(r_urls)
    r_emojins = remover_emojis(r_html)
    r_emticons = remove_emoticons(r_emojins)
    corpus = nltk.sent_tokenize(r_emticons)

    r_pontuacao = remove_pontuacao(corpus)

    contagem_tokens = sklearn.feature_extraction.text.TfidfVectorizer(max_features=1000)  # tokenizar
    tfidf = contagem_tokens.fit_transform(r_pontuacao)
    palavras = contagem_tokens.get_feature_names()  # palavras correspondentes ao indice
    print("VOCABULARIO E FREQUENCIA:", contagem_tokens.vocabulary_)
    print("VETOE TF - IDF:", contagem_tokens.idf_)
    print("MATRIZ:", tfidf.toarray())


# QUESTÃO 6 b)
def words_counts(texto):
    # PRÉ PROCESSAMENTO
    r_urls = remover_url(texto)
    r_html = remover_html(r_urls)
    r_emojins = remover_emojis(r_html)
    r_emticons = remove_emoticons(r_emojins)
    corpus = nltk.sent_tokenize(r_emticons)

    r_pontuacao = remove_pontuacao(corpus)

    contagem_v = sklearn.feature_extraction.text.CountVectorizer()
    contagem_n = contagem_v.fit_transform(r_pontuacao)
    palavras = contagem_v.get_feature_names()  # PALAVRAS
    contagem_palavras = contagem_n.toarray().sum(axis=0)
    print("Ocorrencias: ", contagem_v.vocabulary_)  # palavras e sua ocorrencias
    print("Contagem: ", contagem_palavras)  # contagem palavras


# QUESTÃO 6 a)
def binary(texto):
    # dividir meu corpus em frases
    r_urls = remover_url(texto)
    r_html = remover_html(r_urls)
    r_emojins = remover_emojis(r_html)
    r_emticons = remove_emoticons(r_emojins)
    corpus = nltk.sent_tokenize(r_emticons)
    r_pontuacao = remove_pontuacao(corpus)

    contagem_tokens = sklearn.feature_extraction.text.CountVectorizer(max_features=1000)  # tokenizar
    bag_of_words = contagem_tokens.fit_transform(r_pontuacao)

    print("VETORES BINARIO: \n1 - OCORRE O TERMO\n0 - NÃO OCORRE O TERMO:\n\n", bag_of_words.toarray())


# QUESTÃO 5
def func_stemming(texto):
    stemmer = PorterStemmer()
    for x in range(len(texto)):
        p_stemmer = stemmer.stem(texto[x])
        texto[x] = p_stemmer
    return texto


# QUESTÃO 4
def func_lemmatization(texto):
    lemma = WordNetLemmatizer()
    for x in range(len(texto)):
        p_lemma = lemma.lemmatize(texto[x])
        texto[x] = p_lemma
    return texto


# QUESTÃO 3 F
def convert_emoticons(texto):
    for emot in EMOTICONS:
        lista_emoticon_b = EMOTICONS[emot]
        substitui_emo = lista_emoticon_b.replace(",", "")
        substitui_emo = substitui_emo.replace(":", "")
        dividir_emo = substitui_emo.split()
        texto = re.sub(r'(' + emot + ')', '_'.join(dividir_emo), texto)
    return texto


# QUESTÃO 3 e)
def convert_emojis(texto):
    lista_emo = []
    for x in range(len(texto)):
        if texto[x] in emoji.EMOJI_UNICODE or texto[x] in emoji.UNICODE_EMOJI:
            palavra_emoji = emoji.demojize(texto[x])
            remove_espaco = re.sub(r"_", " ", palavra_emoji)
            translator = Translator()
            translations = translator.translate(remove_espaco, dest='pt_BR')
            traduz = translations.text
            lista_emo.append(traduz)
        else:
            lista_emo.append(texto[x])
    return lista_emo


# QUESTÃO 3 d)
def spelling_correction(texto):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", texto)


# QUESTÃO 3 c)
def convert_porcento(texto):
    lista_tex = []
    for x in range(len(texto)):
        if texto[x] == '$' or texto[x] == '%':
            conv = uni.name(texto[x])
            translator = Translator()
            translations = translator.translate(conv, dest='pt_BR')
            traduz = translations.text
            lista_tex.append(traduz)
        else:
            lista_tex.append(texto[x])
    return lista_tex


# QUESTÃO 3 b)
def convert_num(texto):
    result = []
    for x in texto:
        if x.isnumeric():
            saida = num2words(int(x), lang='pt_BR')
            result.append(saida)
        else:
            result.append(x)
    return result


# QUESTÃO 3 a)
def converter_data(texto):
    print(texto)
    result = []
    for x in texto:
        if x.isnumeric():
            saida = num2words(int(x), lang='pt_BR')
            result.append(saida)
        else:
            result.append(x)

    return result


# QUESTÃO 2 k)
def spelling_correction(texto):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", texto)


# QUESTÃO 2 i)
def remover_html(texto):
    html = r'<.*?>'
    rhtml = re.sub(pattern=html, repl=' ', string=texto)
    rhtml = re.sub(r'\s+', ' ', rhtml)  # remover espaço
    return rhtml


# QUESTÃO 2 h)
def remove_emoticons(texto):
    emoticons = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')  # biblioteca
    remoticons = emoticons.sub(r'', texto)
    remoticons = re.sub(r'\s+', '', remoticons)  # remover espaço
    return remoticons


# QUESTÃO 2 g)
def remover_url(texto):
    remove_url = r'https?://\S+|www\.\S+'
    remove_links = re.sub(pattern=remove_url, repl=' ', string=texto)
    remove_links = re.sub(r'\s+', '', remove_links)  # remover espaço
    return remove_links


# QUESTÃO 2 f)
def remover_emojis(texto):
    # string = ' '.join(texto)
    remove = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        "]+", flags=re.UNICODE)
    saida = re.sub(remove, '', texto)
    return saida


# QUESTÃO 2 e)
def remover_rare_words(texto):
    tokens = tokenizar(texto)
    freq = nltk.FreqDist(tokens)
    rare_words = freq.keys()[-50:]
    filtered_sentence = [w for w in tokens if not w in rare_words]
    return filtered_sentence


# QUESTÃO 2 d)
def remover_freq(texto):
    freq = nltk.FreqDist(texto)
    for pos, val in freq.items():
        if val != 1 and val != 0:
            remover = [pos]
            resultado = [word for word in freq if not word in remover]
            return resultado


# QUESTÃO 2 c)
def remover_stopwords(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    tokens = tokenizar(texto)
    filtered_sentence = [w for w in tokens if not w in stopwords]
    print(filtered_sentence)
    return filtered_sentence


# QUESTÃO 2 b)
def remove_pontuacao(texto):
    for x in range(len(texto)):
        texto[x] = texto[x].lower()  # Deixa minusculo
        texto[x] = re.sub(r'\W', ' ', texto[x])  # remove pontuação
        texto[x] = re.sub(r'\s+', ' ', texto[x])  # substirui por espaço
    return texto


# QUESTÃO 2 a)
def minusculo(texto):
    return texto.lower()


# QUESTÃO 1 a)
def tokenizar(texto):
    tokens = nltk.word_tokenize(texto)
    return tokens


# QUESTÃO 1
def tokenizar_frase_copus(texto):
    for x in range(len(texto)):
        texto[x] = nltk.word_tokenize(texto[x])
    return texto


if __name__ == '__main__':

    arq = open('test.txt', encoding='utf-8').read()

    print("Selecione uma opção:")
    print("1 - TOKENIZAR TEXTO")
    print("2 - LIMPEZA TEXTO")
    print("3 - NORMALIZAÇÃO TEXTO")
    print("4 - LEMMATIZAÇÃO DO TEXTO")
    print("5 - STEMMING DO TEXTO")
    print("6 - BINARY - SE A PALAVRA ESTÁ NO TEXTO NÃO")
    print("7 - NUMERO DE OCORRENCIAS DE PALAVRAS N0 TEXTO")
    print("8 - TERM FREQUENCIES - TERM FREQUENCY-INVERSE DOCUMENT")
    print("9 - BIGRAMAS E TRIGRAMAS")
    opc = input()

    if int(opc) == 1:  # 1 - TOKENIZAR TEXTO"
        tokens = tokenizar(arq)
        print(tokens)
        remover_rare_words(arq)

    elif int(opc) == 2:  # 2 - LIMPEZA TEXTO
        r_urls = remover_url(arq)
        r_html = remover_html(r_urls)
        r_emojins = remover_emojis(r_html)
        r_emticons = remove_emoticons(r_emojins)
        r_tokens = tokenizar(r_emticons)
        r_pontuacao = remove_pontuacao(r_tokens)
        r_frequencia = remover_freq(r_pontuacao)
        print(r_frequencia)  # saida final

    elif int(opc) == 3:  # 3 - NORMALIZAÇÃO TEXTO
        r_urls = remover_url(arq)
        r_html = remover_html(r_urls)
        c_datas = converter_data(r_html)  # ainda não converte a data
        c_emoticons = convert_emoticons(r_html)
        tokens = tokenizar(c_emoticons)
        c_num = convert_num(tokens)
        c_emojins = convert_emojis(c_num)
        c_porcentagem = convert_porcento(c_emojins)
        print(c_porcentagem)

    elif int(opc) == 4:  # 4 - LEMMATIZAÇÃO DO TEXTO
        tokens = tokenizar(arq)
        lemmatization = func_lemmatization(tokens)
        print(lemmatization)

    elif int(opc) == 5:  # 5 - STEMMING DO TEXTO
        tokens = tokenizar(arq)
        stemming = func_stemming(tokens)
        print(stemming)

    elif int(opc) == 6:  # 6 - BINARY - SE A PALAVRA ESTÁ NO TEXTO NÃO
        binary(arq)

    elif int(opc) == 7:  # 7 - NUMERO DE OCORRENCIAS DE PALAVRAS N0 TEXTO
        words_counts(arq)

    elif int(opc) == 8:  # 8 - TERM FREQUENCIES - TERM FREQUENCY-INVERSE DOCUMENT
        term_frequence(arq)

    elif int(opc) == 9:  # 9 - BIGRAMAS E TRIGRAMAS
        bigramas_e_trigramas(arq)
