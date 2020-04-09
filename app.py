from flask import Flask, request, render_template
from wtforms import Form, validators, SelectField, RadioField, IntegerField
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        try:
            df = pd.read_csv(request.files.get('file'))
            columns = list(df.columns)
            x = []
            columns = [tuple((i, i)) for i in columns]
        except:
            df = pd.read_csv(request.files.get('file'), encoding='cp1252')
            columns = list(df.columns)
            x = []
            columns = [tuple((i, i)) for i in columns]
        
        df.to_csv("files/data.csv", index = False)
        
        form = Data_Fill_Out(request.form)
        form.column_name.choices = columns   
        
        return render_template('choices.html', shape=df.shape, columns = columns, form = form)
    return render_template('index.html')

class Data_Fill_Out(Form):
    column_name = SelectField(u'Columns to use', [validators.DataRequired()])
    features = SelectField(u'How many features', choices=[(100,100), (500,500), (1000,1000), (2000,2000), (4000,4000), (10000,10000)], validators=[validators.DataRequired()])
    ngram = SelectField(u'NGRAMS', choices = [((1,1),(1,1)), ((1,2),(1,2)), ((2,2),(2,2)), ((1,3),(1,3)), ((3,3),(3,3))], validators=[validators.DataRequired()])
    n_topics = SelectField(u'Number of Topics', choices=[(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)], validators=[validators.DataRequired()])
    n_words = SelectField(u'Number of Output Words', choices=[(2,2), (5,5), (10,10), (20,20)], validators=[validators.DataRequired()])
    sum_ratio = SelectField(u'Percent to Summarize', choices=[(.1,"10%"), (.2,"20%"), (.3,"30%"), (.4,"40%"), (.5,"50%")], validators=[validators.DataRequired()])
    
    
@app.route('/choices')
def index():
    form = Data_Fill_Out(request.form)
    return render_template('choices.html', form = form)

@app.route('/prep_transform_predict', methods=['GET', 'POST'])
def choices():
    form = Data_Fill_Out(request.form)
    
    if request.method == 'POST':
        column_name = form.column_name.data
        ngram = eval(form.ngram.data)
        features = int(form.features.data)
        n_topics = int(form.n_topics.data)
        n_words = int(form.n_words.data)
        sum_ratio = float(form.sum_ratio.data)
        
        global sub, findall, sent_tokenize, summarize, LatentDirichletAllocation, CountVectorizer, cosine_similarity, stopwords
        pickle_loaded_list = pickle.load(open("pkl_objects/test.pkl", "rb"))
        stopwords = pickle.load(open("pkl_objects/stopwords.pkl", "rb"))
        sub, findall, sent_tokenize, summarize, LatentDirichletAllocation, CountVectorizer, cosine_similarity = pickle_loaded_list
        
        data = pd.read_csv("files/data.csv")
        
        topic_dict = {}
        for i in data["Question"].unique():
            text = data[data['Question'] == i]
            text = text[column_name].apply(lambda x: preprocessor(str(x)))
            topic = get_topics(text, features = features, ngram = ngram, n_topics = n_topics, n_words = n_words)
            topic_dict[i] = topic
        topic = pd.DataFrame(topic_dict).T
        
        summary_dict = {}
        for quest in data["Question"].unique():
            sum_data = data[(data['Question'] == quest) & (data['QuestionType'] == "Conversation")]
            question = sum_data["QuestionText"].unique()
            question_sent = sent_tokenize(" ".join(str(x) for x in question))

            x = combine_text(sum_data, column_name)
            summary = summarize(x, ratio=sum_ratio) 
            summary = too_similar(summary, question)

            summary_list = []
            for i in summary:
                if len(i) > 3:
                    summary_list.append(i)       
            summary_dict[quest] = {"Summary":summary_list}
        summary = pd.DataFrame(summary_dict).
        return render_template('prep_transform_predict.html', 
                               column_name=column_name, ngram=ngram, features=features,
                               n_topics=n_topics, n_words=n_words, sum_ratio=sum_ratio, 
                               topic_tables=[topic.to_html(classes='table table-hover table-striped table-dark', border=2, justify="center")], topic_titles=topic.columns.values,
                               summary_tables=[summary.to_html(classes='table table-hover table-striped table-dark', border=2, justify="center")], summary_titles=summary.columns.values)
    
    return render_template('choices.html', form = form)

def combine_text(data, col):
    try:
        return " ".join(["".join(i) for i in data["Response"]])
    except:
        return "HI"
    
def preprocessor(text):
    text = sub('<[^>]*>', '', text)
    text = sub("(\s\d+)","", text)
    emoticons = findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    text = " ".join([w for w in [w for w in text.split()] if w not in stopwords])
    return text

def too_similar(summary, quesiton):
    summary = sent_tokenize(summary)
    for i in summary:            
        count_vect = CountVectorizer()
        text_fit1 = count_vect.fit_transform(quesiton)
        text_fit2 = count_vect.transform([i])
        similarity = cosine_similarity(text_fit1, text_fit2)
        if similarity.any() > .75:
            summary.remove(i)
    return [text.lstrip('0123456789.- ') for text in summary]

def get_topics(text, features, ngram, n_topics, n_words):
    count = CountVectorizer(max_features = features, ngram_range = ngram)
    X = count.fit_transform(text)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=123, learning_method='batch')
    X_topics = lda.fit_transform(X)
    feature_names = count.get_feature_names()
    
    topics_lists = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics_lists["Topic: " +  str(topic_idx + 1)] = (list([feature_names[i].title() for i in topic.argsort()[:-n_words - 1:-1]]))
    return topics_lists
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)