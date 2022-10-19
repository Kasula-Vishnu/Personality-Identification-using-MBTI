from csv import writer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pickle
import re
import nltk
import pandas as pd
from numpy import vectorize
nltk.download('punkt')
nltk.download('wordnet')


'''model_XGB = pickle.load(open('model_XGB.pkl', 'rb'))'''
model_logreg = pickle.load(open('model_logreg.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


def input_preprocesing(text):
    filter = []
    review = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(review)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))

    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/predict', methods=['POST'])
def home():
    data_1 = request.form['prediction_text1']
    data_2 = request.form['prediction_text2']
    data_3 = request.form['prediction_text3']
    data_4 = request.form['prediction_text4']
    data = " ".join([data_1, data_2, data_3, data_4])
    preprocessed_data = input_preprocesing(data)
    vectorized_data = vectorizer.transform(preprocessed_data)
    prediction = model_logreg.predict(vectorized_data)[0]
    if prediction == 0:
        personality = 'ENFJ'
    elif prediction == 1:
        personality = 'ENFP'
    elif prediction == 2:
        personality = 'ENTJ'
    elif prediction == 3:
        personality = 'ENTP'
    elif prediction == 4:
        personality = 'ESFJ'
    elif prediction == 5:
        personality = 'ESFP'
    elif prediction == 6:
        personality = 'ESTJ'
    elif prediction == 7:
        personality = 'ESTP'
    elif prediction == 8:
        personality = 'INFJ'
    elif prediction == 9:
        personality = 'INFP'
    elif prediction == 10:
        personality = 'INTJ'
    elif prediction == 11:
        personality = 'INTP'
    elif prediction == 12:
        personality = 'ISFJ'
    elif prediction == 13:
        personality = 'ISFP'
    elif prediction == 14:
        personality = 'ISTJ'
    elif prediction == 15:
        personality = 'ISTP'
    print(personality)

    new_data = [personality, data]

    with open('mbti_1.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_data)
        f_object.close()
    
    return render_template('something.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
