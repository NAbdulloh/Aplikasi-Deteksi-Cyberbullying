from django.db import models
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from django.contrib.auth.models import User


class Classifier(models.Model):
    name = models.CharField(max_length=120)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def predict(self, new_line):
        pipeline = Pipeline([
            ('count_vect', CountVectorizer()),
            ('tfidf_trans', TfidfTransformer()),
            ('multi_nb', MultinomialNB())
        ])

        x = []
        y = []

        for cat in Category.objects.filter(classifier=self):
            for line in cat.training_data.split(','):
                x.append(line)
                y.append(cat.name)

        pipeline.fit(x, y)
        return pipeline.predict([new_line])


class Category(models.Model):
    name = models.CharField(max_length=120)
    training_data = models.TextField()
    classifier = models.ForeignKey("Classifier", on_delete=models.CASCADE)

    def __str__(self):
        return self.name
