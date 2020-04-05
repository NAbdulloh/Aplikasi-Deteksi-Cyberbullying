from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Classifier, Category


class ClassifierSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Classifier
        fields = ('id', 'name', 'owner', 'category_set')


class CategorySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Category
        fields = ('id', 'name', 'training_data', 'classifier')


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'url', 'username')
