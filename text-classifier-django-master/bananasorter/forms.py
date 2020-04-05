from django import forms
from bananasorter.models import Classifier, Category


class ClassifierForm(forms.ModelForm):
    class Meta:
        model = Classifier
        fields = ['name']


class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'training_data']
