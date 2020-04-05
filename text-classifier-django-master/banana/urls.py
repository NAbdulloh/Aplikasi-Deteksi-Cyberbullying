from django.conf.urls import include, url
from django.contrib import admin
from rest_framework import routers
from bananasorter import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'classifiers', views.ClassifierViewSet)
router.register(r'categories', views.CategoryViewSet)


urlpatterns = [
    url(r'^', include('bananasorter.urls')),
    url(r'^', include('django.contrib.auth.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^api/', include(router.urls)),
]
