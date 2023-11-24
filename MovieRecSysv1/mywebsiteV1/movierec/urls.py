from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'index'),
    path('inp/RecommendMovies', views.RecommendMovies, name = 'RecomOutput'),
    path('sim/SimilarMovies', views.SimilarMovies, name = 'SimilarOutput'),
    path('inp/', views.inp, name = 'input'),
    path('sim/', views.sim, name = 'similar'),
    path('curated/', views.curated, name = 'curated'),
    path('curated/Moviesforyou', views.Moviesforyou, name = 'Moviesforyou'),
    path('dataset/', views.dataset, name = 'Dataset_dashboard')
]