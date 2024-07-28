# app.py
import joblib
from flask import Flask, request
from flask_restx import Api, Resource, fields
from sklearn.datasets import load_iris

app = Flask(__name__)
api = Api(app, version='1.0', title='Iris Classifier API',
          description='A simple Iris classifier API')

ns = api.namespace('predict', description='Prediction operations')

model = joblib.load('iris_model.pkl')
iris = load_iris()

iris_model = api.model('IrisModel', {
    'sepal_length': fields.Float(required=True, description='Sepal length'),
    'sepal_width': fields.Float(required=True, description='Sepal width'),
    'petal_length': fields.Float(required=True, description='Petal length'),
    'petal_width': fields.Float(required=True, description='Petal width'),
})

@ns.route('/')
class IrisPrediction(Resource):
    @ns.doc('predict_iris')
    @ns.expect(iris_model)
    def post(self):
        '''Predict the class of iris flower'''
        data = request.json
        prediction = model.predict([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
        return {'prediction': iris.target_names[prediction][0]}

if __name__ == '__main__':
    app.run(debug=True)
