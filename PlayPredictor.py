import numpy as np;
import pandas as pd;
from sklearn.metrics import accuracy_score;
from sklearn import tree;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn import preprocessing;

def KNNClass(classifier,colLabels,colTargets):
	data = colLabels;
	target = colTargets;

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.75);

	classifier.fit(data_train,np.ravel(target_train,order='C'));
	predictions = classifier.predict(data_test);
	print("predictions are:::",predictions)
	accuracy = accuracy_score(target_test,predictions);
	return accuracy;


def main():
	excel_file = "MarvellousInfosystems_PlayPredictor.xlsx"
	batches = pd.read_excel(excel_file);
	cleanup = {"Weather":{"Sunny":1,"Overcast":2,"Rainy":3},"Temperature":{"Hot":1,"Mild":2,"Cool":3},"Play":{"No":0,"Yes":1}}
	batches.replace(cleanup,inplace=True)
	colLabels = pd.DataFrame(batches,columns=['Weather','Temperature']);
	colTargets = pd.DataFrame(batches,columns=['Play']);
	print("target0::::",np.ravel(colTargets))
	classifier = tree.DecisionTreeClassifier();
	
	
	accuracy = KNNClass(classifier,colLabels,colTargets);
	print("Accuracy is ::",accuracy*100,"%")

	#Visualization
	from sklearn.externals.six import StringIO
	import pydot

	dot_data = StringIO();
	tree.export_graphviz(classifier,out_file=dot_data,feature_names=["Weather","Temperature"],class_names=["1","0"],filled=True
		,rounded=True,impurity=False)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph[0].write_pdf("PlayPredictor.pdf")



if __name__ == '__main__':
	main()