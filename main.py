from flask import Flask, render_template, request
app = Flask(__name__)

import pickle
file = open('model_pickle','rb')
clf = pickle.load(file)

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        mydict=request.form
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        bodypain = int(mydict['bodypain'])
        runnynose = int(mydict['cough'])
        diffbreathing = int(mydict['diffbreathing'])
        inputfeatures = [fever,age,bodypain,runnynose,diffbreathing]
        infprob = clf.predict_proba([inputfeatures])[0][1]
        print(infprob)
        return render_template('show.html',inf = round(infprob*100.0))
    return render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)