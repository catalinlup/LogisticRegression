# LogisticRegression
<p>Simple Logistic Regression algorithm created in python from scratch (no Machine Learning Libraries used).</p>
<p>
<h1>Installation</h1>
Just download the script from git.<br>
Then install the additional libraries:<br>
<b>(sudo) pip install matplotlib<br>
(sudo) pip install numpy<br>
(sudo) pip install drawnow<br></b>
</p>
<p>
<h1>How to use it?</h1>
Run the script from console:<br>
python LogisticRegression.py argv1 argv2 argv3 argv4<br>
argv1 - is the training data. It should be formated like train.csv.<br>
argv2 - is the output file. The script will output 3 values(theta0, theta1 and theta2) representing the coefficients of the hypothesis:<br>
h(x)=1/(1+e^-z(x)) where z(x)=theta0*x0 + theta1*x1+theta2*x2 <br>
argv3 - is the learning rate alfa<br>
argv4 - is a threshold value.<br>
Gradient Descent stops when abs(oldCostFunctionValue-CostFunctionValue)'<'threshold. For better accuracy use a lower threshold value (ex:0.01)<br>
Command example:<br>
<b>python LogisticRegression.py train.csv model.txt 8 0.0003</b><br>
</p>
<p>
<h1>Details</h1>
For more information on the topic check out <a href='https://www.coursera.org/learn/machine-learning'>Andrew Ng's Machine Learning Course</a>.<br>
<h2>Hope you enjoy it!</h2>
</p>
