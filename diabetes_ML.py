import numpy as np
data=np.loadtxt(r"C:\Users\Daksh Gohil\Desktop\Daksh\Data Sets\diabetes.csv", delimiter=',')

X_train=data[0:510,0:-1].T
Y_train=data[0:510,-1:].T

X_test=data[510:,0:-1].T
Y_test=data[510:,-1:].T

#Initialise weights and bases to random values here!
b1=np.zeros((8,1))
b2=np.zeros((4,1))
b3=np.zeros((1,1))
W1=np.random.rand(8,8) *0.1
W2=np.random.rand(4,8) *0.1
W3=np.random.rand(1,4) *0.1
# print(W1,W2,W3,sep="\n\n")

# TRAINING THE NEURAL NETWORK
for epoch in range(100000):
    #FORWARD PROPAGATION
    X1=W1 @ X_train + b1
    A1=np.tanh(X1)

    X2=W2 @ A1 + b2
    A2=np.tanh(X2)

    X3=W3 @ A2 +b3
    Y_train_calculated=np.tanh(X3)
    Y_train_threshold = (Y_train_calculated >= 0.5).astype(int)

    #BACKWARD PROPAGATION
    dW3=(((Y_train_calculated-Y_train)*(1-Y_train_calculated**2)) @ X2.T)/510
    db3=np.sum((Y_train_calculated-Y_train)*(1-Y_train_calculated**2),axis=1,keepdims=True)/510

    dW2 = (((W3.T @ ((Y_train_calculated - Y_train) * (1 - Y_train_calculated**2))) * (1 - A2**2)) @ A1.T) / 510
    db2 = np.sum((W3.T @ (Y_train_calculated - Y_train) * (1 - Y_train_calculated**2)) * (1 - A2**2), axis=1, keepdims=True) / A1.shape[1]

    dW1 = (((W2.T @ ((W3.T @ ((Y_train_calculated - Y_train) * (1 - Y_train_calculated**2))) * (1 - A2**2))) * (1 - A1**2)) @ X_train.T) / 510  # (8, 8)
    db1 = np.sum(((W2.T @ ((W3.T @ ((Y_train_calculated - Y_train) * (1 - Y_train_calculated**2))) * (1 - A2**2))) * (1 - A1**2)), axis=1, keepdims=True) / X_train.shape[1]  # (8, 1)
    # print('1st time',dW1)

    # dW1=((1-A1**2)*(W2.T @ ((1-A2**2)*(W3.T@(2*(Y_train_calculated-Y_train)*(1-Y_train_calculated**2))))))@X_train.T / 510
    # print("\n2nd time",dW1)

    # GRADIENT DESCENT
    W3=W3-0.1*dW3
    W2=W2-0.1*dW2
    W1=W1-0.1*dW1
    b3=b3-0.1*db3
    b2=b2-0.1*db2
    b1=b1-0.1*db1
    # print(W3)

# TESTING THE NEURAL NETWORK
X1_test = W1 @ X_test + b1
A1_test = np.tanh(X1_test)

X2_test = W2 @ A1_test + b2
A2_test = np.tanh(X2_test)

X3_test = W3 @ A2_test + b3
Y_test_calculated = np.tanh(X3_test)

# Convert the continuous output to binary using threshold 0.5
Y_test_threshold = (Y_test_calculated >= 0.5).astype(int)

# Calculate Accuracy
accuracy = np.mean(Y_test_threshold == Y_test) * 100
print(f"Accuracy: {accuracy}%")
