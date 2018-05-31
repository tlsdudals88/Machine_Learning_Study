import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Training Data (X, Y), X: Input, Y: Label
X = np.array([-2, -1, 0, 1, 2])
Y = np.array([-3, -1, 1, 3, 5])

# Number of Training Examples
m = len(X) # m = 5

# Initialize Prediction Value (Hypothesis = Prediction)
pred = np.zeros(m)

# Apply various weight and bias to our model (Theta0 = b, Theta1 = w)
w = np.array([0, 1, 2, 3, 4])
b = np.array([0, 1, 2, 3, 4])

# Draw the Scatter Plot
w_position = np.zeros(len(w)*len(b))
b_position = np.zeros(len(w)*len(b))
loss_position = np.zeros(len(w)*len(b))

index = 0
for x in range(0, len(w)):
    for y in range(0, len(b)):

        # Define and Calculate Hypothesis
        for i in range(0, m):
            pred[i] = w[x]*X[i] + b[y]


        # Cost function (Mean Square Error = MSE), Loss = Cost
        total_loss = 0
        for i in range(0, m):
            total_loss = total_loss + (pred[i] - Y[i])**2

        loss = (1/(2*m)) * total_loss
        # print(total_loss)
        print("weight: "+str(x)+", "+"bias: "+str(y)+" ==> loss: "+str(loss))
        
        
        loss_position[index] = loss
        w_position[index] = x
        b_position[index] = y

        index = index + 1

fig = plt.figure()

ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title("Scatter Point")
ax1.scatter(w_position, b_position, loss_position)
ax1.set_xlabel("weigth")
ax1.set_ylabel("bias")
ax1.set_zlabel("loss")

ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title("Contour Line")
ax2.tricontour(w_position, b_position, loss_position)
ax2.set_xlabel("weigth")
ax2.set_ylabel("bias")
ax2.set_zlabel("loss")

ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title("Contour Plot")
ax3.plot_trisurf(w_position, b_position, loss_position)
ax3.set_xlabel("weigth")
ax3.set_ylabel("bias")
ax3.set_zlabel("loss")

plt.show()



