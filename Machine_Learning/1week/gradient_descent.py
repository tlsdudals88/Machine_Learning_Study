import numpy as np
import matplotlib.pyplot as plt

# Training Data (X, Y), X: Input, Y: Label
X = np.array([-2, -1, 0, 1, 2])
Y = np.array([-3, -1, 1, 3, 5])

# Number of Training Examples
m = len(X) # m = 5

# Initialize Prediction Value (Hypothesis = Prediction)
pred = np.zeros(m)

# Apply various weight and bias to our model (Theta0 = b, Theta1 = w)
w = -2
b = 0
LEARNING_RATE = 0.1
NUM_STEPS = 1000

def main():
    
    global w,b
    
    index = 1
    for i in range(0, NUM_STEPS):
        
        hypothesis = np.add(np.dot(w, X), b)
        
        loss = np.dot(1/(2*m), np.sum(np.square(np.subtract(hypothesis, Y))))
        print("step ("+str(index)+") => loss: "+str(format(loss, '.10f')))

        w_gradient = LEARNING_RATE * 1/m * np.sum(np.dot(np.subtract(hypothesis, Y), X))
        b_gradient = LEARNING_RATE * 1/m * np.sum(np.subtract(hypothesis, Y))
        
        w = w - w_gradient
        b = b - b_gradient
        
        if i == 100:
            h_100 = np.add(np.dot(w, X), b)
        
        if i == 500:
            h_500 = np.add(np.dot(w, X), b)

        print("weight: "+str(w))
        print("bias: "+str(b))
        print("\n")

        index = index + 1

    fig = plt.figure()
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("iter 100")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-5, 5)
    ax1.scatter(X, Y)
    ax1.plot(X, h_100)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("iter 500")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-5, 5)
    ax2.scatter(X, Y)
    ax2.plot(X, h_500)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("iter "+str(NUM_STEPS))
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-5, 5)
    ax3.scatter(X, Y)
    ax3.plot(X, hypothesis)
    
    plt.show()


if __name__ == "__main__":
    main()











