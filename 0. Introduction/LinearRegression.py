import  numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt

# 生成訓練資料
def getTrainingData():
    x = np.arange(0,50)
    y = 3 * x + np.random.uniform(-0.1,0.1,50) # y = 3x + b, b = [-0.1 , 0.1)
    # plt.plot(x,y)
    # plt.show()
    return x,y

# 進行線性迴歸
def linearRegression(input, target):

    weight = tf.Variable(1, dtype='float32')
    bias = tf.Variable(0, dtype='float32')
    hypothesis = input*weight + bias

    return hypothesis

def main():

    x_data,y_data = getTrainingData()
    X = tf.placeholder(tf.float32,name='input_x')
    Y = tf.placeholder(tf.float32,name='input_y')
    prediction = linearRegression(X,Y)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    cost = tf.square(prediction - Y)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(25):
        l = .0
        for x,y in zip(x_data,y_data):
            _, tl = sess.run([train,cost], feed_dict={X:x,Y:y})
            l += tl
        print("Epoch #{}, Cost: {} ".format(epoch+1,l))

    '''
    Epoch #1, Cost: 70914.56734465761 
    Epoch #2, Cost: 34013.763582736254 
    Epoch #3, Cost: 12012.684429526329 
    Epoch #4, Cost: 1826.7227954585105 
    Epoch #5, Cost: 31.077858204371296 
    Epoch #6, Cost: 30.839415342547 
    Epoch #7, Cost: 25.622194140858483 
    Epoch #8, Cost: 22.07525367854396 
    Epoch #9, Cost: 19.05356521817157 
    Epoch #10, Cost: 16.290240573056508 
    Epoch #11, Cost: 13.770557274343446 
    Epoch #12, Cost: 10.903501645312645 
    Epoch #13, Cost: 9.027492784982314 
    Epoch #14, Cost: 7.77702748181764 
    Epoch #15, Cost: 6.596044904668815 
    Epoch #16, Cost: 5.677346804761328 
    Epoch #17, Cost: 4.782999586270307 
    Epoch #18, Cost: 3.908879839524161 
    Epoch #19, Cost: 3.275945686342311 
    Epoch #20, Cost: 2.8124115758546395 
    Epoch #21, Cost: 2.4967117235792102 
    Epoch #22, Cost: 2.2941702912794426 
    Epoch #23, Cost: 2.1728150070412084 
    Epoch #24, Cost: 2.1049913551578356 
    Epoch #25, Cost: 2.070145544465049 
    '''

if __name__ == "__main__":
    main()