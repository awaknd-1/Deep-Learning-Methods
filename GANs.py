from matplotlib import pyplot
from numpy import hstack
from numpy.random import rand
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from numpy import zeros
from numpy import ones
from numpy.random import randn

# demonstrate x^2 function
# simple function
def calculate(x):
    return x*x

# define input
inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# caculate output
outputs = [calculate(x) for x in inputs]
# plots the results
pyplot.plot(inputs,outputs)
pyplot.show()
    

    
# define a standalone discriminator
def define_discriminator(n_inputs = 2):
    model = Sequential()
    model.add(Dense(25, activation = 'relu', kernel_initializer = 'he_uniform', input_dim = n_inputs))
    model.add(Dense(1, activation = 'sigmoid'))
    # compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics  = ['accuracy'])
    return model

# define discriminator model
model = define_discriminator()
# summarize the model
model.summary()


# genrate n real samples with class labels
def genrate_real_samples(n):
    # genrate input b/w -0.5 to 0.5
    X1 = rand(n) - 0.5
    # genrate ouput X^2
    X2 = X1*X1
    # stack array
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1, X2))
    # genrate class labels
    y = ones((n,1))
    return X, y


# genrate n fake samples with class labels  
def genrate_fake_sample(n):
    # genrate input b/w [-1, 1]
    X1 = -1 + rand(n)*2
    # genrate ouput X^2
    X2 = -1 + rand(n)*2
    # stack array
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1, X2))
    # genrate class labels
    y = zeros((n,1))
    return X, y

# train the discriminator model 
def train_discriminator(model, n_epochs = 1000, n_batch = 128):
    half_batch = int(n_batch/2)
    # run epochs manually 
    for i in range(n_epochs):
        # genrate real examples
        X_real, y_real = genrate_real_samples(half_batch)
        # update model
        model.train_on_batch(X_real,y_real)
        # update model
        X_fake, y_fake = genrate_fake_sample(half_batch)
        # update model
        model.train_on_batch(X_fake,y_fake)
        # evaluate the model
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _,acc_fake = model.evaluate(X_fake, y_fake, verbose = 0)
        print(i,acc_real, acc_fake)
        
        
# define discriminator model
model = define_discriminator()
# fit the model
train_discriminator(model)

# define a standlone generator 
def define_genrator(latent_dim, n_outputs = 2):
    model = Sequential()
    model.add(Dense(15, activation = 'relu',kernel_initializer = 'he_uniform', input_dim = latent_dim))
    model.add(Dense(n_outputs,activation = 'linear'))
    return model

# define genrator_model
model = define_genrator(5)
# summary od the model
model.summary()



# generate points in latent space as the input for the genrator
def genrator_latent_points(latent_dim,n):
    # genrate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# use the genrator to genrate n fake examples and plot the results
def genrate_fake_samples(generator, latent_dim, n):
    # genrate points in latent space
    x_input = genrator_latent_points(latent_dim, n)
    # predict output 
    X = generator.predict(x_input)
    # plot the results
    y = zeros((n,1))
    return X,y

# size of the latent space
latent_dim = 5
# define the discriminator model
model = define_genrator(latent_dim)
# genrate and plot the genrated samples
genrate_fake_samples(model,latent_dim,100)


# define a combined genrator and discriminator model, forupdating the genrator
def define_gan(genrator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add genrator
    model.add(genrator)
    # add discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return model



# plot real and fake points
def summarize_performance(epoch, genrator, latent_dim, n = 100):
    # prepare real samples
    X_real, y_real = genrate_real_samples(n)
    # evaluate performance on real data
    _,acc_real = discriminator.evaluate(X_real,y_real,verbose = 0)
    # prepare fake samples
    X_fake, y_fake = genrate_fake_samples(genrator,latent_dim, n)
    # performance on fake data
    _,acc_fake = discriminator.evaluate(X_fake,y_fake, verbose = 0)
    # summarize discriminator perfromance
    print(epoch, acc_real, acc_fake)
    # scatter plot for real and fake data
    pyplot.scatter(X_real[:,0],X_real[:,1], color = 'red')
    pyplot.scatter(X_fake[:,1],X_fake[:,1], color = 'blue')
    pyplot.show()
    
    
# train the composite model
def train(g_model,d_model,gan_model, latent_dim, n_epochs = 10000, n_batch = 128, n_eval = 2000):
    # determine half the size of one batch
    half_batch = (n_batch//2)
    # manually enymerate epochs
    for i in range(n_epochs):
        # prepare real samples
        X_real,y_real = genrate_real_samples(half_batch)
        # create fake samples
        X_fake,y_fake = genrate_fake_samples(g_model,latent_dim, half_batch)
        # update the discriminator with real samples
        d_model.train_on_batch(X_real,y_real)
        # update the discriminator model with fake samples
        d_model.train_on_batch(X_fake,y_fake)
        # genrate the latent points for input to the genrator
        x_gan = genrator_latent_points(latent_dim, n_batch)
        # create inverted labels for fake samples
        y_gan = ones((n_batch,1))
        # update the genrator via discrimintors error
        gan_model.train_on_batch(x_gan,y_gan)
        # evaluate model performance every n_val epoch
        if (i+1) % n_eval ==0:
            summarize_performance(i,g_model,d_model,latent_dim)
        


# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
#  create the genrator
genrator = define_genrator(latent_dim)
# create the gan 
gan_model = define_gan(genrator, discriminator)
# summary of the model
train(genrator, discriminator, gan_model, latent_dim)





