"""training.py"""
import tensorflow as tf
from utils.config import CONF

@tf.function
def loss_fn(x, config:CONF, residuals=False):
    """Double pendulum loss function"""
    x_star = tf.constant(config.model.x_star)
    with tf.GradientTape() as hessian:
        hessian.watch(x)
        with tf.GradientTape() as gradient:
            gradient.watch(x)
            hd_val = config.model.hd_fn(x, config.neuralnet)
        grad_hd_val = gradient.gradient(hd_val, x)
    
    #Matching equation residual
    with tf.GradientTape() as gradient:
        gradient.watch(x)
        h_val = config.model.h_fn(x, config)  # Return ha(x)
    grad_h_val = gradient.gradient(h_val, x)  # Return gradx_ha(x)

    j_r = tf.tile(config.model.j, [x.shape[0], 1, 1]) - tf.tile(config.model.r, [x.shape[0], 1, 1]) # J-R
    j_r_grad_h = tf.linalg.matvec(j_r, grad_h_val) # (J-R)*gradH

    jd_rd = tf.tile(config.model.jd, [x.shape[0], 1, 1]) - tf.tile(config.model.rd, [x.shape[0], 1, 1]) #Jd-Rd
    jd_rd_grad_hd = tf.linalg.matvec(jd_rd, grad_hd_val) # (Jd-Rd)*gradHd

    match_residual = tf.reduce_mean(tf.square(jd_rd_grad_hd[:,0] - j_r_grad_h[:,0])) + tf.reduce_mean(tf.square(jd_rd_grad_hd[:,1] - j_r_grad_h[:,1])) # gperp[(Jd-Rd)*gradHd-(J-R)*gradH]=0


    #Structure residuals
    #Positive definite residual: x^T A x > 0 test
    hesshd_val = hessian.batch_jacobian(grad_hd_val, x, unconnected_gradients='zero')

    xTA = tf.linalg.matvec(hesshd_val, x)
    xTAx = tf.reduce_sum(tf.multiply(x, xTA), axis = 1)
    xTAx_residual = tf.reduce_mean(tf.nn.relu(config.neuralnet.epsilon-xTAx))
    pdef_residual = xTAx_residual

    #Minimum at x^\star residual
    with tf.GradientTape() as gradient:
        gradient.watch(x_star)
        val_x_star = config.model.hd_fn(x_star, config.neuralnet)
    grad_x_star = gradient.gradient(val_x_star, x_star)
    min_residual = tf.reduce_mean(tf.square(val_x_star) + tf.reduce_sum(tf.square(grad_x_star)))

    #Positive values residual
    positive_residual = tf.reduce_mean(tf.nn.relu(-hd_val)) # H > 0

    # OBJECTIVE FUNCTION
    loss = match_residual + (pdef_residual + min_residual + positive_residual)

    if residuals:
        tf.print('Residuals:')
        tf.print(' Pdef: ', pdef_residual,
                 '\n Positive val: ', positive_residual,
                 '\n Match: ', match_residual,
                 '\n Min: ', min_residual,
                 '\nTOTAL: ', loss)
    return loss


@tf.function
def train_step(x, config:CONF):
    """One training step"""
    with tf.GradientTape() as loss_tape:
        loss_tape.watch(x)
        loss = loss_fn(x, config)
    gradients_train = loss_tape.gradient(loss, config.neuralnet.trainable_variables)
    config.neuralnet.optimizer.apply_gradients(zip(gradients_train, config.neuralnet.trainable_variables))
    # TRACKERS
    config.neuralnet.train_tracker(loss)


@tf.function
def validation_step(x, config:CONF):
    """One validation step"""
    loss = loss_fn(x, config, residuals = config.neuralnet.print_residuals)
    # TRACKERS
    config.neuralnet.validation_tracker(loss)


def train_fn(x_train, x_test, config:CONF):
    """Training function wrapper:
    x_train: training data by batches
    x_test: validation data
    This function takes the different batches and passes them to the trainer 1by1.
    There is a validation step every %config.neuralnet.print_period epochs"""
    batch_loss = []
    t_loss, v_loss = [], []
    lastepoch = 0
    print_period = config.neuralnet.print_period

    tf.print("Residuals before training: ")
    loss_fn(x_test[0],  config, residuals = True)

    for epoch in range(1, config.neuralnet.epochs+1):
        if epoch%print_period==0:
            tf.print("\nStart epoch: ", epoch)
        
        for step, x_batch_train, in enumerate(x_train, start=1):
            #Reset metrics
            config.neuralnet.train_tracker.reset_state()
            train_step(x_batch_train, config)

            if epoch%print_period==0:
                batch_loss.append(config.neuralnet.train_tracker.result())
                tf.print("Batch {}, Loss: {:.4f}".format(step, config.neuralnet.train_tracker.result()))

        if epoch%print_period==0:
            config.neuralnet.validation_tracker.reset_state()
            validation_step(x_test[0], config)
            
            training_loss = sum(batch_loss)/len(batch_loss)
            batch_loss.clear()
            t_loss.append(training_loss.numpy())
            v_loss.append(config.neuralnet.validation_tracker.result().numpy())
            tf.print("Epoch {}, Training-Loss: {:.4f}, Validation-Loss: {:.4f}".format(epoch, training_loss, config.neuralnet.validation_tracker.result()))

        if config.neuralnet.train_tracker.result()<1e-3 and config.neuralnet.validation_tracker.result() < 1e-3:
            tf.print('STOPPED AT EPOCH: ', epoch)
            lastepoch = epoch
            print('TRAINING FINISHED WITH CONVERGENCE AT EPOCH: {}, FINAL TRAINING-LOSS: {:.4f} AND VALIDATION-LOSS: {:.4f}'.format(lastepoch, config.neuralnet.train_tracker.result(), config.neuralnet.validation_tracker.result()))
            break
        lastepoch = epoch
    if lastepoch == config.neuralnet.epochs:
        print('TRAINING FINISHED WITHOUT CONVERGENCE, FINAL TRAINING-LOSS: {:.4f} AND VALIDATION-LOSS: {:.4f}'.format(config.neuralnet.train_tracker.result(), config.neuralnet.validation_tracker.result()))
    
    tf.print("Residuals after training: ")
    loss_fn(x_test[0], config, residuals = True)
    
    return t_loss, v_loss


if __name__ == '__main__':
    pass