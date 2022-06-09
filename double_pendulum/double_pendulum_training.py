"""training.py"""
import tensorflow as tf
from utils.config import CONF

@tf.function
def loss_fn(x, config:CONF, residuals=False):
    """Loss function"""
    x_star = tf.constant(config.model.x_star)
    with tf.GradientTape() as hessian:
        hessian.watch(x)
        with tf.GradientTape() as gradient:
            gradient.watch(x)
            hd_val = config.model.hd_fn(x, config.neuralnet)
        grad_hd_val = gradient.gradient(hd_val, x)
    
    #Pdef residual
    hesshd_val = hessian.batch_jacobian(grad_hd_val, x, unconnected_gradients='zero')
    # tf.print("Hessian h: ", hesshd_val)

    # xT A x check
    xTA = tf.linalg.matvec(hesshd_val, x)
    xTAx = tf.reduce_sum(tf.multiply(x, xTA), axis = 1)
    xTAx_residual = tf.reduce_mean(tf.nn.relu(config.neuralnet.epsilon-xTAx))
    pdef_residual = xTAx_residual

    #Match residual
    with tf.GradientTape() as gradient:
        gradient.watch(x)
        h_val = config.model.h_fn(x, config)  # Return ha(x), and gradx_ha(x)
    grad_h_val = gradient.gradient(h_val, x)
    # tf.print("Gradient hd: ", grad_hd_val)

    j_r = tf.tile(config.model.j, [x.shape[0], 1, 1]) - tf.tile(config.model.r, [x.shape[0], 1, 1]) #J-R
    j_r_grad_h = tf.linalg.matvec(j_r, grad_h_val) # (J-R)*gradH
    # g_perp = tf.tile(config.model.g_perp, [x.shape[0], 1])
    # g_perp_j_r_grad_h = tf.multiply(g_perp, j_r_grad_h)# g^perp*(J-R)*gradH
    # tf.print("g_perp*J-R*gradH: ", g_perp_j_r_grad_h)

    jd_rd = tf.tile(config.model.jd, [x.shape[0], 1, 1]) - tf.tile(config.model.rd, [x.shape[0], 1, 1]) #J-R
    jd_rd_grad_hd = tf.linalg.matvec(jd_rd, grad_hd_val) # (Jd-Rd)*gradHd
    # g_perp_jd_rd_grad_hd = tf.multiply(g_perp, jd_rd_grad_hd)# g^perp*(Jd-Rd)*gradHd
    # match_residual = tf.reduce_mean(tf.reduce_sum(tf.square(g_perp_jd_rd_grad_hd-g_perp_j_r_grad_h),1)) # Add over each sample and mean over all the samples
    match_residual = tf.reduce_mean(tf.square(jd_rd_grad_hd[:,0] - j_r_grad_h[:,0]))
    # tf.print("Match residual: ", match_residual)

    #Min residual
    with tf.GradientTape() as gradient:
        gradient.watch(x_star)
        val_x_star = config.model.hd_fn(x_star, config.neuralnet)
    grad_x_star = gradient.gradient(val_x_star, x_star)
    min_residual = tf.reduce_mean(tf.square(val_x_star) + tf.reduce_sum(tf.square(grad_x_star)))
    # tf.print("Min redidual: ", min_residual)

    #Positive values residual
    positive_residual = tf.reduce_mean(tf.nn.relu(-hd_val))

    # OBJECTIVE FUNCTION
    loss = (pdef_residual + (match_residual + min_residual + positive_residual))

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
    """Training function wrapper"""
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