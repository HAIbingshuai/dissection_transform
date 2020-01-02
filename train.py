from tqdm import tqdm
import tensorflow as tf

from transform_model import attention_Graph
from hyperparams import Hyperparams as hp

g = attention_Graph(is_training=True)
print("Graph loaded")
sv = tf.train.Supervisor(graph=g.graph, logdir=hp.logdir, save_model_secs=0)

with sv.managed_session() as sess:
    for epoch in range(1, hp.num_epochs + 1):
        if sv.should_stop():
            break

        for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            loss, _ = sess.run([g.mean_loss, g.train_op])
            if (step) % 100 == 0:
                print(step, ":", loss)

        gs = sess.run(g.global_step)
        sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

print("Done")
