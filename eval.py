import codecs
import os
from nltk.translate.bleu_score import corpus_bleu
from data_access import get_eval_data
from transform_model import *
import tensorflow as tf


def eval():
    # Load graph
    g = attention_Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = get_eval_data()
    XX_W2I, XX_I2W, YY_W2I, YY_I2W = get_X_Y_dictional()

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            #  Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists('results'): os.mkdir('results')

            # run
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in enumerate(range(len(X) // hp.batch_size)):

                    # Get mini-batches










                    x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    # Autoregressive inference
                    # 在测试的时候是一个一个预测
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    # Write to file
                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise

                        got = " ".join(YY_I2W[str(idx)] for idx in pred)
                        fout.write("- 输入: " + source + "\n")
                        fout.write("- 应输出: " + target + "\n")
                        fout.write("- 实输出: " + got + "\n\n\n\n")
                        fout.flush()

                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                # Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()
    print("Done")
