import os
import progressbar
import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, epochs, train_batch, val_batch, train_data, val_data, log_dir, model_save_path, train_batch_num,
                 val_batch_num):

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_data = train_data
        self.val_data = val_data
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.train_batch_num = train_batch_num
        self.val_batch_num = val_batch_num

        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

        self.train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")
        self.val_summary_writer = tf.summary.create_file_writer(log_dir + "/val")

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="train_accuracy_top1")
        self.train_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="train_accuracy_top5")

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="val_accuracy_top1")
        self.val_accuracy_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="val_accuracy_top5")

    def lr_decay(self, epoch):
        if epoch < 60:
            return 0.1
        elif epoch < 100:
            return 0.01
        elif epoch < 150:
            return 0.001
        else:
            return 0.0001

    def train_epoch(self, curr_epoch):

        pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                    progressbar.Timer(), ", ", progressbar.Variable('LR', width=1, precision=4), ", ",
                    progressbar.Variable('Top1', width=2, precision=4), ", ", progressbar.Variable('Top5', width=2, precision=4), ", ",
                    progressbar.Variable('Loss', width=2, precision=4)]
        pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=self.train_batch_num,
                                       prefix="Epoch {}/{}: ".format(curr_epoch, self.epochs)).start()

        self.train_loss.reset_states()
        self.train_accuracy_top1.reset_states()
        self.train_accuracy_top5.reset_states()

        for batch, (images, labels) in enumerate(self.train_data):
            loss = self.train_step(images, labels)
            self.train_loss(loss)
            pbar.update(batch, LR=self.optimizer.learning_rate.numpy(), Top1=self.train_accuracy_top1.result().numpy(),
                        Top5=self.train_accuracy_top5.result().numpy(), Loss=self.train_loss.result().numpy())
        pbar.finish()

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.model(images, training=True)
            cross_entropy_loss = self.loss_object(labels, predictions)
            regularization_losses = self.model.losses
            total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))

        self.train_accuracy_top1(y_true=labels, y_pred=predictions)
        self.train_accuracy_top5(y_true=labels, y_pred=predictions)

        return total_loss

    def validate_epoch(self):
        pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                    progressbar.Timer(), ", ", progressbar.Variable('Top1', width=2, precision=4), ", ",
                    progressbar.Variable('Top5', width=2, precision=4), ", ", progressbar.Variable('Loss', width=2, precision=4)]
        pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=self.val_batch_num, prefix="Val: ").start()

        self.val_loss.reset_states()
        self.val_accuracy_top1.reset_states()
        self.val_accuracy_top5.reset_states()

        for batch, (images, labels) in enumerate(self.val_data):
            self.validate_step(images, labels)

            pbar.update(batch, Top1=self.val_accuracy_top1.result().numpy(), Top5=self.val_accuracy_top5.result().numpy(),
                        Loss=self.val_loss.result().numpy())

        pbar.finish()

    @tf.function
    def validate_step(self, images, labels):
        predictions = self.model(images, training=False)
        regularization_losses = self.model.losses

        cross_entropy_loss = self.loss_object(labels, predictions)
        total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
        self.val_loss(total_loss)
        self.val_accuracy_top1(y_true=labels, y_pred=predictions)
        self.val_accuracy_top5(y_true=labels, y_pred=predictions)

    def __call__(self, resume=False, val=False):
        best_top1 = 0
        start_epoch = 0

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, best_top1=tf.Variable(0), epoch=tf.Variable(0))
        checkpointManager = tf.train.CheckpointManager(checkpoint, directory=self.model_save_path, max_to_keep=1,
                                                       checkpoint_name="model_best.ckpt")
        if resume:
            checkpoint.restore(checkpointManager.latest_checkpoint)
            best_top1 = checkpoint.best_top1.numpy()
            start_epoch = checkpoint.epoch.numpy() + 1  # if resume, start from next epoch

        if val:
            self.validate_epoch()
            return

        for epoch in range(start_epoch, self.epochs):
            self.optimizer.learning_rate = self.lr_decay(epoch)

            self.train_epoch(epoch)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy_top1', self.train_accuracy_top1.result(), step=epoch)
                tf.summary.scalar('accuracy_top5', self.train_accuracy_top5.result(), step=epoch)

            self.validate_epoch()

            with self.val_summary_writer.as_default():
                tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                tf.summary.scalar('accuracy_top1', self.val_accuracy_top1.result(), step=epoch)
                tf.summary.scalar('accuracy_top5', self.val_accuracy_top5.result(), step=epoch)

            val_top1 = self.val_accuracy_top1.result().numpy()
            if val_top1 > best_top1:
                best_top1 = val_top1
                checkpoint.best_top1.assign(best_top1)
                checkpointManager.save()

            checkpoint.epoch.assign_add(1)


class DisTrainer(Trainer):
    def __init__(self, strategy, *args, **kwargs):
        super(DisTrainer, self).__init__(*args, **kwargs)
        self.strategy = strategy
        self.train_global_batch = self.train_batch * self.strategy.num_replicas_in_sync
        self.val_global_batch = self.val_batch * self.strategy.num_replicas_in_sync
        self.train_data = self.strategy.experimental_distribute_dataset(self.train_data)
        self.val_data = self.strategy.experimental_distribute_dataset(self.val_data)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

    def compute_loss(self, labels, predictions):
        per_example_loss = self.loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.train_global_batch)

    # distribute train step
    @tf.function
    def train_step(self, dis_images, dis_labels):
        def step_fn(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                cross_entropy_loss = self.loss_object(labels, predictions)
                regularization_losses = self.model.losses
                total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))

            self.train_accuracy_top1(y_true=labels, y_pred=predictions)
            self.train_accuracy_top5(y_true=labels, y_pred=predictions)

            return total_loss

        per_replica_losses = self.strategy.run(step_fn, args=(dis_images, dis_labels))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # distribute validate step
    @tf.function
    def validate_step(self, dis_images, dis_labels):
        def step_fn(images, labels):
            predictions = self.model(images, training=False)
            regularization_losses = self.model.losses

            cross_entropy_loss = self.loss_object(labels, predictions)
            total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
            self.val_loss(total_loss)
            self.val_accuracy_top1(y_true=labels, y_pred=predictions)
            self.val_accuracy_top5(y_true=labels, y_pred=predictions)

        return self.strategy.run(step_fn, args=(dis_images, dis_labels))
