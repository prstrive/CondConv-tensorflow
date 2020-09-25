import os
import datetime
import progressbar
from config import *
import tensorflow as tf
from network import get_net
from dataset import get_datasets
from trainer import Trainer, DisTrainer


class Model:
    def __init__(self, args):
        self.args = args

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.args.gpu_ids)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        if self.args.dataset == "cifar10":
            self.args.num_classes = 10
        elif self.args.dataset == "cifar100":
            self.args.num_classes = 100

        self.model_save_path = os.path.join(self.args.models_path, self.args.arch + str(self.args.num_layers))
        os.makedirs(self.model_save_path, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.args.logs_path, self.args.arch + str(self.args.num_layers), current_time)
        self.log_dir = os.path.join(self.args.logs_path, self.args.arch + str(self.args.num_layers), current_time)

    def main(self):
        if self.args.distribute:
            self.distribute_run()
        else:
            self.run()

    def run(self):
        train_date, train_batch_num, val_data, val_batch_num = get_datasets(name=self.args.dataset, train_batch=self.args.train_batch,
                                                                            val_batch=self.args.val_batch)
        model = get_net(arch=self.args.arch, num_layers=self.args.num_layers, num_experts=self.args.num_experts,
                        num_classes=self.args.num_classes)
        model.build(input_shape=(None, 32, 32, 3))
        model.summary()

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.lr, momentum=0.9, decay=0.0001, nesterov=True)

        trainer = Trainer(model=model, optimizer=optimizer, epochs=self.args.epochs, val_data=val_data,
                          train_batch=self.args.train_batch, val_batch=self.args.val_batch, train_data=train_date,
                          log_dir=self.log_dir, model_save_path=self.model_save_path, train_batch_num=train_batch_num,
                          val_batch_num=val_batch_num)

        trainer(resume=self.args.resume, val=self.args.val)

    def distribute_run(self):
        strategy = tf.distribute.MirroredStrategy()
        train_global_batch = self.args.train_batch * strategy.num_replicas_in_sync
        val_global_batch = self.args.val_batch * strategy.num_replicas_in_sync
        train_date, train_batch_num, val_data, val_batch_num = get_datasets(name=self.args.dataset, train_batch=train_global_batch,
                                                                            val_batch=val_global_batch)
        with strategy.scope():
            model = get_net(arch=self.args.arch, num_layers=self.args.num_layers, num_experts=self.args.num_experts,
                            num_classes=self.args.num_classes)
            model.build(input_shape=(None, 32, 32, 3))
            model.summary()

            optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.lr, momentum=0.9, decay=0.0001, nesterov=True)

            dis_trainer = DisTrainer(strategy=strategy, model=model, optimizer=optimizer, epochs=self.args.epochs, val_data=val_data,
                                     train_batch=self.args.train_batch, val_batch=self.args.val_batch, train_data=train_date,
                                     log_dir=self.log_dir, model_save_path=self.model_save_path, train_batch_num=train_batch_num,
                                     val_batch_num=val_batch_num)

            dis_trainer(resume=self.args.resume, val=self.args.val)
