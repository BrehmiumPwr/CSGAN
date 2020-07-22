from datacache import DataCache

class SingleStepUpdate(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.do_update()

    def do_update(self):
        gen_loss, disc_loss, cur_step, lr, _, _ = self.model.session.run([self.model.g_loss,
                                                                          self.model.d_loss,
                                                                          self.model.global_step,
                                                                          self.model.learning_rate,
                                                                          self.model.g_optim,
                                                                          self.model.d_optim],
                                                                         feed_dict={self.model.istraining: True})
        return gen_loss, disc_loss, cur_step, lr


class DualStepUpdate(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.do_update()

    def do_update(self):
        disc_loss, _ = self.model.session.run([self.model.d_loss,
                                               self.model.d_optim],
                                              feed_dict={
                                                  self.model.istraining: True})  # , options=run_options,run_metadata = run_metadata)

        # summary_writer.add_run_metadata(run_metadata, "mySess")
        # run generator update
        gen_loss, cur_step, _, lr = self.model.session.run([self.model.g_loss,
                                                            self.model.global_step,
                                                            self.model.g_optim,
                                                            self.model.learning_rate],
                                                           feed_dict={self.model.istraining: True})
        return gen_loss, disc_loss, cur_step, lr


class DualStepWithDataCacheUpdate(object):
    def __init__(self, model):
        self.model = model
        self.datacache = DataCache(size=16)

    def __call__(self, *args, **kwargs):
        return self.do_update()

    def do_update(self):
        # summary_writer.add_run_metadata(run_metadata, "mySess")
        # run generator update
        gen_loss, cur_step, _, lr, images, conditions = self.model.session.run([self.model.g_loss,
                                                                                self.model.global_step,
                                                                                self.model.g_optim,
                                                                                self.model.learning_rate,
                                                                                self.model.fake_images,
                                                                                self.model.g_fake_color_batch],
                                                                               feed_dict={self.model.istraining: True})

        self.datacache.fill(datapoint=(images, conditions))
        images, conditions = self.datacache.get_data()

        disc_loss, _ = self.model.session.run([self.model.d_loss,
                                               self.model.d_optim],
                                              feed_dict={self.model.istraining: True,
                                                         self.model.fake_images: images,
                                                         self.model.g_fake_color_batch: conditions})

        return gen_loss, disc_loss, cur_step, lr
