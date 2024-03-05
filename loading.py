import tensorflow as tf


class Loading:
    def __init__(self, parent_obj):
        self.parent_obj = parent_obj

    def load(self, n_classes, anchor_prior_shapes, weights_path):
        self.parent_obj.init_model(n_classes, anchor_prior_shapes)
        self.parent_obj.model.load_weights(weights_path)
