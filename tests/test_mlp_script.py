import runpy
from pathlib import Path
import sys
import types
import numpy as np


def fake_load_data():
    x_train = np.random.rand(10, 28, 28)
    y_train = np.random.randint(0, 10, 10)
    x_test = np.random.rand(2, 28, 28)
    y_test = np.random.randint(0, 10, 2)
    return (x_train, y_train), (x_test, y_test)


class DummySequential:
    def __init__(self, *args, **kwargs):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def summary(self):
        pass

    def compile(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        return (0.0, 1.0)

    @property
    def output_shape(self):
        return (None, 10)


class DummyModule(types.ModuleType):
    pass


def setup_fake_tensorflow(monkeypatch):
    tf = DummyModule('tensorflow')
    keras = DummyModule('tensorflow.keras')
    datasets = DummyModule('tensorflow.keras.datasets')
    mnist = DummyModule('tensorflow.keras.datasets.mnist')
    utils = DummyModule('tensorflow.keras.utils')
    models = DummyModule('tensorflow.keras.models')
    layers = DummyModule('tensorflow.keras.layers')

    mnist.load_data = fake_load_data
    utils.plot_model = lambda *a, **k: None
    utils.to_categorical = lambda y: np.eye(10)[y]
    models.Sequential = DummySequential
    layers.Dense = lambda *a, **k: None
    layers.Activation = lambda *a, **k: None
    layers.Dropout = lambda *a, **k: None

    datasets.mnist = mnist
    keras.datasets = datasets
    keras.utils = utils
    keras.models = models
    keras.layers = layers
    tf.keras = keras

    modules = {
        'tensorflow': tf,
        'tensorflow.keras': keras,
        'tensorflow.keras.datasets': datasets,
        'tensorflow.keras.datasets.mnist': mnist,
        'tensorflow.keras.utils': utils,
        'tensorflow.keras.models': models,
        'tensorflow.keras.layers': layers,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_mlp_mnist_script_runs(tmp_path, monkeypatch):
    setup_fake_tensorflow(monkeypatch)
    script_path = Path(__file__).resolve().parents[1] / 'chapter1-keras-quick-tour' / 'mlp-mnist-1.3.2.py'
    globals_dict = runpy.run_path(str(script_path))
    model = globals_dict['model']
    assert model.output_shape[-1] == 10
