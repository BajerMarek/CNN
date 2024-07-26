import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Going_modular import model_builder,data_setup
train_dataloader, test_dataloader, class_names = data_setup.Create_dataloader(...)

model = model_builder.TinyVgg(input_shape=3,
                              hidden_units=10,
                              output_shape=len(class_names))