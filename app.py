import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve
import gradio as gr

urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5", "mnist-model.h5")
model = tf.keras.models.load_model("mnist-model.h5")

def recognize_digit(image):
    image = image.reshape(1, -1)  
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

output_component = gr.Label(num_top_classes=5)

gr.Interface(fn=recognize_digit, 
             inputs="sketchpad", 
             outputs=output_component,
             title="Digit Sketchpad",
             description="Draw a digit between 0 and 9.").launch(share=True);