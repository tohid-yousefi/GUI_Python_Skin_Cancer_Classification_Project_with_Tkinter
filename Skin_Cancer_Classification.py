# Import Necessary Libraries
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

# Load Dataset
skin_df = pd.read_csv("HAM10000_metadata.csv")

# Data Preprocessing
data_folder_name = "HAM10000_images/"
ext = ".jpg"
skin_df["path"] = [data_folder_name + img_id + ext for img_id in skin_df["image_id"]]
skin_df["image"] = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes

# Save DataFrame to Pickle File
skin_df.to_pickle("skin_df.pkl")

# Load Pickle File
skin_df = pd.read_pickle("skin_df.pkl")

# Standardization - Normalization
x_train = np.asarray(skin_df["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean)/x_train_std

# One-Hot Encoding
y_train = to_categorical(skin_df["dx_idx"], num_classes=skin_df["dx"].nunique())

# Create CNN Architecture
input_shape = x_train.shape[1:]
num_classes=skin_df["dx"].nunique()

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation="relu", padding="same", input_shape=input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3), activation="relu", padding="same"))
model.add(Conv2D(64, kernel_size = (3,3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.summary()

optimizer = Adam(lr = 0.0001)
model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 5
batch_size = 25

history = model.fit(x = x_train, y = y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
model.save("my_model_2.h5")

# Load Model
model_1 = load_model("my_model_1.h5")
model_2 = load_model("my_model_2.h5")

# Prediction
index = 2
y_pred = model_1.predict(x_train[index].reshape(1,75,100,3))
y_pred_class = np.argmax(y_pred, axis=1)

# ******* GUI *********
window = tk.Tk()
window.geometry("1088x644")
window.title("Skin Cancer Classification")

# Global Variables

img_path = ""
img_name = ""
count = 0

# Menu
def imageResize(img):
    basewidth = 500
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img

def openImage():
    global img_path
    global img_name
    global count
    
    count += 1
    if count != 1:
        messagebox.showinfo(title="Warning", message="Only one image can be opened")
    else:
        img_path = filedialog.askopenfilename(title="Select an image file...")
        img_name = img_path.split("/")[-1].split(".")[0]
        tk.Label(frame1, text=img_name, bd=3).pack(pady=10)
        
        #open & show image
        img = Image.open(img_path)
        img = imageResize(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(frame1, image=img)
        panel.image = img
        panel.pack(padx=15, pady=10)
        
        #image features
        data = pd.read_csv("HAM10000_metadata.csv")
        cancer = data[data.image_id==img_name]
        
        for i in range(cancer.size):
            x = 0.5
            y = (i/10)/2
            tk.Label(frame3, font=("Times", 12), text=str(cancer.iloc[0, i])).place(relx=x, rely=y)
        
    
    
menubar = tk.Menu(window)
window.config(menu = menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label="File", menu=file)
file.add_command(label="Open Image", command=openImage)

# Frames
frame_left = tk.Frame(window, width=540, height=640, bd="2")
frame_left.grid(row=0, column=0)

frame_right = tk.Frame(window, width=540, height=640, bd="2")
frame_right.grid(row=0, column=1)

frame1 = tk.LabelFrame(frame_left, text="Image", width=540, height=500)
frame1.grid(row=0, column=0)

frame2 = tk.LabelFrame(frame_left, text="Model & Save", width=540, height=140)
frame2.grid(row=1, column=0)

frame3 = tk.LabelFrame(frame_right, text="Features", width=270, height=640)
frame3.grid(row=0, column=0)

frame4 = tk.LabelFrame(frame_right, text="Results", width=270, height=640)
frame4.grid(row=0, column=1)

#frame3

def classification_Func():
    if img_path != "" and models.get() != "":
        if models.get() == "Model_1":
            classification_model = model_1
        else:
            classification_model = model_2
        
        z = skin_df[skin_df.image_id == img_name]
        z = z.image.values[0].reshape(1,75,100,3)
        z = (z - x_train_mean)/x_train_std
        
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_cancer = list(skin_df.dx.unique())[h_index]
        
        for i in range(len(h)):
            x = 0.5
            y = (i/10)/2
            
            if i != h_index:
                tk.Label(frame4, text=str(h[i])).place(relx=x, rely=y)
            else:
                tk.Label(frame4, text=str(h[i]), bg="green").place(relx=x, rely=y)
        
        if check_var.get() == 1:
            val = entry.get()
            entry.config(state="disabled")
            path_name = val + ".txt"
            save_text = img_path + "  , " + str(predicted_cancer)
            text_file = open(path_name, "w")
            text_file.write(save_text)
            text_file.close()
        
    else:
        messagebox.showinfo(title="Warning", message="At First Choose Image and Model")
    

columns = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
for i in range(len(columns)):
    x = 0.1
    y = (i/10)/2
    tk.Label(frame3, font=("Times", 12), text=str(columns[i]) + ": ").place(relx=x,rely=y)

classify_btn = tk.Button(frame3, text="Classify", font=("Times", 13), 
                         bg="red", activebackground="orange", 
                         command=classification_Func)
classify_btn.place(relx = 0.25, rely=0.5)

#frame4
labels = skin_df.dx.unique()
for i in range(len(labels)):
    x = 0.1
    y = (i/10)/2
    tk.Label(frame4, font=("Times", 12), text=str(labels[i]) + ": ").place(relx=x, rely=y)

#frame2
model_selection_label = tk.Label(frame2, text="Choose Classification Model: ")
model_selection_label.grid(row=0, column=0, padx=5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2, textvariable=models, values=("Model_1", "Model_2"), state="readonly")
model_selection.grid(row=0, column=1, padx=5)

check_var = tk.IntVar()
check_var.set(0)
xbox = tk.Checkbutton(frame2, text="Save Classification Results", variable=check_var)
xbox.grid(row=1, column=0, pady=5)

entry = tk.Entry(frame2, width=23)
entry.insert(index=0, string="Saving name...")
entry.grid(row=1, column=1)

window.mainloop()





















