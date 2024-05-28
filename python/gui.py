
import numpy as np
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,filedialog,messagebox
from PIL import Image, ImageTk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

img_name = "image1.jpg"
wm_name = "watermark2.jpg"
watermarked_img = "Watermarked_Image.jpg"
key = 50
bs = 8
w1 = 64
w2 = 64
fact = 8
indx = 0
indy = 0
b_cut = 50
val1 = []
val2 = []
img_ref = []


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def NCC(img1, img2):
    return abs(np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2)))

def watermark_image(img, wm):
    if len(img.shape) == 2:  # Jika gambar grayscale, ubah jadi berwarna
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if len(wm.shape) == 2:  # Jika watermark grayscale, ubah jadi berwarna
        wm = cv.cvtColor(wm, cv.COLOR_GRAY2BGR)
    
    c1, c2, _ = img.shape  # Update to handle color images
    c1x = c1
    c2x = c2
    c1 -= b_cut * 2
    c2 -= b_cut * 2
    w1, w2, _ = wm.shape  # Update to handle color images

    print(c1, c2, w1, w2)
    canvas.create_text(
        400.0,
        412.0,
        anchor="nw",
        text=f'({c1},{c2}),({w1},{w2})',
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )

    if c1 * c2 // (bs * bs) < w1 * w2:
        print("Watermark too large.")
        return img

    st = set()
    blocks = (c1 // bs) * (c2 // bs)
    print("Blocks available", blocks)
    canvas.create_text(
        540.0,
        412.0,
        anchor="nw",
        text=f'{blocks}',
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )
    blocks_needed = w1 * w2

    i = 0
    j = 0
    imf = np.float32(img)
    while i < c1x:
        while j < c2x:
            for k in range(3):  # Apply DCT for each color channel
                dst = cv.dct(imf[i:i + bs, j:j + bs, k] / 1.0)
                imf[i:i + bs, j:j + bs, k] = cv.idct(dst)
            j += bs
        j = 0
        i += bs

    final = img
    random.seed(key)
    i = 0
    print("Blocks needed", blocks_needed)
    canvas.create_text(
        400.0,
        445.0,
        anchor="nw",
        text=f'{blocks_needed}',
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )
    cnt = 0
    while i < blocks_needed:
        to_embed = wm[i // w2][i % w2]
        ch = [0, 0, 0]
        if to_embed[0] >= 127:  # Embed watermark based on the first channel
            to_embed = [1, 1, 1]
            ch = [255, 255, 255]
        else:
            to_embed = [0, 0, 0]

        wm[i // w2][i % w2] = ch

        x = random.randint(1, blocks)
        if x in st:
            continue
        st.add(x)
        n = c1 // bs
        m = c2 // bs
        ind_i = (x // m) * bs + b_cut
        ind_j = (x % m) * bs + b_cut
        for k in range(3):  # Embed watermark into each color channel
            dct_block = cv.dct(imf[ind_i:ind_i + bs, ind_j:ind_j + bs, k] / 1.0)
            elem = dct_block[indx][indy]
            elem /= fact
            ch = elem
            if to_embed[k] % 2 == 1:
                if math.ceil(elem) % 2 == 1:
                    elem = math.ceil(elem)
                else:
                    elem = math.ceil(elem) - 1
            else:
                if math.ceil(elem) % 2 == 0:
                    elem = math.ceil(elem)
                else:
                    elem = math.ceil(elem) - 1

            dct_block[indx][indy] = elem * fact
            val1.append((elem * fact, to_embed[k]))
            if cnt < 5:
                cnt += 1

            final[ind_i:ind_i + bs, ind_j:ind_j + bs, k] = cv.idct(dct_block)
            imf[ind_i:ind_i + bs, ind_j:ind_j + bs, k] = cv.idct(dct_block)
        i += 1

    final = np.uint8(final)
    print("PSNR is:", psnr(imf, img))
    canvas.create_text(
        540.0,
        445.0,
        anchor="nw",
        text=f'{psnr(imf,img)}',
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )
    # cv.imshow("Final", final)
    cv.imwrite(watermarked_img, final)
    global im_watermarked
    im_watermarked = Image.open(watermarked_img)
    img_width, img_height = im_watermarked.size
    if img_width > 338 or img_height > 180:
            #loop over the image width and height until it is approximately
            #the same size as the canvas
            while img_width > 338 or img_height > 180:
                img_width *= .99
                img_height *= .99
            im_watermarked = im_watermarked.resize((int(img_width),int(img_height)))
    image_image_4 = ImageTk.PhotoImage(im_watermarked)
    canvas.img = image_image_4
    canvas.create_image(539, 243, image=image_image_4, anchor=tk.CENTER)
    img_ref.append(image_image_4)

    return imf

def open_image():
    file_path1 = filedialog.askopenfilename()
    if file_path1:
        global img_name
        global im
        global image_1
        img_name = file_path1
        im = Image.open(file_path1)
        img_width, img_height = im.size
        print("Image selected:", img_name)
        img = cv.imread(img_name)  # Read in color
        if img_width > 338 or img_height > 180:
            #loop over the image width and height until it is approximately
            #the same size as the canvas
            while img_width > 338 or img_height > 180:
                img_width *= .99
                img_height *= .99
            im = im.resize((int(img_width),int(img_height)))
            
        if img is not None:
            image_image_1 = ImageTk.PhotoImage(im)
            canvas.img = image_image_1
            image_1 = canvas.create_image(181, 243, image=image_image_1)
            img_ref.append(image_image_1)
        else:
            messagebox.showerror("Error", "Failed to load image.")
     
def open_watermark():
    file_path = filedialog.askopenfilename()
    if file_path:
        global wm_name
        global im2
        global image_2
        wm_name = file_path
        im2 = Image.open(file_path)
        img_width, img_height = im2.size
        print("Watermark selected:", wm_name)
        wm = cv.imread(wm_name)  # Read in color
        if img_width > 159 or img_height > 116:
            #loop over the image width and height until it is approximately
            #the same size as the canvas
            while img_width > 159 or img_height > 116:
                img_width *= .99
                img_height *= .99
            im2 = im2.resize((int(img_width),int(img_height)))
            
        if wm is not None:
            image_image_2 = ImageTk.PhotoImage(im2)
            canvas.img = image_image_2
            image_2 = canvas.create_image(270, 421, image=image_image_2, anchor=tk.CENTER)
            img_ref.append(image_image_2)
            wm = cv.resize(wm, dsize=(159, 116), interpolation=cv.INTER_CUBIC)
            # cv.imshow('Selected Watermark', wm)
        else:
            messagebox.showerror("Error", "Failed to load watermark.")

def embed_watermark():
    img = cv.imread(img_name)  # Read in color
    wm = cv.imread(wm_name)  # Read in color
    if img is not None and wm is not None:
        wm = cv.resize(wm, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
        watermark_image(img, wm)
        messagebox.showinfo("Info", "Watermark embedding completed!")
    else:
        messagebox.showerror("Error", "Failed to load image or watermark.")

if __name__ == "__main__":
    
    window = Tk()

    window.geometry("720x512")
    window.configure(bg = "#FFFFFF")
    window.title("Watermark Embedding")

    canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 512,
        width = 720,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )
    
    canvas.place(x = 0, y = 0)
    canvas.create_text(
        264.0,
        31.0,
        anchor="nw",
        text="Image Watermarking using DCT\n",
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )     
    canvas.create_text(
        102.0,
        69.0,
        anchor="nw",
        text="Open Image ",
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )

    canvas.create_text(
        281.0,
        69.0,
        anchor="nw",
        text="Open Watermark",
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )

    canvas.create_text(
        460.0,
        69.0,
        anchor="nw",
        text="Embed Watermark",
        fill="#000000",
        font=("Inter Bold", 12 * -1)
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: open_image(),
        relief="flat"
    )
    button_1.place(
        x=117.0,
        y=92.0,
        width=128.0,
        height=40.0
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: open_watermark(),
        relief="flat"
    )
    button_2.place(
        x=296.0,
        y=92.0,
        width=128.0,
        height=40.0
    )

    button_image_3 = PhotoImage(
        file=relative_to_assets("button_3.png"))
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: embed_watermark(),
        relief="flat"
    )
    button_3.place(
        x=476.0,
        y=92.0,
        width=128.0,
        height=40.0
    )

    image_image_1 = PhotoImage(
        file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(
        181.0,
        243.0,
        image=image_image_1
    )

    image_image_2 = PhotoImage(
        file=relative_to_assets("image_2.png"))
    image_2 = canvas.create_image(
        270.0,
        421.0,
        image=image_image_2
    )
    
    canvas.create_text(
        141.0,
        336.0,
        anchor="nw",
        text="Image Source",
        fill="#000000",
        font=("Inter", 12 * -1)
    )
    
    canvas.create_text(
        505.0,
        336.0,
        anchor="nw",
        text="Embedded",
        fill="#000000",
        font=("Inter", 12 * -1)
    )
    
    canvas.create_text(
        237.0,
        482.0,
        anchor="nw",
        text="Watermark",
        fill="#000000",
        font=("Inter", 12 * -1)
    )

    canvas.create_rectangle(
        370.0,
        363.0,
        708.0,
        479.0,
        fill="#FFFFFF",
        outline=""
    )
    
    canvas.create_text(
    507.0,
    372.0,
    anchor="nw",
    text="Information",
    fill="#000000",
    font=("Inter", 12 * -1)
    )

    canvas.create_text(
        400.0,
        397.0,
        anchor="nw",
        text="Shape :",
        fill="#000000",
        font=("Inter", 12 * -1)
    )

    canvas.create_text(
        540.0,
        397.0,
        anchor="nw",
        text="Block Available :",
        fill="#000000",
        font=("Inter", 12 * -1)
    )

    canvas.create_text(
        400.0,
        430.0,
        anchor="nw",
        text="Block Needed :",
        fill="#000000",
        font=("Inter", 12 * -1)
    )

    canvas.create_text(
        540.0,
        430.0,
        anchor="nw",
        text="PSNR :",
        fill="#000000",
        font=("Inter", 12 * -1)
    )

    image_image_4 = PhotoImage(
        file=relative_to_assets("image_4.png"))
    image_4 = canvas.create_image(
        539.0,
        243.0,
        image=image_image_4
    )
    window.resizable(False, False)
    window.mainloop()
