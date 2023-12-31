from PIL import Image
import torchvision.transforms as transforms
import onnxruntime
import numpy as np 

def to_numpy(tensor):
    print("tensor.requires_grad: ",tensor.requires_grad)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession("../../data/super_resolution.onnx", 
                                           providers=["CPUExecutionProvider"])


img = Image.open("../../data/super_reso.png")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

print(ort_session.get_inputs()[0].name)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}

ort_outs = ort_session.run(None, ort_inputs) # The result

print("len(ort_outs): ",len(ort_outs[0]))
img_out_y = ort_outs[0]

print("img_out_y.size: ", img_out_y.size)
print("np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]).size: ",np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]).size)
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

print("img_out_y.size: ", img_out_y.size)
print("Image.BICUBIC: ",Image.BICUBIC)

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("../../data/out_super_reso.jpg")