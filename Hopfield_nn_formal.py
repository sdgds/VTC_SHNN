# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import copy
import scipy.io as scio
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import itertools
from sklearn.metrics.pairwise import euclidean_distances
import sys
sys.path.append('D:\\TDCNN')
import BrainSOM
import Hopfield_VTCSOM
import Generative_adv_picture


### Data
def bao_preprocess_pic(img):
    img = img.resize((224,224))
    img = np.array(img)-237.169
    picimg = torch.Tensor(img).permute(2,0,1)
    return picimg

data_transforms = {
    'see': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'val_resize': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'see_flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1)]),
    'flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()


def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_pca(som, pca, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    mean_features = np.mean(Response, axis=0)
    std_features = np.std(Response, axis=0)
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som, (mean_features, std_features)

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1])) - 1
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask  
    
def Picture_activation(pic_dir, som, pca, pca_index, mean_features, std_features, mask=None):
    """"mask is like (3,224,224)"""
    img = Image.open(pic_dir).convert('RGB')
    if mask!=None:
        picimg = data_transforms['val'](img) * mask
    else:
        picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    if mask!=None:
        img_mask_see = np.multiply(img_see, mask.permute(1,2,0).data.numpy())
    else:
        img_mask_see = img_see
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, img_mask_see, response_som

def Pure_picture_activation(pic_dir, prepro_method, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB')
    if prepro_method=='val':
        picimg = data_transforms['val'](img)
        img_see = np.array(data_transforms['see'](img))
    if prepro_method=='val_resize':
        picimg = data_transforms['val_resize'](img)
        img_see = np.array(data_transforms['see'](img))
    if prepro_method=='flip':
        picimg = data_transforms['flip'](img)
        img_see = np.array(data_transforms['see_flip'](img))
    if prepro_method=='bao':
        picimg = bao_preprocess_pic(img)
        img_see = np.array(data_transforms['val'](img))        
    picimg = picimg.unsqueeze(0) 
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som

def Upset_picture_activation(pic_dir, block_num_row, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB').resize((224,224))
    img = np.array(img)
    block_num_row = np.uint8(block_num_row)
    t = np.uint8(224/block_num_row)
    for time in range(1000):
        left_up_row_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_1 = left_up_row_1+t
        right_down_col_1 = left_up_col_1+t
        temp = copy.deepcopy(img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1, :])
        left_up_row_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_2 = left_up_row_2+t
        right_down_col_2 = left_up_col_2+t
        img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1,:] = img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:]
        img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:] = temp
    img = Image.fromarray(img)
    picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som
    
def plot_memory(img, initial_state, state, memory_pattern):
    plt.figure(figsize=(23,4))
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Original picture');plt.axis('off')
    plt.subplot(142)
    plt.imshow(initial_state)
    plt.title('Initial state');plt.axis('off')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(state)
    plt.title('Stable state');plt.axis('off')
    plt.colorbar() 
    plt.subplot(144)
    plt.imshow(memory_pattern)
    plt.title('right state');plt.axis('off')
    plt.colorbar() 
    
                
                
                
### sigma=6.2
som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_6.2.npy')

Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
Data = zscore(Data)
pca = PCA()
pca.fit(Data)
Response_som, (mean_features,std_features) = Functional_map_pca(som, pca, [0,1,2,3])
Response_face = Response_som[:111,:,:]
Response_place = Response_som[111:172,:,:]
Response_body = Response_som[172:250,:,:]
Response_object = Response_som[250:,:,:]
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
threshold_cohend = 0.5
face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
training_pattern = np.array([face_mask.reshape(-1),
                             place_mask.reshape(-1),
                             limb_mask.reshape(-1),
                             object_mask.reshape(-1)])


model = Hopfield_VTCSOM.Stochastic_Hopfield_nn(x=200, y=200, pflag=1, nflag=-1,
                                               patterns=[face_mask,place_mask,limb_mask,object_mask])
model.reconstruct_w_with_structure_constrain([training_pattern], 'exponential', 0.023) # Human(0.0238)






"Specialization vs Generalization"
###############################################################################
###############################################################################
### Specialization
def Get_mean_std(): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    return Response.mean(axis=0), Response.std(axis=0)
mean, std = Get_mean_std()


## SOM (No interaction/ No external field/)
External_field_prior = np.zeros((200,200))
files = os.listdir(r'D:\TDCNN\Data\fLoc_stim\all_stim\\')
Image_state_dict = dict()
for index,f in tqdm(enumerate(files)):
    pic_dir = r'D:\TDCNN\Data\fLoc_stim\all_stim\\' + f
    img_see, img_mask_see, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean, std)
    Image_state_dict[index] = initial_state
    
images_response = []
for k,v in Image_state_dict.items():
    images_response.append(v)
images_response = np.array(images_response)

plt.figure(dpi=400)
plt.plot(images_response[:, np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=1))

plt.figure(dpi=400)
temp = images_response[:, np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=1)
face_activation = np.hstack((temp[:144], temp[432:432+144]))
body_activation = np.hstack((temp[144:144+144], temp[1008:1008+144]))
car_activation = temp[288:288+144]
scene_activation = temp[576:576+144]
house_activation = temp[720:720+144]
instrument_activation = temp[864:864+144]
number_activation = np.hstack((temp[1152:1152+144], temp[1440:1440+144]))
scramble_activation = temp[1440:1440+144]
plt.bar(range(8), [face_activation.mean(), body_activation.mean(), car_activation.mean(), scene_activation.mean(), house_activation.mean(),
                   instrument_activation.mean(), number_activation.mean(), scramble_activation.mean()])
plt.scatter(np.zeros(face_activation.shape),face_activation,marker='.',s=1)
plt.scatter(np.ones(body_activation.shape),body_activation,marker='.',s=1)
plt.scatter(2*np.ones(car_activation.shape),car_activation,marker='.',s=1)
plt.scatter(3*np.ones(scene_activation.shape),scene_activation,marker='.',s=1)
plt.scatter(4*np.ones(house_activation.shape),house_activation,marker='.',s=1)
plt.scatter(5*np.ones(instrument_activation.shape),instrument_activation,marker='.',s=1)
plt.scatter(6*np.ones(number_activation.shape),number_activation,marker='.',s=1)
plt.scatter(7*np.ones(scramble_activation.shape),scramble_activation,marker='.',s=1)
plt.errorbar(range(8), [face_activation.mean(), body_activation.mean(), car_activation.mean(), scene_activation.mean(), house_activation.mean(),
                   instrument_activation.mean(), number_activation.mean(), scramble_activation.mean()], 
                       [face_activation.std()/np.sqrt(288), body_activation.std()/np.sqrt(288), car_activation.std()/np.sqrt(144), scene_activation.std()/np.sqrt(144), house_activation.std()/np.sqrt(144),
                   instrument_activation.std()/np.sqrt(144), number_activation.std()/np.sqrt(288), scramble_activation.std()/np.sqrt(144)],
            linestyle='None', ecolor='black', elinewidth=1, capsize=2)



## SHNN (No external field/)
External_field_prior = np.zeros((200,200))
files = os.listdir(r'D:\TDCNN\Data\fLoc_stim\all_stim\\')
Image_state_dict = dict()
for index,f in enumerate(files):
    print(index)
    pic_dir = r'D:\TDCNN\Data\fLoc_stim\all_stim\\' + f
    img_see, img_mask_see, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean, std)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                             H_prior=np.zeros((200,200)), H_bottom_up=initial_state, 
                                             epochs=150000, save_inter_step=150000)
    Image_state_dict[index] = stable_state[0].reshape(200,200)
    
images_response = []
for k,v in Image_state_dict.items():
    images_response.append(v)
images_response = np.array(images_response)

#images_response = np.load('D:\TDCNN\SHNN\images_response.npy')
plt.figure(dpi=400)
temp = images_response[:, np.where(face_mask==1)[0], np.where(face_mask==1)[1]].mean(axis=1)
face_activation = np.hstack((temp[:144], temp[432:432+144]))
body_activation = np.hstack((temp[144:144+144], temp[1008:1008+144]))
car_activation = temp[288:288+144]
scene_activation = temp[576:576+144]
house_activation = temp[720:720+144]
instrument_activation = temp[864:864+144]
number_activation = np.hstack((temp[1152:1152+144], temp[1440:1440+144]))
scramble_activation = temp[1440:1440+144]
plt.bar(range(8), [face_activation.mean(), body_activation.mean(), car_activation.mean(), scene_activation.mean(), house_activation.mean(),
                   instrument_activation.mean(), number_activation.mean(), scramble_activation.mean()])
plt.scatter(np.zeros(face_activation.shape),face_activation,marker='.',s=1)
plt.scatter(np.ones(body_activation.shape),body_activation,marker='.',s=1)
plt.scatter(2*np.ones(car_activation.shape),car_activation,marker='.',s=1)
plt.scatter(3*np.ones(scene_activation.shape),scene_activation,marker='.',s=1)
plt.scatter(4*np.ones(house_activation.shape),house_activation,marker='.',s=1)
plt.scatter(5*np.ones(instrument_activation.shape),instrument_activation,marker='.',s=1)
plt.scatter(6*np.ones(number_activation.shape),number_activation,marker='.',s=1)
plt.scatter(7*np.ones(scramble_activation.shape),scramble_activation,marker='.',s=1)
plt.errorbar(range(8), [face_activation.mean(), body_activation.mean(), car_activation.mean(), scene_activation.mean(), house_activation.mean(),
                   instrument_activation.mean(), number_activation.mean(), scramble_activation.mean()], 
                       [face_activation.std()/np.sqrt(288), body_activation.std()/np.sqrt(288), car_activation.std()/np.sqrt(144), scene_activation.std()/np.sqrt(144), house_activation.std()/np.sqrt(144),
                   instrument_activation.std()/np.sqrt(144), number_activation.std()/np.sqrt(288), scramble_activation.std()/np.sqrt(144)],
            linestyle='None', ecolor='black', elinewidth=1, capsize=2)





### Generalization
# Half face
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.zeros((3,224,224))
mask[:,:135,:] = 1
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                    mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                H_prior=External_field_prior, H_bottom_up=initial_state, epochs=200000, save_inter_step=1000)
plot_memory(img_mask_see, initial_state, stable_state[0].reshape(200,200), 
            training_pattern[0].reshape(200,200))
## visulization
model.dynamics_pattern('Half_face.gif', model.dynamics_state)
Dynamic_states = model.dynamics_state
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()










"Dynamics + Prior (face vase illusion)"
###############################################################################
###############################################################################
### Face Vase illusion
## top down (initial steps)
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_vase_illusion_1.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((object_mean_act-face_mean_act)/(face_mean_act+object_mean_act), label='(obj-face)/(face+obj)')
plt.plot((object_mean_act-face_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(obj-face-body-place)/(face+obj+body+place)')
plt.legend()



## top down (initial steps)
External_field_prior = copy.deepcopy(face_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_vase_illusion_2.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()




### Face stimuli + Object top down
## top down (initial steps)
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/FC_009_M5.png'
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_stim_obj_topdown.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()


### Object stimuli + Face top down
## top down (initial steps)
External_field_prior = copy.deepcopy(face_mask) * 2
pic_dir = 'D://TDCNN//HCP//HCP_WM//object/TO_076_TOOL76_BA.png'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Object_stim_face_topdown.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((object_mean_act-face_mean_act)/(face_mean_act+object_mean_act), label='(obj-face)/(face+obj)')
plt.plot((object_mean_act-face_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(obj-face-body-place)/(face+obj+body+place)')
plt.legend()



### Random stimuli + bottom up
# bottom up (initial steps)
mean_stable_state = np.zeros((200,200))
pic_list = os.listdir('D://TDCNN//HCP//HCP_WM//face/')
pic_list.remove('.DS_Store')
for i in range(30):
    External_field_prior = np.zeros((200,200))
    pic_dir = 'D://TDCNN//HCP//HCP_WM//face/' + np.random.choice(pic_list,1)[0]
    img_see, alexnet_output, initial_state = Upset_picture_activation(pic_dir, 4, som, pca, [0,1,2,3], mean_features, std_features)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                              epochs=160000, save_inter_step=1000)
    plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
    mean_stable_state += stable_state[0].reshape(200,200)
mean_stable_state = mean_stable_state/30


### Random stimuli + bottom up + top down
External_field_prior = copy.deepcopy(face_mask) * 2
pic_list = os.listdir('D://TDCNN//HCP//HCP_WM//face/')
pic_list.remove('.DS_Store')
mean_stable_state = np.zeros((200,200))
for i in range(30):
    pic_dir = 'D://TDCNN//HCP//HCP_WM//face/' + np.random.choice(pic_list,1)[0]
    img_see, alexnet_output, initial_state = Upset_picture_activation(pic_dir, 4, som, pca, [0,1,2,3], mean_features, std_features)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
    ## dynamics only (posterior steps)
    External_field_prior = np.zeros((200,200))
    stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
    plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
    mean_stable_state += stable_state[0].reshape(200,200)
mean_stable_state = mean_stable_state/30





