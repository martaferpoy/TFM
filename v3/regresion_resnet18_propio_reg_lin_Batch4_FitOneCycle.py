#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install timm


# In[49]:


import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import time
import timm
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# In[50]:


device


# In[51]:


from comet_ml import Experiment

experiment = Experiment(
  api_key="",
  project_name="apnea-bispectrum-tfm",
  workspace=""
)

experiment.set_name("REGRESION ResNet18 propio reg_lin Batch 4 FitOneCycle")


# In[52]:


data_t = open('/home/marta/MsC/data/data_train.pkl', 'rb')
data_train = pickle.load(data_t)

data_v = open('/home/marta/MsC/data/data_validation.pkl', 'rb')
data_validation = pickle.load(data_v)


# In[53]:


len(data_train), len(data_validation)


# In[54]:


hiperparametros = {
    'epocas' : 150,
    'lr' : 0.1,
    'batch_size' : 4,
    'seed' : 56389856
}


# In[55]:


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        
    def __getitem__(self, index):
        data_ind = copy.deepcopy(self.data[index])

        name_ind = data_ind['name']
        x_torres_ind = data_ind['image thorres']
        x_abdo_ind = data_ind['image abdo']
        y_events_ind = data_ind['all events']
        y_central_ind = data_ind['central events']
        y_hypo_ind = data_ind['hypo events']
        y_osa_ind = data_ind['osa events']

        thorres = torch.from_numpy(np.moveaxis(x_torres_ind / 255, -1, 0))
        abdores = torch.from_numpy(np.moveaxis(x_abdo_ind / 255, -1, 0))
        
        x = torch.cat([thorres, abdores], axis = 0)
        y = torch.Tensor([y_events_ind]).float()

        return x, y, name_ind

    def __len__(self):
        return len(self.data)


# In[56]:


dataset_train = MyDataset(data_train)
dataset_validation = MyDataset(data_validation)


# In[57]:


dataloader_train = DataLoader(dataset_train, batch_size = hiperparametros['batch_size'], shuffle = True)
dataloader_validation = DataLoader(dataset_validation, batch_size = hiperparametros['batch_size'], shuffle = True)


# In[58]:


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = torch.nn.Sequential()

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input_x):
        shortcut = self.shortcut(input_x)
        input_x = torch.nn.ReLU()(self.bn1(self.conv1(input_x)))
        input_x = torch.nn.ReLU()(self.bn2(self.conv2(input_x)))
        input_x = input_x + shortcut
        return torch.nn.ReLU()(input_x)


# In[59]:


class ResNet18(torch.nn.Module):
    def __init__(self, in_channels, resblock, outputs=2):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.layer1 = torch.nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = torch.nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = torch.nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = torch.nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )
        
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=outputs, bias=True)

    def forward(self, x):
        output_L0 = self.layer0(x)
        output_L1 = self.layer1(output_L0)
        output_L2 = self.layer2(output_L1)
        output_L3 = self.layer3(output_L2)
        output_L4 = self.layer4(output_L3)
        
        output_adaptive_avg_pool = self.adaptiveavgpool(output_L4).reshape((-1, 512))
        output_fc = self.fc(output_adaptive_avg_pool)
        
        return output_fc


# In[60]:


model = ResNet18(6, ResBlock, outputs=1).to(device)


# In[87]:


def validate_loss_kappa_acc(val_loss, val_kappa, val_accuracy, coefs, intercept):
    model.eval()
    loss_medio_val = 0
    pasos_val = 0
  
    dict_patients_val = {}
    outputs_list_val = torch.empty(0).to(device)
    labels_list_val = torch.empty(0).to(device)
    names_list_val = ()

    with torch.no_grad():
        for data in dataloader_validation:
            inputs, labels, names = data[0].to(device, dtype=torch.float32), data[1].to(device), data[2]
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            loss_medio_val += loss_val.item()
            outputs_list_val = torch.cat((outputs_list_val, outputs))
            labels_list_val = torch.cat((labels_list_val, labels))
            names_list_val = names_list_val + names

            pasos_val += 1
      
    val_loss.append(loss_medio_val/pasos_val)
    experiment.log_metric('Loss validation',loss_medio_val/pasos_val)
    
    dict_patients_val = group_pacients(outputs_list_val.detach().to('cpu').numpy(), labels_list_val.detach().to('cpu').numpy(), names_list_val, dict_patients_val)

    kappa, accuracy = get_AHI_kappa_acc_val(dict_patients_val, coefs, intercept)

    val_kappa.append(kappa)
    val_accuracy.append(accuracy)
    
    experiment.log_metric('Kappa validation',kappa)
    experiment.log_metric('Accuracy validation',accuracy)
    
    return(val_loss, val_kappa, val_accuracy, loss_medio_val/pasos_val, kappa, accuracy)


# In[88]:


def reg_lin(dictionary):
    AHI_pred = []
    AHI_real = []
  
    for key in [*dictionary]:
        individual = dictionary[key]
        outputs_ind = [item[0] for item in individual]
        labels_ind = [item[1] for item in individual]

        AHI_pred.append(sum(outputs_ind)/len(outputs_ind)*3)
        AHI_real.append(sum(labels_ind)/len(labels_ind)*3)
        
    reg = LinearRegression().fit(AHI_pred, AHI_real)
    return(reg.coef_, reg.intercept_)


# In[89]:


def group_pacients(output_batch, labels_batch, names_batch, dict_p):
    for i in range(output_batch.shape[0]):
        output = output_batch[i]; label = labels_batch[i]; name = names_batch[i]

        if name in [*dict_p]:
            dict_p[name].append([output, label])
        else:
            dict_p[name] = [[output, label]]

    return dict_p


# In[90]:


def get_AHI_kappa_acc(dictionary):
    AHI_pred = []
    AHI_real = []
  
    for key in [*dictionary]:
        individual = dictionary[key]
        outputs_ind = [item[0] for item in individual]
        labels_ind = [item[1] for item in individual]

        AHI_pred.append(sum(outputs_ind)/len(outputs_ind)*3)
        AHI_real.append(sum(labels_ind)/len(labels_ind)*3)

    AHI_pred_disc = np.digitize(AHI_pred, bins = np.array([5, 15, 30]))
    AHI_real_disc = np.digitize(AHI_real, bins = np.array([5, 15, 30]))

    kappa = cohen_kappa_score(AHI_pred_disc, AHI_real_disc)

    accuracy = sum(AHI_pred_disc == AHI_real_disc).item()/len(AHI_real_disc)
    
    return kappa, accuracy


# In[126]:


def get_AHI_kappa_acc_val(dictionary, coefs, intercept):
    AHI_pred = []
    AHI_real = []
  
    for key in [*dictionary]:
        individual = dictionary[key]
        outputs_ind = [item[0] for item in individual]
        labels_ind = [item[1] for item in individual]

        AHI_pred.append(sum(outputs_ind)/len(outputs_ind)*3)
        AHI_real.append(sum(labels_ind)/len(labels_ind)*3)
        
    AHI_pred_reg = intercept[0]+ np.array(AHI_pred)*coefs[0][0] 
    
    AHI_pred_disc = np.digitize(AHI_pred_reg.tolist(), bins = np.array([5, 15, 30]))
    AHI_real_disc = np.digitize(AHI_real, bins = np.array([5, 15, 30]))

    kappa = cohen_kappa_score(AHI_pred_disc, AHI_real_disc)

    accuracy = sum(AHI_pred_disc == AHI_real_disc).item()/len(AHI_real_disc)
    
    return kappa, accuracy


# In[127]:


criterion = torch.nn.HuberLoss(reduction= "mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=hiperparametros['lr'])


# In[ ]:


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
                                                steps_per_epoch=len(dataloader_train),
                                                epochs=50)


# In[129]:


loss_list =[]; validation_loss_list = []
kappa_list = []; validation_kappa_list = []
accuracy_list = []; validation_accuracy_list = []

loss_min = 10000000; kappa_max = -100000; accuracy_max = -1000
best_model = None
best_outputs = torch.empty(0).to(device)
best_labels = torch.empty(0).to(device)
best_names = ()
best_coefs = []
best_intercept = []

tiempo = time.time()

for epoch in range(hiperparametros['epocas']):
    experiment.set_epoch(epoch)

    loss_medio = 0
    pasos = 0
    
    dict_patients = {}
    outputs_list = torch.empty(0).to(device)
    labels_list = torch.empty(0).to(device)
    names_list = ()

    # Bucle de entrenamiento
    for i, data in enumerate(dataloader_train, 0): 
        inputs, labels, names = data[0].to(device, dtype=torch.float32), data[1].to(device), data[2]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_medio += loss.item()
        loss.backward()
        optimizer.step()
        
        outputs_list = torch.cat((outputs_list, outputs))
        labels_list = torch.cat((labels_list, labels))
        names_list = names_list + names
        
        pasos += 1
        
    scheduler.step()

    print("Epoch = ", epoch)
    loss_list.append(loss_medio/pasos)
    
    experiment.log_metric('Loss train', loss_medio/pasos)
    
    dict_patients = group_pacients(outputs_list.detach().to('cpu').numpy(), labels_list.detach().to('cpu').numpy(), names_list, dict_patients)
    coefs, intercept = reg_lin(dict_patients)
    
    kappa, accuracy = get_AHI_kappa_acc(dict_patients)

    kappa_list.append(kappa)
    accuracy_list.append(accuracy)
    experiment.log_metric('Kappa train',kappa)
    experiment.log_metric('Accuracy train',accuracy)

    validation_loss_list, validation_kappa_list, validation_accuracy_list, loss_actual, kappa_actual, accuracy_actual = validate_loss_kappa_acc(validation_loss_list, validation_kappa_list, validation_accuracy_list, coefs, intercept)
    
    # chequeo la metrica que tengo en esta epoca, y si es mejor que en epocas anteriores, guardo el modelo
    if(kappa_actual > kappa_max):
        loss_min = loss_actual
        kappa_max = kappa_actual
        accuracy_max = accuracy_actual
        best_model = copy.deepcopy(model)
        best_outputs = outputs_list
        best_labels = labels_list
        best_names = names_list
        best_coefs = coefs
        best_intercept = intercept
    
    model.train()
        
experiment.log_metric('Best loss validation', loss_min)
experiment.log_metric('Best kappa validation', kappa_max)
experiment.log_metric('Best accuracy validation', accuracy_max)
experiment.log_metric('Coef_reg_lin', best_coefs[0][0])
experiment.log_metric('Intercept_reg_lin', best_intercept[0])

print("Time required: ", time.time() - tiempo)


# In[ ]:


last_lr = scheduler.get_last_lr()
print('last lr: ', last_lr)
experiment.log_metric('Last learning rate', last_lr)


# In[130]:


def get_AHI(dictionary):
    AHI_pred = []
    AHI_real = []
  
    for key in [*dictionary]:
        individual = dictionary[key]
        outputs_ind = [item[0] for item in individual]
        labels_ind = [item[1] for item in individual]

        AHI_pred.append(sum(outputs_ind)/len(outputs_ind)*3)
        AHI_real.append(sum(labels_ind)/len(labels_ind)*3)

    AHI_pred_disc = np.digitize(AHI_pred, bins = np.array([5,15,30]))
    AHI_real_disc = np.digitize(AHI_real, bins = np.array([5,15,30]))

    return AHI_pred_disc, AHI_real_disc


# In[132]:


dict_best_patients = {}
dict_best_patients = group_pacients(best_outputs.detach().to('cpu').numpy(), best_labels.detach().to('cpu').numpy(), best_names, dict_best_patients)
    
best_y_pred, best_y_real = get_AHI(dict_best_patients)


# In[133]:


AHI_pred_disc, AHI_real_disc = get_AHI(dict_best_patients)


# In[134]:


print('Best coefs: ', best_coefs[0][0])
print('Best intercept: ', best_intercept[0])


# In[135]:


def sens_especif_mat_conf_por_clases_inferiores(AHI_real_disc_, AHI_pred_disc_, clase):
    FP = sum(AHI_pred_disc_[AHI_real_disc_ > clase] <= clase)
    TP = sum(AHI_pred_disc_[AHI_real_disc_ <= clase] <= clase)
    TN = sum(AHI_pred_disc_[AHI_real_disc_ > clase] > clase)
    FN = sum(AHI_pred_disc_[AHI_real_disc_ <= clase] > clase)
    sensibilidad = TP/(TP + FN)
    especifidad = TN/(TN + FP)
    VPP = TP/(TP + FP)
    VPN = TN/(TN + FN)
    
    AHI_real_disc_clase = np.copy(AHI_real_disc_)
    AHI_real_disc_clase[AHI_real_disc_clase > clase] = 100
    AHI_real_disc_clase[AHI_real_disc_clase <= clase] = 0

    AHI_pred_disc_clase = np.copy(AHI_pred_disc_)
    AHI_pred_disc_clase[AHI_pred_disc_clase > clase] = 100
    AHI_pred_disc_clase[AHI_pred_disc_clase <= clase] = 0
    
    best_cfmatrix_val_clase = confusion_matrix(AHI_real_disc_clase, AHI_pred_disc_clase)
    
    group_counts = ["{0:0.0f}".format(value) for value in
                best_cfmatrix_val_clase.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         best_cfmatrix_val_clase.flatten()/np.sum(best_cfmatrix_val_clase)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    plt.rcdefaults()
    sns.set(font_scale=1.4)
    sns.set(rc = {'figure.figsize':(5,4)})

    sns.heatmap(best_cfmatrix_val_clase, annot=labels, annot_kws={"size": 14}, fmt='',  cmap='Blues')

    plt.title("\n Matriz de confusión")
    plt.xlabel("Estimado \n \n \n")
    plt.ylabel("\n \n Real")
    #plt.show()
    plt.clf()
    plt.close()

    return sensibilidad, especifidad, VPP, VPN


# In[136]:


try:
    sensibilidad_0_inf, especifidad_0_inf, VPP_0_inf, VPN_0_inf = sens_especif_mat_conf_por_clases_inferiores(AHI_real_disc, AHI_pred_disc, 0)
    print("Sensibilidad = ", sensibilidad_0_inf)
    print("Especifidad = ", especifidad_0_inf)
    print("VPP = ", VPP_0_inf)
    print("VPN = ", VPN_0_inf)
except Exception:
    print('Division by zero error')


# In[137]:


try:
    sensibilidad_1_inf, especifidad_1_inf, VPP_1_inf, VPN_1_inf = sens_especif_mat_conf_por_clases_inferiores(AHI_real_disc, AHI_pred_disc, 1)
    print("Sensibilidad = ", sensibilidad_1_inf)
    print("Especifidad = ", especifidad_1_inf)
    print("VPP = ", VPP_1_inf)
    print("VPN = ", VPN_1_inf)
except Exception:
    print('Division by zero error')


# In[138]:


try:
    sensibilidad_2_inf, especifidad_2_inf, VPP_2_inf, VPN_2_inf = sens_especif_mat_conf_por_clases_inferiores(AHI_real_disc, AHI_pred_disc, 2)
    print("Sensibilidad = ", sensibilidad_2_inf)
    print("Especifidad = ", especifidad_2_inf)
    print("VPP = ", VPP_2_inf)
    print("VPN = ", VPN_2_inf)
except Exception:
    print('Division by zero error')


# In[139]:


best_cfmatrix_val = confusion_matrix(AHI_real_disc, AHI_pred_disc)
cf_mat_row = np.zeros((4,4))
for i in range(best_cfmatrix_val.shape[0]):
    cf_mat_row[i,:] = best_cfmatrix_val[i]/sum(best_cfmatrix_val[i])


# In[140]:


group_counts = ["{0:0.0f}".format(value) for value in
                best_cfmatrix_val.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_mat_row.flatten()]

labels = [f"{v1}\n{v2}\n" for v1, v2 in
          zip(group_counts,group_percentages)]

labels = np.asarray(labels).reshape(4,4)

plt.rcdefaults()
sns.set(font_scale=1.4)
sns.set(rc = {'figure.figsize':(5,4)})

sns.heatmap(cf_mat_row, annot=labels, annot_kws={"size": 14}, fmt='',  cmap='Blues')

plt.title("\n Matriz de confusión por clase en validación")
plt.xlabel("Estimado \n \n \n")
plt.ylabel("\n \n Real")
experiment.log_figure(figure_name='matriz_confusion_porcentaje', figure = plt)
plt.show()
plt.clf()
plt.close()


# In[48]:


print('Loss en entrenamiento = ', loss_list[len(loss_list)-1])
print('Loss en validacion = ', validation_loss_list[len(validation_loss_list)-1])
print('Kappa en entrenamiento = ', kappa_list[len(kappa_list)-1])
print('Kappa en validacion = ', validation_kappa_list[len(validation_kappa_list)-1])
print('Accuracy en entrenamiento = ', accuracy_list[len(accuracy_list)-1])
print('Accuracy en validacion = ', validation_accuracy_list[len(validation_accuracy_list)-1])


# In[117]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(loss_list, color = "coral", label = "train")
plt.plot(validation_loss_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n', color = 'black')
plt.ylabel('\n \n Loss', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Loss_plot', figure = plt)
plt.show()
plt.clf()
plt.close()


# In[118]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(kappa_list, color = "coral", label = "train")
plt.plot(validation_kappa_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n', color = 'black')
plt.ylabel('\n \n Kappa', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Kappa_plot', figure = plt)
plt.show()
plt.clf()
plt.close()


# In[119]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(accuracy_list, color = "coral", label = "train")
plt.plot(validation_accuracy_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n ', color = 'black')
plt.ylabel('\n \n Accuracy', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Accuracy_plot', figure = plt)
plt.show()
plt.clf()
plt.close()


# In[96]:


torch.save(best_model.state_dict(), '/home/marta/MsC/trained_models/regresion_resnet18_propio_reg_lin_Batch4_FitOneCycle.pth')


# In[ ]:


parameters_reg_lin = {"coef" : float(best_coefs[0][0]), "intercept" : float(best_intercept[0])}
  
with open('/home/marta/MsC/coefs/coefs_regresion_resnet18_propio_reg_lin_Batch4_FitOneCycle.txt', 'w') as convert_file:
     convert_file.write(json.dumps(parameters_reg_lin))


# In[97]:


experiment.log_notebook('regresion_resnet18_propio_reg_lin_Batch4_FitOneCycle.ipynb')
experiment.end()


# In[ ]:




