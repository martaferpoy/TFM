#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import time
import timm
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# In[27]:


device


# In[ ]:


from comet_ml import Experiment


experiment = Experiment(
  api_key = "",
  project_name = "apnea-bispectrum",
  workspace=""
)
experiment.set_name("Densenet 121 Ridge Central")


# In[29]:


data_t = open('/home/marta/MsC/data/data_train.pkl', 'rb')
data_train = pickle.load(data_t)

data_v = open('/home/marta/MsC/data/data_validation.pkl', 'rb')
data_validation = pickle.load(data_v)


# In[30]:


len(data_train), len(data_validation)


# In[73]:


hiperparametros = {
    'epocas' : 50,
    'lr' : 0.001,
    'batch_size' : 32,
    'seed' : 56389856
}


# In[74]:


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        
    def __getitem__(self, index):
        name_ind = self.data[index]['name']
        x_torres_ind = self.data[index]['image thorres']
        x_abdo_ind = self.data[index]['image abdo']
        y_events_ind = self.data[index]['all events']
        y_central_ind = self.data[index]['central events']
        y_hypo_ind = self.data[index]['hypo events']
        y_osa_ind = self.data[index]['osa events']
        
        
        thorres = torch.from_numpy(np.moveaxis(x_torres_ind / 255, -1, 0))
        abdores = torch.from_numpy(np.moveaxis(x_abdo_ind / 255, -1, 0))
        
        x = torch.cat([thorres, abdores], axis = 0)
        
        y = torch.Tensor([y_central_ind]).float()

        return x, y, name_ind

    def __len__(self):
        return len(self.data)


# In[75]:


dataset_train = MyDataset(data_train)
dataset_validation = MyDataset(data_validation)


# In[76]:


dataloader_train = DataLoader(dataset_train, batch_size = hiperparametros['batch_size'], shuffle = True)
dataloader_validation = DataLoader(dataset_validation, batch_size = hiperparametros['batch_size'], shuffle = True)


# In[77]:


model = timm.create_model('densenet121', pretrained=True, num_classes = 1, in_chans = 6).to(device)


# In[78]:


def validate_loss_kappa_acc(val_loss, val_kappa, val_accuracy):
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

    kappa, accuracy = get_AHI_kappa_acc(dict_patients_val)

    val_kappa.append(kappa)
    val_accuracy.append(accuracy)
    
    experiment.log_metric('Kappa validation',kappa)
    experiment.log_metric('Accuracy validation',accuracy)
    
    return(val_loss, val_kappa, val_accuracy, loss_medio_val/pasos_val, kappa, accuracy, outputs_list_val, labels_list_val, names_list_val)


# In[79]:


criterion = torch.nn.HuberLoss(reduction= "mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=hiperparametros['lr'])


# In[80]:


def group_pacients(output_batch, labels_batch, names_batch, dict_p):
    for i in range(output_batch.shape[0]):
        output = output_batch[i]; label = labels_batch[i]; name = names_batch[i]

        if name in [*dict_p]:
            dict_p[name].append([output, label])
        else:
            dict_p[name] = [[output, label]]

    return dict_p


# In[81]:


def get_AHI_kappa_acc(dictionary):
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

    kappa = cohen_kappa_score(AHI_pred_disc, AHI_real_disc)

    accuracy = sum(AHI_pred_disc == AHI_real_disc).item()/len(AHI_real_disc)
    
    return kappa, accuracy


# In[82]:


loss_list =[]; validation_loss_list = []
kappa_list = []; validation_kappa_list = []
accuracy_list = []; validation_accuracy_list = []

loss_min = 10000000; kappa_max = -100000; accuracy_max = -1000
best_model = None; best_outputs = None; best_labels = None; best_names = None

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
    for i,data in enumerate(dataloader_train, 0): 
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

    print("Epoch = ", epoch)
    loss_list.append(loss_medio/pasos)
    
    experiment.log_metric('Loss train',loss_medio/pasos)

    dict_patients = group_pacients(outputs_list.detach().to('cpu').numpy(), labels_list.detach().to('cpu').numpy(), names_list, dict_patients)
    
    kappa, accuracy = get_AHI_kappa_acc(dict_patients)

    kappa_list.append(kappa)
    accuracy_list.append(accuracy)
    experiment.log_metric('Kappa train',kappa)
    experiment.log_metric('Accuracy train',accuracy)

    validation_loss_list, validation_kappa_list, validation_accuracy_list, loss_actual, kappa_actual, accuracy_actual, best_val_outputs, best_val_labels, best_val_names = validate_loss_kappa_acc(validation_loss_list, validation_kappa_list, validation_accuracy_list)

    # chequeo la metrica que tengo en esta epoca, y si es mejor que en epocas anteriores, guardo el modelo
    if(kappa_actual > kappa_max):
        loss_min = loss_actual
        kappa_max = kappa_actual
        accuracy_max = accuracy_actual
        best_model = copy.deepcopy(model)
        best_outputs = best_val_outputs
        best_labels = best_val_labels
        best_names = best_val_names
    
    model.train()
        
experiment.log_metric('Best loss validation', loss_min)
experiment.log_metric('Best kappa validation', kappa_max)
experiment.log_metric('Best accuracy validation', accuracy_max)

print("Time required: ", time.time() - tiempo)


# In[83]:


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


# In[84]:


dict_best_patients = {}
dict_best_patients = group_pacients(best_outputs.detach().to('cpu').numpy(), best_labels.detach().to('cpu').numpy(), best_names, dict_best_patients)


# In[85]:


AHI_pred_disc, AHI_real_disc = get_AHI(dict_best_patients)



# In[91]:

try:
    best_cfmatrix_val = confusion_matrix(AHI_real_disc, AHI_pred_disc)
    cf_mat_row = np.zeros((3, 3))
    for i in range(best_cfmatrix_val.shape[0]):
        cf_mat_row[i,:] = best_cfmatrix_val[i]/sum(best_cfmatrix_val[i])


    group_counts = ["{0:0.0f}".format(value) for value in
                    best_cfmatrix_val.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_mat_row.flatten()]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
            zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3, 3)

    plt.rcdefaults()
    sns.set(font_scale=1.4)
    sns.set(rc = {'figure.figsize':(5,4)})

    sns.heatmap(cf_mat_row, annot=labels, annot_kws={"size": 14}, fmt='',  cmap='Blues')

    plt.title("\n Matriz de confusión por clase en validación")
    plt.xlabel("Estimado \n \n \n")
    plt.ylabel("\n \n Real")
    experiment.log_figure(figure_name='matriz_confusion_porcentaje', figure = plt)
    #plt.show()
    plt.clf()
    plt.close()

except ValueError:
  print('Error Value error')
  pass

# In[92]:


print('Loss en entrenamiento = ', loss_list[len(loss_list)-1])
print('Loss en validacion = ', validation_loss_list[len(validation_loss_list)-1])
print('Kappa en entrenamiento = ', kappa_list[len(kappa_list)-1])
print('Kappa en validacion = ', validation_kappa_list[len(validation_kappa_list)-1])
print('Accuracy en entrenamiento = ', accuracy_list[len(accuracy_list)-1])
print('Accuracy en validacion = ', validation_accuracy_list[len(validation_accuracy_list)-1])


# In[93]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(loss_list, color = "coral", label = "train")
plt.plot(validation_loss_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n', color = 'black')
plt.ylabel('\n \n Loss', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Loss_plot', figure = plt)
#plt.show()
plt.clf()
plt.close()


# In[94]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(kappa_list, color = "coral", label = "train")
plt.plot(validation_kappa_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n', color = 'black')
plt.ylabel('\n \n Kappa', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Kappa_plot', figure = plt)
#plt.show()
plt.clf()
plt.close()


# In[95]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(accuracy_list, color = "coral", label = "train")
plt.plot(validation_accuracy_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n \n ', color = 'black')
plt.ylabel('\n \n Accuracy', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
experiment.log_figure(figure_name='Accuracy_plot', figure = plt)
#plt.show()
plt.clf()
plt.close()


# In[96]:


torch.save(best_model.state_dict(), '/home/marta/MsC/trained_models/densenet121_ridge_central.pth')


# In[97]:


experiment.log_notebook('densenet121_ridge_central.ipynb')
experiment.end()


# In[ ]:




