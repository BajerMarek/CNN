import numpy as np
#! funkce která rozdělí z jednoho souboru data do dvou složek v poměru 1:4
def Rozděl_data(data_path:str,
                label_path:str):
    """funkce která rozdělí z jednoho souboru.npy data do dvou složek v poměru 1:4.
        Data by mněli mít shape(celkový_pocet_kusu, výška, šířka)
        Argumenty:
            class_names: list 
            data_path: cesta která vede k souboru s daty !! soubor musí mít koncovku .npy!!
            label_path: cesta která vede k souboru kde je řešení k datům !! soubor musí mít koncovku .npy a být ve formátu one-hot-vector!!"""
    data = np.load(data_path)
    label =np.load(label_path)
    celkovy_pocet_dat = data.shape[0]
    train_pocet = celkovy_pocet_dat/100*20
    test_pocet = celkovy_pocet_dat/100*80
    pocet_class = len(label[0])
    list_class_listu =[]

    for i in range(pocet_class):        #? mam list ve do kterého budu ukládat hodnoty dokud se nebude rovnat 96000
        list_class_listu.append([])
    """"omezana_data = data[:100]


for index in range(len(omezana_data)):
   if label[index][0] == 1.0:
    label_data[0].append([index])
    print(index)
print(label[:10])
print(label_data)"""

    

    
    