# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:05:34 2020

@author: scabini
"""
import networkx as nx
import numpy as np
from collections import Counter
import random
from random import choices
from random import uniform
# import model 

#gambiarra pra ajustar casos em que o rng for ruim
def check_comunities(comunities, n):    
    chksum=0
    for comunity in comunities:
        chksum+= comunity+1        
    if chksum != n:
        diff = chksum - n        
        if diff < 0:
            for i in range(0, np.absolute(diff)):
                comunities.append(0)
        else:
            return False   
              
    return comunities

#layer 1 - vizinhança - grupos com pelo menos 1 adulto e tamanhos de acordo com
# a distribuiçao de tamanhos de familias do brazil
def generate_layer1(G, parameters): 
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    ############### DADOS G1 2010
    fam_structure = parameters['fam_structure']
   
    prob_infection = parameters['prob_home']
    n_fam = int(np.round(parameters['n_nodes'] * 0.30)) #30%=razao numero de familias/população (estimado de ibge/g1 2010)

    comunities = False
    while comunities==False:
        comunities = choices(population=[0,1,2,3,4,5,6,7,8,9], weights=fam_structure[:], k=n_fam)
        comunities = check_comunities(comunities, parameters['n_nodes'])
        
    if parameters['verbose']:
        print("Quantidade de familias por tamanho: ", (Counter([i+1 for i in comunities])))
   
    nodes = list(G.nodes) #lista de nós

    ages = nx.get_node_attributes(G, 'age') #lista de nos com faixa etaria
    nodes_adults = []
    
    #selecionando os nos com id de faixa etaria >= 2 (adultos)
    i=0
    while len(nodes_adults)!=len(comunities):
        if ages[i]>=2:
            nodes_adults.append(i)
            nodes.remove(i)
        i+=1
            
    # nodes_adults = nodes_adults[0:len(comunities)]
    avg_size = 0
    localss = [0] * G.number_of_nodes()
   
    local = 0
   
    qtde_layers =  (len(parameters['layers_0']))
    variable_prob = ((6-qtde_layers)*0.2)
    prob = ((1*(3*7/(24*7)))*prob_infection) 
    prob = prob+(prob*variable_prob)
    for comunity in comunities: #para cada comunidade...   
        #calculo da probabilidade de infeccao, depende do tamanho, tempo de
        #interacao e qtde media de pessoas proximas (detalhes na planilha)

        nodes_to_conect = []
        father = nodes_adults.pop()
        nodes_to_conect.append(father) #seleciona 1 adulto
        for size in range(1,comunity+1): #cria a comunidade de acordo com seu tamanho            
            if nodes: #enquanto ainda restarem nós...                                
                random_node = nodes.pop()
                nodes_to_conect.append(random_node) #seleciona nó aleatório
                localss[random_node] = local
            else:
                break
                
        #conecta todos com todos do grupo nodes_to_conect, peso da aresta=prob de infecção
        #essa é a comunidade/familia 
        avg_size+=len(nodes_to_conect)
        for i in range(0, len(nodes_to_conect)):
            for j in range(i+1, len(nodes_to_conect)):
                G.add_edge(nodes_to_conect[i], nodes_to_conect[j], weight=prob, layer=1)
                
        local+=1
   
    if parameters['verbose']:
        print('Tamanho medio de familia: ', avg_size/len(comunities))
        
    localss =  dict(zip(list(G.nodes),localss))
    nx.set_node_attributes(G, localss, 'local')
    return G

#layer 2 - escolas - grupos de pessoas de idade (0-17) de tamanhos de
# acordo com o tamanho medio de salas de aula no país
def generate_layer_school(G, parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    #intervalos para escolher uniformemente durante criaçao    
    # prob_infection = [0.13, 0.25] #prob de infecção varia de acordo com variação de tamanhos
    tamanhos = [16, 30] #tamanho de sala varia de acordo com estatisticas ensino fundamental>medio
        
    prob_infection = parameters['prob_school']
    ages = nx.get_node_attributes(G, 'age') #lista de nos com faixa etaria
    nodes = []
    #selecionando os nos com id de faixa etaria <= 1 (0-17)
    for i in range(0,len(ages)):
        if ages[i]<=1:
            nodes.append(i)
            
    random.shuffle(nodes)
    n_salas = 0
        
    avg_size = 0    
    while nodes:
        n_salas+=1        
        nodes_to_conect = []
        
        size = random.randint(tamanhos[0], tamanhos[1])
        
        #calculo da probabilidade de infeccao, depende do tamanho, tempo de
        #interacao e qtde media de pessoas proximas (detalhes na planilha)
        prob = (5/(size))*((4*5)/(24*7))*prob_infection         
        
        for i in range(0,size): #cria a comunidade de acordo com seu tamanho
            if nodes: #enquanto ainda restarem nós...
                #como as idades sao determinadas aleatoriamente, pegar o ultimo no tbm fica aleatorio
                random_node = nodes.pop() 
                nodes_to_conect.append(random_node) #seleciona nó aleatório de nodes
            else:
                break
                
        #conecta todos com todos do grupo nodes_to_conect, peso da aresta=prob de infecção
        #essa é a comunidade 
        avg_size+=len(nodes_to_conect)
        for i in range(0, len(nodes_to_conect)):
            for j in range(i+1, len(nodes_to_conect)):            
                G.add_edge(nodes_to_conect[i], nodes_to_conect[j], weight=prob, layer=2)
                
    if parameters['verbose']:                
        print('Qtde salas:', n_salas, ' - Tamanho medio: ', avg_size/(n_salas))
    
    return G


#layer 3 - atividades religiosas/templos
def generate_layer_church(G, parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    #intervalos para escolher uniformemente durante criaçao    
    prob_religiosos = parameters['qtde_religiao'] #metade da populacao de cristaos em 2019, fonte: Veja
    prob_religiosos = int(np.round(prob_religiosos*G.order()))
    
    tamanhos = [(10,50), (51,80), (81,100)] #intervalo de tamanhos de templos/igrejas ou grupos de pessoas que os frequentam             
    prob_sizes = [0.552786405, 0.292858506, 0.15435509]
    
    # tamanhos= [(50,100), (101,200), (201,300), (301,400), (401,500), (501,1000), (1001,5000), (5001,10000), (10001,50000), (50001,100000)]
    # prob_sizes = [0.5527864045, 0.2472135955, 0.0750908150, 0.0354664659, 0.0204130244, 0.0381586767, 0.0261059192, 0.0026340818, 0.0018020836, 0.0001818299]
    
    prob_infection = parameters['prob_religion']

    nodes = list(G.nodes)
    random.shuffle(nodes)
    nodes = nodes[0:prob_religiosos]
    n_templos = 0
        
    avg_size = 0    
    while nodes:
        n_templos+=1        
        nodes_to_conect = []
        
        size = random.choices(tamanhos, prob_sizes, k=1)
        size = random.randint(size[0][0], size[0][1])
        
        #calculo da probabilidade de infeccao, depende do tamanho, tempo de
        #interacao e qtde media de pessoas proximas
        prob = (6/(size))*(2/(24*7))*prob_infection

        
        for i in range(0,size): #cria a comunidade de acordo com seu tamanho
            if nodes: #enquanto ainda restarem nós...
                random_node = nodes.pop() 
                nodes_to_conect.append(random_node) #seleciona nó aleatório de nodes
            else:
                break
                
        #conecta todos com todos do grupo nodes_to_conect, peso da aresta=prob de infecção
        #essa é a comunidade 
        avg_size+=len(nodes_to_conect)
        for i in range(0, len(nodes_to_conect)):
            for j in range(i+1, len(nodes_to_conect)):              
                G.add_edge(nodes_to_conect[i], nodes_to_conect[j], weight=prob, layer=3)
    
    if parameters['verbose']:                
        print('Qtde de templos:', n_templos, ' - Tamanho medio: ', avg_size/(n_templos))
    
    return G



#layer 4 - transporte público
def generate_layer_transport(G, parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    #intervalos para escolher uniformemente durante criaçao    
    prob_tpublico = parameters['qtde_transporte'] #fracao da população que utiliza transporte publico
    prob_tpublico = int(np.round(prob_tpublico*G.order()))
    tamanhos = [10, 40] #intervalo de tamanhos/qtde de pessoas que viajam em onibus/metro         
    prob_infection = parameters['prob_transport']

    nodes = list(G.nodes)
    random.shuffle(nodes)
    nodes = nodes[0:prob_tpublico]
    n_v = 0
        
    avg_size = 0    
    while nodes:
        n_v+=1        
        nodes_to_conect = []
        
        size = random.randint(tamanhos[0], tamanhos[1])

        #calculo da probabilidade de infeccao, depende do tamanho, tempo de
        #interacao e qtde media de pessoas proximas
        prob = (8/(size))*((parameters['tempo_transporte']*7)/(24*7))*prob_infection
        
        for i in range(0,size): #cria a comunidade de acordo com seu tamanho
            if nodes: #enquanto ainda restarem nós...
                random_node = nodes.pop() 
                nodes_to_conect.append(random_node) #seleciona nó aleatório de nodes
            else:
                break
                
        #conecta todos com todos do grupo nodes_to_conect, peso da aresta=prob de infecção
        #essa é a comunidade 
        avg_size+=len(nodes_to_conect)
        for i in range(0, len(nodes_to_conect)):
            for j in range(i+1, len(nodes_to_conect)):
                G.add_edge(nodes_to_conect[i], nodes_to_conect[j], weight=prob, layer=4)
                
    if parameters['verbose']:                
        print('Qtde de veiculos:', n_v, ' - Tamanho medio: ', avg_size/(n_v))
    
    return G


#layer 5 - encontros sociais aleatorios, superficies, e todas as outras coisas
def generate_layer_random(G, parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    #intervalos para escolher uniformemente durante criaçao    
    rand_interaction = [1, 10] #intervalo de interacoes sociais aleatorias que as pessoas podem ter    
    qtde_interactions = int(np.round(G.order()*(sum(rand_interaction)/2))) #media

    prob_infection = parameters['prob_random']
    #calculo da probabilidade de infeccao, depende do tamanho, tempo de
    #interacao e qtde media de pessoas proximas
    prob = (1*((1/(24*7))))*prob_infection
        
    nodes = list(G.nodes)
    random.shuffle(nodes)
    
    
     
    for i in range(0, qtde_interactions):
        source = random.choice(nodes)
        target = random.choice(nodes)
        G.add_edge(source, target, weight=prob, layer=5)
        
    if parameters['verbose']: 
        print('Qtde de links aleatorios:', qtde_interactions)
    
    return G



#layer 6 - trabalho - grupos de pessoas de idade de trabalho, tamanhos dos
# grupos é uma aleatorio [5, 50]
def generate_layer_work(G, parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])    
    #intervalos para escolher uniformemente durante criaçao    
    # prob_infection = [0.13, 0.25] #prob de infecção varia de acordo com variação de tamanhos
    tamanhos = [5, 30] #tamanho de sala varia de acordo com estatisticas ensino fundamental>medio
        
    prob_infection = parameters['prob_work']
    ages = nx.get_node_attributes(G, 'age') #lista de nos com faixa etaria
    nodes = []
    #selecionando os nos com id de faixa etaria <= 1 (0-17)
    for i in range(0,len(ages)):
        if ages[i]>=2 and ages[i]<=4:
            nodes.append(i)
            
    random.shuffle(nodes)
    n_empresas = 0
        
    avg_size = 0    
    while nodes:
        n_empresas+=1        
        nodes_to_conect = []
        
        size = random.randint(tamanhos[0], tamanhos[1])
        
        #calculo da probabilidade de infeccao, depende do tamanho, tempo de
        #interacao e qtde media de pessoas proximas (detalhes na planilha)
        prob = (3/(size))*((8*5)/(24*7))*prob_infection         
        
        for i in range(0,size): #cria a comunidade de acordo com seu tamanho
            if nodes: #enquanto ainda restarem nós...
                #como as idades sao determinadas aleatoriamente, pegar o ultimo no tbm fica aleatorio
                random_node = nodes.pop() 
                nodes_to_conect.append(random_node) #seleciona nó aleatório de nodes
            else:
                break
                
        #conecta todos com todos do grupo nodes_to_conect, peso da aresta=prob de infecção
        #essa é a comunidade 
        avg_size+=len(nodes_to_conect)
        for i in range(0, len(nodes_to_conect)):
            for j in range(i+1, len(nodes_to_conect)):            
                G.add_edge(nodes_to_conect[i], nodes_to_conect[j], weight=prob, layer=6)
                
    if parameters['verbose']:                
        print('Qtde empresas:', n_empresas, ' - Tamanho medio: ', avg_size/(n_empresas))
    
    return G


















