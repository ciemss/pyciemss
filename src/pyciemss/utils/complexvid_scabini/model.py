# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:12:38 2020

@author: scabini
"""

import networkx as nx
# import model as md
import numpy as np
from comunities import *
from random import choices

def createGraph(parameters):
    np.random.seed(parameters['seed'])
    random.seed(parameters['seed'])
    G = nx.Graph()
    G.add_nodes_from(range(0,parameters['n_nodes']))
    
    ####distribuicoes estatisticas para modelar os nos (fonte: IBGE 2019)
   
    age_dist = parameters['age_dist']
    
    node_ages = choices([0,1,2,3,4,5], age_dist[:], k= parameters['n_nodes'])
    
    node_state = []
    for i in range(0,parameters['n_nodes']):
        node_state.append(0) #id de estado 0 = 'susceptible'
        
    states = dict(zip(list(G.nodes),node_state))
    ages = dict(zip(list(G.nodes),node_ages))
    days_infected = dict(zip(list(G.nodes),np.zeros((G.order()))))
    
    nx.set_node_attributes(G, ages, 'age') #add o atributo age aos nos (index da faixa etaria)
    nx.set_node_attributes(G, states, 'condition') #estado inicial de todos nodes eh suscetivel
    nx.set_node_attributes(G, days_infected, 'days infected') #dias desde que o nó ta infectado
    
    for layer in parameters['layers_0']:
        G = include_layer(G, layer, parameters)
    
    return G

#remove camadas. "layer" é um inteiro entre 1 e qtde de camadas
def remove_layer(G, layer):
    # print("Removendo camada: ", layer)
    layers = {'casas':1, 'escolas':2, 'igrejas':3, 'transporte':4, 'aleatorio':5, 'trabalho':6} 
    layer = layers[layer]    
    edges = G.edges()    
    for edge in edges:
        a=edges[edge[0], edge[1]]
        if a['layer'] == layer:
            G.remove_edge(edge[0], edge[1])
    return G

def change_layer(G, layer, new_weight):
    # print("Alterando camada: ", layer)
    layers = {'casas':1, 'escolas':2, 'igrejas':3, 'transporte':4, 'aleatorio':5, 'trabalho':6} 
    layer = layers[layer]    
    edges = G.edges()    
    for edge in edges:
        a=edges[edge[0], edge[1]]
        if a['layer'] == layer:
            G[edge[0]][edge[1]]['weight'] = new_weight
            
    return G

def scale_layer(G, layer, scale):
    # print("Alterando camada: ", layer)
    layers = {'casas':1, 'escolas':2, 'igrejas':3, 'transporte':4, 'aleatorio':5, 'trabalho':6} 
    layer = layers[layer]    
    edges = G.edges()    
    for edge in edges:
        a=edges[edge[0], edge[1]]
        if a['layer'] == layer:
            G[edge[0]][edge[1]]['weight'] = G[edge[0]][edge[1]]['weight']*scale
            
    return G

def include_layer(G, layer, parameters):
    layers = {'casas':1, 'escolas':2, 'igrejas':3, 'transporte':4, 'aleatorio':5, 'trabalho':6}
    # print("Incluindo camada: ", layer)
    if layers[layer]==1:
        G = generate_layer1(G, parameters) #casas
    elif layers[layer]==2:
        G = generate_layer_school(G, parameters) #escolas
    elif layers[layer]==3:
        G = generate_layer_church(G, parameters) #templos
    elif layers[layer]==4:
        G = generate_layer_transport(G, parameters) #transporte 
    elif layers[layer]==5:
        G = generate_layer_random(G, parameters) #aleatorio  
    elif layers[layer]==6:
        G = generate_layer_work(G, parameters) #trabalho   
            
    return G

#isolamento pra sintomas leves/moderados = fica só em casa
def isolate_node(G, node, how='hospital - total'):
    #how pode ser 'no', no caso nao faz nada e retorna o grafo original
    
    edges = G.edges(node)
    edges = [i for i in edges]
    alledges = G.edges()
    #caso otimista do hospital= isola totalmente = remove todas arestas
    #essa funcao tbm é usada pra remover recuperados e mortos da rede
    if 'hospital - total' in how:
        for edge in edges:
            G.remove_edge(edge[0], edge[1])
    
    #isolamento em casa otimista, isola totamente em casa= remove todas camadas menos a 1
    elif 'home - total' in how:
        for edge in edges:
            a=alledges[edge]            
            if a['layer'] != 1:
                G.remove_edge(edge[0], edge[1])
                
    #isolamento em casa pessimista, matem conexoes em casa e aleatorias      
    elif 'home - partial' in how:
        for edge in edges:
            a=alledges[edge]            
            if a['layer'] != 1 and a['layer'] != 5:
                G.remove_edge(edge[0], edge[1])
    
    #caso pessimista do hospital, mantem arestas aleatorias
    elif 'hospital - partial' in how:
        for edge in edges:
            a=alledges[edge]  
            if a['layer'] != 5:
                G.remove_edge(edge[0], edge[1])
    return G



















    
    