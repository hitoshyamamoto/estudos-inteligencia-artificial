import random
import numpy
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
import matplotlib.pyplot as plt

class Produto():
    def __init__(self, nome, espaço, valor):
        self.nome = nome
        self.espaço = espaço
        self.valor = valor
        
listaProdutos = []
listaProdutos.append(Produto("Geladeira Dako",0.751,999.90))
listaProdutos.append(Produto("Iphone 6",0.8000899,2911.12))
listaProdutos.append(Produto("TV 55' ",0.480,4346.99))
listaProdutos.append(Produto("TV 50' ",0.290,3999.90))
listaProdutos.append(Produto("TV 42' ",0.200,2999.00))
listaProdutos.append(Produto("Notebook Dell",0.00350,2499.90))
listaProdutos.append(Produto("Ventilador Panasonic",0.496,199.90))
listaProdutos.append(Produto("Microondas Eletrolux",0.0424,308.66))
listaProdutos.append(Produto("Microondas LG",0.0544,429.90))
listaProdutos.append(Produto("Microondas Panasonic",0.319,299.29))
listaProdutos.append(Produto("Geladeira Brastemp",0.635,849.00))
listaProdutos.append(Produto("Geladeira Consul",0.870,1199.89))
listaProdutos.append(Produto("Notebook Lenovo",0.498,1999.90))
listaProdutos.append(Produto("Notebook Asus",0.527,3999.00))

espaços  = []
valores = []
nomes   = []

for produto in listaProdutos:
    espaços.append(produto.espaço)
    valores.append(produto.espaço)
    nomes.append(produto.nome)
limite = 3

toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=len(espaços))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def avaliação(individual):
    nota = 0
    somaEspaços = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            nota += valores[i]
            somaEspaços += espaços[i]
    if somaEspaços > limite:
        nota = 1
    return nota/100000,

toolbox.register("evaluate", avaliação)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate",tools.mutFlipBit, indpb = 0.01)
toolbox.register("select",tools.selRoulette)

if __name__ == "__main__":
    random.seed(1)
    população = toolbox.population(n=50)
    probabilidadeCrossover = 1.0
    probabilidadeMutação = 0.01
    numeroGerações = 200
    
    estatisticas = tools.Statistics(key=lambda individuo: individuo.fitness.values)
    estatisticas.register("max", numpy.max)
    estatisticas.register("min", numpy.min)
    estatisticas.register("med", numpy.mean)
    estatisticas.register("std", numpy.std)
    
    população, info = algorithms.eaSimple(população, toolbox,
                                          probabilidadeCrossover,
                                          probabilidadeMutação,
                                          numeroGerações, estatisticas)
    melhores = tools.selBest(população, 1)
    for individuo in melhores:
        print(individuo)
        print(individuo.fitness)
        soma=0
        for i in range(len(listaProdutos)):
            if individuo[i] == 1:
                soma += listaProdutos[i].valor
                print("Nome: %s R$ %s " % (listaProdutos[i].nome,
                                           listaProdutos[i].valor))
        print("Melhor solução: %s" % soma)
    
    valoresGrafico = info.select("max")
    plt.plot(valoresGrafico)
    plt.title("Acompanhamento dos valores")
    plt.show()




