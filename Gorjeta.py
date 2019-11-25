import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#
#
# Função para traçar fuzzy sets
def plotmf(universe, names, mf, xticks, xlabel):
    plt.figure()
    for name in names:
        plt.plot(universe, 100*mf[name].mf, label=name)
    plt.legend() 
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Pertinência (%)')
    plt.xticks(xticks)
    plt.show()
    
#
# Função para fuzzyficação
def myfuzzification(universe, names, mf, val_in):
    val_fuzz = {}
    for name in names:
        val_fuzz[name] = fuzz.interp_membership(universe, mf[name].mf, val_in)
    return val_fuzz

#
# Variáveis fuzzy de entrada
#
# Universo da qualiade [0 a 10]
uni_qualidade = np.arange(0, 10.1, 0.1)
xticks_qual = np.arange(0, 11, 1)
#
# Comida
nome_comida = ['ruim', 'boa']
comida = ctrl.Antecedent(uni_qualidade, 'comida')
comida['ruim'] = fuzz.trimf(uni_qualidade, [0, 0, 10])
comida['boa'] = fuzz.trimf(uni_qualidade, [0, 10, 10])
plotmf(uni_qualidade, nome_comida, comida, xticks_qual, 'Qualidade da comida')
#
# Serviço
nome_servico = ['ruim', 'médio', 'bom' ]
servico = ctrl.Antecedent(uni_qualidade, 'serviço')
servico.automf(names = nome_servico)
plotmf(uni_qualidade, nome_servico, servico, xticks_qual, 'Qualidade do serviço')


# Variável fuzzy de saída
#
# Universo da gorjeta [5 a 20]
uni_gorjeta = np.arange(0, 12.1, 0.1)
xtick_gorjeta = np.arange(0, 13, 1)
#
# Gorjeta
nome_gorjeta = ['baixa', 'média', 'alta']
gorjeta = ctrl.Consequent(uni_gorjeta, 'gorjeta')
gorjeta.automf(names = nome_gorjeta)
plotmf(uni_gorjeta, nome_gorjeta, gorjeta, xtick_gorjeta, 'Gorjeta (%)')

##
## Regras fuzzy
#rule1 = ctrl.Rule(servico['ruim'] | comida['ruim'], gorjeta['baixa'])
#rule2 = ctrl.Rule(servico['médio'], gorjeta['média'])
#rule3 = ctrl.Rule(servico['bom'] & comida['boa'], gorjeta['alta'])

#
# Valore de teste
comida_in = 5
servico_in = 7

#
# Hard way

# Fuzzyficação
# comida
comida_fuzz = myfuzzification(uni_qualidade, nome_comida, comida, comida_in)
# serviço
servico_fuzz = myfuzzification(uni_qualidade, nome_servico, servico, servico_in)

# Inferência (regras)
#
# Regra 1 - Se comida ruim OU serviço ruim, então gorjeta baixa
# usa np.fmax para OU e np.fmin para E
rule1 = np.fmax(comida_fuzz['ruim'], servico_fuzz['ruim'])
gorjeta_baixa = np.fmin(rule1, gorjeta['baixa'].mf)
#
# Regra 2 - Se serviço médio, então gorjeta média
# usa np.fmax para OU e np.fmin para E
rule2 = servico_fuzz['médio']
gorjeta_media = np.fmin(rule2, gorjeta['média'].mf)
#
# Regra 3 - Se serviço médio, então gorjeta média
# usa np.fmax para OU e np.fmin para E
rule3 = np.fmin(comida_fuzz['boa'], servico_fuzz['bom'])
gorjeta_alta = np.fmin(rule3, gorjeta['alta'].mf)

#
# Gráfico gorjeta
uni_gorjeta0 = np.zeros_like(uni_gorjeta)
fig, ax0 = plt.subplots()
#plt.grid()
ax0.fill_between(uni_gorjeta, uni_gorjeta0, 100*gorjeta_baixa, facecolor='b', alpha=0.7)
ax0.plot(uni_gorjeta, 100*gorjeta['baixa'].mf, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(uni_gorjeta, uni_gorjeta0, 100*gorjeta_media, facecolor='g', alpha=0.7)
ax0.plot(uni_gorjeta, 100*gorjeta['média'].mf, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(uni_gorjeta, uni_gorjeta0, 100*gorjeta_alta, facecolor='r', alpha=0.7)
ax0.plot(uni_gorjeta, 100*gorjeta['alta'].mf, 'r', linewidth=0.5, linestyle='--')
plt.ylabel('Pertinência (%)')
plt.xlabel('Gorjeta (%)')
plt.show()

#
# Defuzzificação
 
# Agregando reulstados das inferências (preenchimento)
# Para virar um gráfico só 
aggregated = np.fmax(gorjeta_baixa,
                     np.fmax(gorjeta_media, gorjeta_alta))


# Figura da agregação
fig, ax0 = plt.subplots()

ax0.plot(uni_gorjeta, 100*gorjeta['baixa'].mf, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(uni_gorjeta, 100*gorjeta['média'].mf, 'g', linewidth=0.5, linestyle='--')
ax0.plot(uni_gorjeta, 100*gorjeta['alta'].mf, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(uni_gorjeta, uni_gorjeta0, 100*aggregated, facecolor='Orange', alpha=0.7)
#ax0.plot([gorjeta_out, gorjeta_out], [0, 100*gorjeta_act], 'k', linewidth=1.5, alpha=0.9)
plt.ylabel('Pertinência (%)')
plt.xlabel('Gorjeta (%)')
plt.show()

print('Qualidade da comida:')
print(comida_in)
print('Qualidade do serviço:')
print(servico_in)

# Calculo do centróida (deffuzificação)
gorjeta_out = fuzz.defuzz(uni_gorjeta, aggregated, 'centroid')  # Cálculo do centróide
gorjeta_act = fuzz.interp_membership(uni_gorjeta, aggregated, gorjeta_out)  # for plot

print('Gorjeta (%): ')
print(gorjeta_out)


