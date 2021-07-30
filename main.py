#Tarefa:
#Disciplina:Princípios de Modelagem Matemática - CEFET MG
#Objetivo:
#Assimilar o conceito de caminhadas aleatórias e implementação do método de Monte Carlo.

#imports de bibliotecas Python
import numpy
import numpy as np

# Implementação do gerador de números aleatórios congruencial
#Fonte: https://stackoverflow.com/questions/19140589/linear-congruential-generator-in-python
from matplotlib import pyplot as plt

def set_seed(s=123456789):
    global seed
    seed = s

set_seed()

def lcg(m=2**32+1, a=1103515245, c=12345):
    global seed
    seed = (a*seed + c) % m
    return seed


# Parâmetros randomwalk

#Numero de caminhadas
walk_n = 10
#número de passos
step_n = 10000
#Dimensão do passo (posição / tempo)
step_set = [-1,1]
#origem
origin = 0


# Simulação Random Walk 1 D comPython
#Fonte: https://www.youtube.com/watch?v=hqSnruUe3tA
#Fonte: https://towardsdatascience.com/random-walks-with-python-8420981bc4bc

def random_walk(step_n, step_set, origin):
    steps = np.zeros((step_n,), dtype=int)
    steps = np.insert(steps, 0, origin, axis=0)
    for step in range(1, step_n + 1):
        direction = lcg() % 2
        steps[step] = steps[step - 1] + step_set[direction]

    return steps

# Impressão dos gráficos simulação Random Walk

walkers = np.zeros((walk_n,step_n+1))
for i in range(0,walk_n):
    walkers[i] = random_walk(step_n,step_set,origin)

plt.rcParams['figure.figsize'] = [25, 10]
plt.title("Random Walk 1D")
plt.xlabel("Tempo")
plt.ylabel("Posição")

for steps in walkers:
    plt.plot(np.arange(0, step_n+1), steps, color=numpy.random.rand(3,))

plt.show()


#Calculo do desvio quadrático médio (Mean Square Displacement)
#Fonte: https://www.youtube.com/watch?v=fWFVXEXwIBQ

def msd(tau, path):
    if tau == 0: return 0

    displacement_sum = 0
    for t in range(0, len(path) - tau, tau):
        displacement_sum += (path[t + tau] - path[t]) ** 2
    return displacement_sum / (len(path) / tau)


# Ajuste de curva e impressão
#Calcule o desvio quadrático médio (<R^2>) para as 10 amostras e represente graficamente em escala log-log  <R^2>

tau_range = 100
msds = np.zeros((walk_n, tau_range), dtype=float)

for i in range(0, walk_n):
    for tau in range(0, tau_range):
        msds[i][tau] = msd(tau, walkers[i])

plt.rcParams['figure.figsize'] = [25, 10]
plt.title("Representação Gráfica (log-log)")

plt.xlabel("Tau values (s)")
plt.xscale("log")
plt.ylabel("MDS")
plt.yscale("log")

for i in range(0, walk_n):
    plt.plot(np.arange(0, tau_range), msds[i], color=numpy.random.rand(3, ), label=f"walker {i}")

plt.legend(loc="upper left")
plt.show()

# Ajuste a curva encontrada com uma lei de potência (alométrica)

msd_mean = np.mean(msds, axis=0)

plt.rcParams['figure.figsize'] = [25, 10]
plt.title("Ajuste de curva  (log-log)")

plt.xlabel("Valores (s)")
plt.xscale("log")
plt.ylabel("MDS")
plt.yscale("log")

plt.plot(np.arange(0, tau_range), msds[i], color=numpy.random.rand(3,))

plt.show()

#Teste de hipóteses com o valor obtido para o expoente da lei de potência
#Fonte: https://ichi.pro/pt/ajuste-basico-de-curva-de-dados-cientificos-com-python-164454906602761

from scipy.optimize import curve_fit

# Função para cálculo da lei de potência
def power_law(x, a, b):
    return a*np.power(x, b)

pars, cov = curve_fit(f=power_law, xdata=np.arange(0, tau_range), ydata=msd_mean, p0=[1, 5])
print(f"a: {pars[0]}, b: {pars[1]}")
print(f"y = {pars[0]}*x^{pars[1]}")

# Impressão dos gráficos

plt.rcParams['figure.figsize'] = [25, 10]
plt.title("Lei de potência - Gráfico (log-log)")

plt.xlabel("Valores (s)")
plt.xscale("log")
plt.ylabel("MDS")
plt.yscale("log")

x_data = np.arange(0, tau_range)
y_data = [power_law(x,pars[0],pars[1]) for x in x_data]

plt.scatter(np.arange(0, tau_range), msds[i], color='b')
plt.plot(x_data, y_data, color='r', label="y = a*x^b")

plt.legend(loc="upper left")
plt.show()

# Teorema Central do Limite
#Fonte:https://towardsdatascience.com/central-limit-theorem-explained-with-python-code-230884d40ce0

new_walk_n = 1000

walkers = np.zeros((new_walk_n,step_n+1))
for i in range(0,new_walk_n):
    walkers[i] = random_walk(step_n,step_set,origin)

final_positions = walkers[:,step_n]


from scipy.stats import norm

(mu, sigma) = norm.fit(final_positions)

n, bins, patches = plt.hist(final_positions, 100, density=1)

y = norm.pdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r', linewidth=2)

#Impressões Gráficas
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)

plt.show()

#Fim