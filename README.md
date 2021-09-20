# TE05253 - Redes Neurais Artificiais
Prof. Dra. Adriana Rosa Garcez Castro

Equipe:
- [Diego Dandas](https://github.com/diegodantasf)
- [Gustavo Fontenele](https://github.com/gustavofont)
- [Jeremias Abreu](https://github.com/j-abreu)

#### Implementação simples do Algoritmo BackPropagation com Gradiente Descendente

O algoritmo BackProgation, também chamado apenas de BackProp, foi demonstrado pela primeira vez por [(Rumelhart et al., 1986a)](https://www.nature.com/articles/323533a0) e é, hoje em dia, amplamente utilizado para treinar redes neurais _feedforward_, porém, existem diversas versões deste algoritmo para treinar outros tipos de redes neurais artificiais.

Neste trabalho aprensentamos uma implementação em Python de uma versão simplificada do algoritmo enquanto que usamos um conjunto de dados sintético gerado pelos próprios autores deste trabalho para treino e validação de um Perceptron Multi-Camadas usando o algoritmo implementado juntamente com o algoritmo do Gradiente Descendente para a atualização do pesos da MLP baseada no erro, o qual é calculado usando a função de Mean Squared Error (erro quadrático médio).

[TODO: add fórmulas simples do gradiente descendente e da MSE]

O código fonte do algoritmo pode ser encontrado em [link](https://github.com/diegodantasf/ufpa-rna-backprop)

Os dados que usamos para treino e validação foram gerados a partir da função f(x) = x^2 + W, onde W é um ruído gaussiano.
Usamos [TODO: add numero de amostras] amostas para o treino e [TODO: add numero de amostras] para validação.
A MLP foi treinada por [TODO: add número de épocas] épocas. Logo abaixo mostramos os graficos de erro ao longo das épocas e de validação mostrando o resultado esperado e o resultado da MLP.
