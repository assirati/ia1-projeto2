# Banco Público de Perguntas — Arguição

Vinte e seis perguntas. **Duas serão sorteadas** para cada aluno no dia.
A terceira será improvisada a partir do seu pôster e não está aqui. Estudem.

Várias perguntas pedem que vocês olhem para os **seus** números. Tenham o
pôster e o notebook à mão.

---

## Módulo 1 — Aprendizado de máquina

**1.1** Por que o projeto separa treino, validação e teste? O que daria
errado se vocês escolhessem a configuração da submissão olhando para o
mesmo período em que ela vai ser avaliada?

**1.2** A sua série tem tendência de crescimento. Explique por que o k-NN
não consegue prever um nível de demanda que nunca apareceu no treino, e
aponte nos seus dados onde isso aparece.

**1.3** No k-NN, o que acontece com o erro no treino e o erro na validação
quando k vai de 1 a 25? Relacione com viés e variância, usando os seus
números.

**1.4** A regressão linear da sua série tem viés diferente de zero na
validação. Qual o sinal, quanto vale, e qual mecanismo produz esse viés?

**1.5** O que a penalização l2 (ridge) faz com os coeficientes da
regressão linear? Em que situação ela melhora a validação, e por que
nesta série ela ajudou pouco ou muito?

**1.6** Vocês compararam uma divisão aleatória com uma divisão temporal
dentro do treino. Qual estimou melhor o erro real na validação? Por que,
em série temporal, as duas podem dar respostas diferentes?

**1.7** Um classificador que sempre responde "não precisa de doca extra"
tem acurácia alta na sua série. Quanto, e por que isso não o torna bom?
Que métrica vocês usariam no lugar, e por quê?

**1.8** A regressão logística devolve uma probabilidade. Como vocês
escolheriam o limiar de decisão para abrir a doca extra, se abrir sem
necessidade custa menos que deixar de abrir quando precisa?

---

## Módulo 2 — Redes neurais

**2.1** Uma rede com 16 neurônios ocultos e ativação identidade teve
praticamente o mesmo erro que a regressão linear. Mostre algebricamente
por que isso acontece.

**2.2** Escreva o gradiente da perda em relação a um peso da primeira
camada de uma rede com uma camada oculta, para a ativação e a perda que
vocês usaram. Aponte o trecho de `engine/rede.py` que calcula cada
termo.

**2.3** O que `checar_gradiente` compara? Em que situação o erro relativo
fica grande mesmo com o backpropagation correto? Construa um exemplo com
ReLU ou com perda MAE.

**2.4** Com 4 camadas de sigmoide, a norma do gradiente da primeira camada
ficou muito menor que a da última. Explique o mecanismo, com um limite
numérico para o fator que cada camada multiplica.

**2.5** O que é uma ReLU morta? Em que condições de treino vocês viram
neurônios mortos, e por que um neurônio morto não volta a aprender?

**2.6** A perda MSE e a perda MAE levam a previsões diferentes para a
mesma série. O que cada uma estima (média, mediana, outro quantil), e
qual combina com a métrica WAPE?

**2.7** O gradiente da perda pinball no quantil 0,8 não é simétrico.
Escreva o gradiente e explique por que a rede treinada com essa perda
passa a prever acima da média.

**2.8** Com SGD e taxa de aprendizado alta, a rede divergiu. Descreva o
que acontece com os pesos passo a passo, e por que normalizar a demanda
muda a taxa a partir da qual isso ocorre.

**2.9** O perceptron não parou de errar na doca extra, mesmo depois de 50
épocas. O que isso diz sobre os dados? O que a regressão logística faz
de diferente para chegar a uma resposta estável?

**2.10** Com `inicializacao="zeros"`, a rede termina o treino prevendo o
mesmo valor para todos os dias. Que valor é esse, e por que os pesos
nunca saem do zero? Use a regra da cadeia.

---

## Placar da turma

**3.1** Sua submissão teve desempenho diferente na sua série e na média
das 30 séries do placar. Quantifique a diferença e explique a causa.

**3.2** O erro da sua submissão no teste foi maior que na validação.
Separe as causas: o que é do período (Black Friday, Natal), o que é do
modelo, o que é acaso das sementes. Como vocês distinguiriam uma da outra?

**3.3** Escolha uma submissão de outro grupo que superou a sua. O que ela
captura que a sua não captura? Aponte a feature ou a configuração.

**3.4** O placar de Computação ordena por WAPE e o de Produção por custo
de estoque. Uma submissão pode ficar bem num e mal no outro? Dê o
mecanismo e, se houver, o exemplo da tabela.

---

## Transversais

**T.1** Aponte, no seu pôster, o resultado que mais contrariou sua
predição. O que você faria diferente com mais uma semana?

**T.2** Descreva o erro de IA generativa que vocês documentaram no anexo.
Como foi detectado, e o que teria acontecido se não fosse?

**T.3** Uma afirmação do seu pôster que os dados brutos sustentam menos do
que parece. Qual é, e o que faltaria medir?

**T.4** Treinar a mesma rede com sementes diferentes deu WAPEs
diferentes. Quanto variou nos seus dados, e quantas sementes seriam
necessárias para afirmar que uma configuração é melhor que outra?
